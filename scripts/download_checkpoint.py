"""Download external checkpoints (Hugging Face, Google Drive, direct URL).

Usage examples:
  python scripts/download_checkpoint.py --hf repo_id model.pth --out-dir checkpoints
  python scripts/download_checkpoint.py --gdrive 1a2B3c4D5e6F --out checkpoints/model.pth
  python scripts/download_checkpoint.py --url https://example.com/model.pth --out checkpoints/model.pth

Notes:
- Hugging Face: requires `huggingface_hub` (optional token via --token or HF_HOME env vars).
- Google Drive: uses `gdown` (handles shared file ids/urls).
- Direct URL: uses requests and streams with tqdm progress.
- Baidu Netdisk: not automated here; follow printed instructions (use browser or aria2/BaiduPCS-Go).

This script was added to help download large external checkpoint files without committing them to git.
"""
from __future__ import annotations
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except Exception:
    print("Please install requests and tqdm: pip install requests tqdm")
    raise

# optional imports
try:
    import gdown
except Exception:
    gdown = None

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_http(url: str, out_path: Path, chunk_size: int = 32768) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with out_path.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=out_path.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return out_path


def download_gdrive(gdrive_id_or_url: str, out_path: Path) -> Path:
    if gdown is None:
        raise RuntimeError("gdown is not installed. Run: pip install gdown")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # gdown accepts either id or full url
    url = gdrive_id_or_url
    # if user passed a direct id, convert to url
    if not (url.startswith("http") or url.startswith("https")) and len(url) > 10:
        url = f"https://drive.google.com/uc?id={url}"
    print(f"Downloading from Google Drive: {url}")
    # gdown.download will return the output path
    downloaded = gdown.download(url, str(out_path), quiet=False)
    if downloaded is None:
        raise RuntimeError("gdown failed to download the file")
    return Path(downloaded)


def download_hf(repo_id: str, filename: str, out_path: Path, token: str | None = None) -> Path:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is not installed. Run: pip install huggingface-hub")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from Hugging Face repo: {repo_id} -> {filename}")
    # hf_hub_download returns local path; we copy to out_path if names differ
    local = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    if Path(local) == out_path:
        return out_path
    shutil.copy(local, out_path)
    return out_path


def ensure_executable(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def download_baidu_instructions(share_url: str | None = None) -> None:
    print("\nBaidu Netdisk download is not automated by this script due to authentication and anti-bot protections.")
    print("Options to download from Baidu:")
    print("  1) Open the share URL in a browser and download manually.")
    print("  2) Use BaiduPCS-Go (https://github.com/qjfoidnh/BaiduPCS-Go) to download from the command line.")
    print("     Example: ./BaiduPCS-Go d <share_id> <save_path>")
    print("  3) If you can create a direct HTTP link or upload the file to Google Drive/Hugging Face, use that link with this script.")
    if share_url:
        print(f"Share URL: {share_url}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download external model checkpoints")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Direct HTTP/HTTPS URL to download")
    group.add_argument("--gdrive", help="Google Drive file id or share URL")
    group.add_argument("--hf", nargs=2, metavar=("REPO_ID", "FILENAME"), help="Hugging Face repo id and filename")
    group.add_argument("--baidu", help="Baidu Netdisk share URL (instructions only)")

    parser.add_argument("--out", required=True, help="Output path (file) or directory if --out-dir used")
    parser.add_argument("--out-dir", help="Output directory (filename will be taken from source) — mutually exclusive with --out")
    parser.add_argument("--token", help="Hugging Face token (if needed)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = parser.parse_args(argv)

    out = Path(args.out) if args.out else None
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = None

    try:
        if args.url:
            # if out is directory, derive filename from url
            if out and out.is_dir():
                filename = os.path.basename(args.url.split("?")[0]) or "downloaded.bin"
                out = out / filename
            elif out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = os.path.basename(args.url.split("?")[0]) or "downloaded.bin"
                out = out_dir / filename
            if out.exists() and not args.force:
                print(f"File already exists: {out} (use --force to overwrite)")
                return 0
            downloaded = download_http(args.url, out)
        elif args.gdrive:
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                # try to infer filename using gdown.info? just set generic name
                out = out_dir / "gdrive_download"
            if out.exists() and not args.force:
                print(f"File already exists: {out} (use --force to overwrite)")
                return 0
            downloaded = download_gdrive(args.gdrive, out)
        elif args.hf:
            repo_id, filename = args.hf
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / filename
            if out.exists() and not args.force:
                print(f"File already exists: {out} (use --force to overwrite)")
                return 0
            downloaded = download_hf(repo_id, filename, out, token=args.token)
        elif args.baidu:
            download_baidu_instructions(args.baidu)
            return 0
        else:
            parser.print_help()
            return 1

        print(f"Downloaded to: {downloaded}")
        size = downloaded.stat().st_size
        print(f"Size: {size} bytes")
        print(f"SHA256: {sha256_of_file(downloaded)}")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
