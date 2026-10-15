"""Minimal inference script for Swin-Large backbone + ArcFace head.

Usage examples:
  python scripts/infer_swin_arcface.py --checkpoint checkpoints/model.pth --img path/to/image.jpg --device cpu
  python scripts/infer_swin_arcface.py --checkpoint checkpoints/model.pth --img-dir data/faces/ --out-dir out/embs --device 0

Behavior:
- Loads model architecture from models/swin_arcface.build_model (pretrained=False by default).
- Attempts to load weights from a checkpoint file (supports several common dict layouts).
- Produces L2-normalized embeddings from the backbone and saves them as .npy files (or prints them).

This is intentionally minimal and defensive; adapt to your checkpoint naming conventions as needed.
"""
from __future__ import annotations
import argparse
import os
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

try:
    from models.swin_arcface import build_model
except Exception as e:
    raise RuntimeError("Failed to import models.swin_arcface. Make sure the repository root is on PYTHONPATH and dependencies (timm, torch) are installed.") from e


LOG = logging.getLogger("infer_swin_arcface")


def parse_args():
    p = argparse.ArgumentParser(description="Inference for Swin + ArcFace: extract embeddings from images")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint (pth) file")
    p.add_argument("--img", help="Single image path to run inference on")
    p.add_argument("--img-dir", help="Directory of images to run inference on (saves one .npy per image)")
    p.add_argument("--out-dir", help="Directory to save embeddings (.npy). If omitted, prints embeddings to stdout", default=None)
    p.add_argument("--device", default="cpu", help="Device: 'cpu' or GPU index (0,1,...)")
    p.add_argument("--img-size", type=int, default=384, help="Input image size for the backbone")
    p.add_argument("--num-classes", type=int, default=1000, help="Number of classes for ArcFace head if checkpoint contains head state (not used for embeddings)")
    return p.parse_args()


def make_transform(img_size: int):
    # Use simple resize + to tensor + ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def find_state_dict_keys(checkpoint: Dict) -> Dict[str, object]:
    """Return a dict of keys we might load from checkpoint.

    Common patterns supported:
      - checkpoint contains 'backbone' and/or 'arcface_head'
      - checkpoint contains 'state_dict' which may contain prefixed keys ('backbone.' / 'module.backbone.')
      - checkpoint is itself a state_dict for the model
    """
    out = {}
    if not isinstance(checkpoint, dict):
        return out
    if "backbone" in checkpoint or "arcface_head" in checkpoint:
        out.update(checkpoint)
    if "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        if isinstance(sd, dict):
            out["state_dict"] = sd
    # also raw dict
    out["raw"] = checkpoint
    return out


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if any(k.startswith(prefix) for k in state_dict.keys()):
        stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        return stripped
    return state_dict


def load_weights_into_backbone(backbone: torch.nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    keys = find_state_dict_keys(ckpt)

    # Try direct 'backbone' key
    if "backbone" in keys and isinstance(keys["backbone"], dict):
        sd = keys["backbone"]
        backbone.load_state_dict(sd)
        LOG.info("Loaded backbone weights from checkpoint['backbone']")
        return

    # Try arcface style where full model state dict may be present under 'state_dict'
    if "state_dict" in keys:
        sd = keys["state_dict"]
        # common prefixes
        for prefix in ["backbone.", "module.backbone.", "model.backbone."]:
            stripped = strip_prefix_if_present(sd, prefix)
            if stripped is not sd:
                backbone.load_state_dict(stripped)
                LOG.info("Loaded backbone weights from checkpoint['state_dict'] with prefix %s", prefix)
                return
        # fallback: try to load by matching keys (if keys equal)
        try:
            backbone.load_state_dict(sd)
            LOG.info("Loaded backbone weights from checkpoint['state_dict'] (direct) ")
            return
        except Exception:
            LOG.debug("Could not load checkpoint['state_dict'] directly into backbone")

    # Try raw dict (checkpoint itself might be state_dict)
    raw = keys.get("raw")
    if isinstance(raw, dict):
        # try removing common module prefix
        for prefix in ["module.", "model."]:
            stripped = strip_prefix_if_present(raw, prefix)
            if stripped is not raw:
                try:
                    backbone.load_state_dict(stripped)
                    LOG.info("Loaded backbone weights from raw checkpoint with prefix %s", prefix)
                    return
                except Exception:
                    pass
        try:
            backbone.load_state_dict(raw)
            LOG.info("Loaded backbone weights from raw checkpoint (direct)")
            return
        except Exception as e:
            LOG.warning("Failed to load raw checkpoint into backbone: %s", e)

    raise RuntimeError("Could not find suitable backbone state_dict in checkpoint. Inspect your checkpoint layout.")


def image_to_tensor(img_path: str, transform) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    return transform(img)


def compute_embedding(backbone: torch.nn.Module, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    backbone.eval()
    with torch.no_grad():
        x = img_t.unsqueeze(0).to(device)
        feats = backbone(x)  # expect (1, feat_dim)
        if isinstance(feats, tuple) or isinstance(feats, list):
            feats = feats[0]
        feats = feats.detach().cpu().squeeze(0).numpy()
        # L2 normalize
        norm = np.linalg.norm(feats) + 1e-8
        feats = feats / norm
        return feats


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if not args.img and not args.img_dir:
        raise SystemExit("Provide --img or --img-dir")

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        try:
            dev_idx = int(args.device)
            device = torch.device(f"cuda:{dev_idx}" if torch.cuda.is_available() else "cpu")
        except Exception:
            device = torch.device("cpu")

    # Build model architecture (do not download pretrained weights from timm here)
    backbone, arc_head = build_model(num_classes=args.num_classes, pretrained=False, img_size=args.img_size)
    backbone.to(device)

    # Load checkpoint into backbone
    LOG.info("Loading checkpoint: %s", args.checkpoint)
    load_weights_into_backbone(backbone, args.checkpoint)

    transform = make_transform(args.img_size)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    if args.img:
        paths.append(Path(args.img))
    if args.img_dir:
        img_dir = Path(args.img_dir)
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                paths.append(p)

    if len(paths) == 0:
        raise SystemExit("No images found for inference")

    for p in paths:
        try:
            img_t = image_to_tensor(str(p), transform)
            emb = compute_embedding(backbone, img_t, device)
            if out_dir:
                out_path = out_dir / (p.stem + ".npy")
                np.save(out_path, emb)
                LOG.info("Saved embedding: %s", out_path)
            else:
                print(f"{p}: embedding shape={emb.shape}")
                # print first 8 values for quick sanity
                print(emb[:8].tolist())
        except Exception as e:
            LOG.exception("Failed to process %s: %s", p, e)


if __name__ == "__main__":
    main()
