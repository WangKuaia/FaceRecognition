"""Inference helper using InsightFace (FaceAnalysis) to extract face embeddings and compare similarity.

Usage:
    pip install -r requirements.txt
    python scripts/infer_insightface.py --img1 path/to/face1.jpg --img2 path/to/face2.jpg --device cpu

Notes:
- This script uses the `insightface` Python package (it will download required recognition/detection models automatically).
- For GPU run with `--device 0` (or another GPU index). Use `--device cpu` to force CPU.
"""

import argparse
import sys
import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    print("Failed to import insightface. Please install requirements: pip install insightface")
    raise


def prepare_face_analyzer(device: str = "cpu") -> FaceAnalysis:
    # device: 'cpu' or '0','1', ... for GPU index
    fa = FaceAnalysis(allowed_modules=["detection", "recognition"])  # detection + recognition
    if device == "cpu":
        ctx_id = -1
    else:
        try:
            ctx_id = int(device)
        except:
            ctx_id = 0
    print(f"Preparing InsightFace FaceAnalysis (ctx_id={ctx_id}) ...")
    fa.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return fa


def get_first_face_embedding(fa: FaceAnalysis, img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    faces = fa.get(img)
    if len(faces) == 0:
        raise RuntimeError(f"No face detected in image: {img_path}")
    emb = faces[0].embedding  # 512-d float32
    emb = np.asarray(emb, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    parser = argparse.ArgumentParser(description="InsightFace inference: extract embeddings and compare two images")
    parser.add_argument("--img1", required=True, help="Path to first image")
    parser.add_argument("--img2", required=True, help="Path to second image")
    parser.add_argument("--device", default="cpu", help="Device: 'cpu' or GPU index (0,1,...)")
    args = parser.parse_args()

    fa = prepare_face_analyzer(args.device)
    emb1 = get_first_face_embedding(fa, args.img1)
    emb2 = get_first_face_embedding(fa, args.img2)
    sim = float(np.dot(emb1, emb2))
    print(f"Embedding dim: {emb1.shape[0]}")
    print(f"Cosine similarity (dot of normalized embeddings): {sim:.6f}")


if __name__ == "__main__":
    main()
