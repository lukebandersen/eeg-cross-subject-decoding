#!/usr/bin/env python
"""
rank_reconstructions.py -- find the best reconstruction to put on the poster.

Scores every generated image against its ground-truth counterpart with CLIP
cosine similarity, ranks them, and copies the top N (generated + ground truth,
side by side) into ./poster_candidates/ ready to upload.

WHY BY METRIC, NOT BY EYE: "highest CLIP similarity of 200" is a criterion you
can state and defend at a poster session. "It looked best" is not.

NOTE ON THE NUMBER: this reports per-image CLIP *cosine similarity*, which is a
DIFFERENT quantity from the 0.8009 in the paper's Table 3. That table reports
CLIP two-way identification ACCURACY (chance = 0.50) averaged over 200 images.
Do not print this script's per-image score as if it were that 0.80. Caption the
poster figure with the aggregate from the paper and label the image as the
best-scoring example. See --caption output at the end.

USAGE (from repo root):
    python rank_reconstructions.py
    python rank_reconstructions.py --top 5
    python rank_reconstructions.py --gt-dir image_set/test_images
"""
import argparse
import os
import shutil
import sys

BENCH = ("Generation/outputs/benchmark/sub-08/07-01_02-06/"
         "generated_imgs_encoder_only/sub-08")


def find_image(folder):
    """First image file inside a folder (concept dirs hold one generation)."""
    if not os.path.isdir(folder):
        return None
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            return os.path.join(folder, f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=BENCH,
                    help="dir of per-concept generated images (encoder-only)")
    ap.add_argument("--gt-dir", default="image_set/test_images",
                    help="ground-truth THINGS test images")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default="poster_candidates")
    args = ap.parse_args()

    if not os.path.isdir(args.gen_dir):
        sys.exit(f"generated dir not found: {args.gen_dir}")
    if not os.path.isdir(args.gt_dir):
        sys.exit(f"ground-truth dir not found: {args.gt_dir}\n"
                 "Pass the right path with --gt-dir.")

    try:
        import torch
        import open_clip
        from PIL import Image
    except ImportError as e:
        sys.exit(f"missing dependency: {e}\n"
                 "This needs torch, open_clip_torch and pillow (already in the "
                 "BCI env used for feature extraction).")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading CLIP ViT-H-14 on {dev} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=dev)
    model.eval()

    concepts = sorted(d for d in os.listdir(args.gen_dir)
                      if os.path.isdir(os.path.join(args.gen_dir, d)))
    print(f"scoring {len(concepts)} concepts ...")

    # map ground-truth folders by a normalized concept name
    gt_map = {}
    for d in os.listdir(args.gt_dir):
        p = os.path.join(args.gt_dir, d)
        if os.path.isdir(p):
            # THINGS test dirs look like "00012_antelope"; key on the tail
            key = d.split("_", 1)[-1].lower()
            gt_map[key] = p

    rows, missing = [], []
    with torch.no_grad():
        for i, c in enumerate(concepts, 1):
            gen_path = find_image(os.path.join(args.gen_dir, c))
            gt_dir = gt_map.get(c.lower())
            gt_path = find_image(gt_dir) if gt_dir else None
            if not gen_path or not gt_path:
                missing.append(c)
                continue
            try:
                a = preprocess(Image.open(gen_path).convert("RGB")).unsqueeze(0).to(dev)
                b = preprocess(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(dev)
                fa = model.encode_image(a)
                fb = model.encode_image(b)
                fa = fa / fa.norm(dim=-1, keepdim=True)
                fb = fb / fb.norm(dim=-1, keepdim=True)
                sim = float((fa @ fb.T).item())
                rows.append((sim, c, gen_path, gt_path))
            except Exception as ex:
                missing.append(f"{c} ({type(ex).__name__})")
            if i % 25 == 0:
                print(f"  {i}/{len(concepts)}")

    if not rows:
        sys.exit("No pairs scored. Check --gt-dir matches the concept names.")

    rows.sort(reverse=True)
    print("\n" + "=" * 62)
    print(f" TOP {args.top} RECONSTRUCTIONS BY CLIP COSINE SIMILARITY")
    print("=" * 62)
    for sim, c, _g, _t in rows[:args.top]:
        print(f"  {sim:.4f}   {c}")
    print("-" * 62)
    print(f"  scored {len(rows)}/{len(concepts)}   "
          f"median {sorted(r[0] for r in rows)[len(rows)//2]:.4f}   "
          f"worst {rows[-1][0]:.4f}")
    if missing:
        print(f"  unmatched: {len(missing)} (e.g. {', '.join(missing[:3])})")

    os.makedirs(args.out, exist_ok=True)
    for rank, (sim, c, gen_path, gt_path) in enumerate(rows[:args.top], 1):
        shutil.copy(gen_path,
                    os.path.join(args.out, f"{rank:02d}_{c}_RECON_{sim:.3f}.png"))
        shutil.copy(gt_path,
                    os.path.join(args.out, f"{rank:02d}_{c}_TRUTH.jpg"))
    print(f"\n  copied top {args.top} pairs -> {args.out}/")

    best = rows[0]
    print("\n" + "=" * 62)
    print(" SUGGESTED POSTER CAPTION (honest about which number is which)")
    print("=" * 62)
    print(f'  "Zero-shot reconstruction from EEG (subject 08, best-scoring of')
    print(f'   200 test concepts: {best[1]}). Across all 200 images the')
    print( '   encoder-only pipeline reaches CLIP two-way identification of')
    print( '   0.80 and AlexNet(5) of 0.84 (chance = 0.50), confirming the')
    print( '   embeddings capture genuine visual content."')
    print("=" * 62)


if __name__ == "__main__":
    main()
