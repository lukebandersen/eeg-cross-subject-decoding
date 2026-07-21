#!/usr/bin/env python
"""Extract ViT-H-14 features for EEG-ImageNet, keeping only categories with <=25% missing
images (70/80 categories, ~3191 images). Matches THINGS: precision='fp32', RAW features."""
import argparse, os
from collections import Counter
import torch
from PIL import Image
import open_clip

MISS_THRESHOLD = 0.25

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth', nargs='+', required=True)
    ap.add_argument('--img_root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=20)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    unique_fnames = set()
    for p in args.pth:
        d = torch.load(p, map_location='cpu', weights_only=False)
        for trial in d['dataset']:
            unique_fnames.add(trial['image'])
    fnames_all = sorted(unique_fnames)

    miss, tot = Counter(), Counter()
    for f in fnames_all:
        w = f.split('_')[0]; tot[w] += 1
        if not os.path.exists(os.path.join(args.img_root, w, f)):
            miss[w] += 1
    kept_wnids = {w for w in tot if (miss[w] / tot[w]) <= MISS_THRESHOLD}
    dropped_wnids = set(tot) - kept_wnids
    print(f"Categories: {len(tot)} total -> keeping {len(kept_wnids)} (<= {MISS_THRESHOLD:.0%} missing), "
          f"dropping {len(dropped_wnids)}")

    fnames, missing_in_kept = [], []
    for f in fnames_all:
        w = f.split('_')[0]
        if w not in kept_wnids:
            continue
        if os.path.exists(os.path.join(args.img_root, w, f)):
            fnames.append(f)
        else:
            missing_in_kept.append(f)
    fnames.sort()
    print(f"Images in kept categories: {len(fnames)} present, {len(missing_in_kept)} residual-missing (skipped)")

    paths = [os.path.join(args.img_root, f.split('_')[0], f) for f in fnames]
    assert all(os.path.exists(p) for p in paths), "a kept path vanished -- abort"
    assert len(paths) == len(fnames), "paths/fnames desync -- abort"

    device = args.device if torch.cuda.is_available() else 'cpu'
    model, preprocess, _ = open_clip.create_model_and_transforms(
        'ViT-H-14', pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
    model.eval()
    print(f"OpenCLIP ViT-H-14 loaded on {device}")

    feats_list = []
    for i in range(0, len(paths), args.batch):
        batch = paths[i:i + args.batch]
        imgs = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in batch]).to(device)
        with torch.no_grad():
            feats_list.append(model.encode_image(imgs).cpu())
        if (i // args.batch) % 20 == 0:
            print(f"  {i}/{len(paths)}")
    img_features = torch.cat(feats_list, dim=0).detach()
    print(f"img_features shape: {img_features.shape}")

    fname_to_idx = {f: i for i, f in enumerate(fnames)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'img_features': img_features, 'fname_to_idx': fname_to_idx,
                'kept_wnids': sorted(kept_wnids), 'dropped_wnids': sorted(dropped_wnids),
                'miss_threshold': MISS_THRESHOLD}, args.out)
    print(f"Saved to {args.out}  ({len(fname_to_idx)} images, {len(kept_wnids)} categories)")

if __name__ == '__main__':
    main()
