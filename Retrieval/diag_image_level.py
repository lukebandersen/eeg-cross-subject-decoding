#!/usr/bin/env python
"""Diagnostic: evaluate the trained sub-01 ATMS model at IMAGE level
(retrieve each test trial's specific image among test-set images).
If image-level >> chance while category-level == chance, the train(per-image)/
eval(per-category) granularity mismatch is the culprit."""
import os, sys, random
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from eeg_encoders import build_encoder, SUBJECT_ID_ENCODERS, NORMALIZE_FEAT_ENCODERS
from eegimagenet_dataset import EEGImageNetDataset

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PTH = ['C:/Users/mwolff3/Desktop/EEG-ImageNet-Dataset/EEG-ImageNet_1.pth',
       'C:/Users/mwolff3/Desktop/EEG-ImageNet-Dataset/EEG-ImageNet_2.pth']
FEATS = '../emb_eeg/eegimagenet_ViT-H-14_features.pt'

import glob
cands = sorted(glob.glob('./models/contrast/ATMS/sub-01/*/best.pth'))
if not cands:
    print("No checkpoint at ./models/contrast/ATMS/sub-01/*/best.pth"); sys.exit(1)
CKPT = cands[-1]
print(f"Using checkpoint: {CKPT}")

test_ds = EEGImageNetDataset(PTH, FEATS, exclude_subject=0, train=False,
                             mode='intra', n_times=250, pad_to=63, seed=42)

model = build_encoder('ATMS', n_chans=63, n_times=250)
sd = torch.load(CKPT, map_location='cpu', weights_only=False)
if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded weights | missing={len(missing)} unexpected={len(unexpected)}")
model.to(DEVICE).eval()

use_sid = 'ATMS' in SUBJECT_ID_ENCODERS
normalize = 'ATMS' in NORMALIZE_FEAT_ENCODERS

per_image_feats = test_ds._per_image_feats
f2i = test_ds._fname_to_idx

eeg_feats, fnames = [], []
with torch.no_grad():
    for i in range(len(test_ds)):
        x, label, _, _, fname, img_feat = test_ds[i]
        x = x.unsqueeze(0).to(DEVICE)
        if use_sid:
            sid = torch.zeros(1, dtype=torch.long, device=DEVICE)
            out = model(x, sid)
        else:
            out = model(x)
        ef = (out[0] if isinstance(out, tuple) else out).float().cpu()
        eeg_feats.append(ef.squeeze(0)); fnames.append(fname)
eeg_feats = torch.stack(eeg_feats)

uniq_fnames = sorted(set(fnames))
fn_to_pool = {fn: j for j, fn in enumerate(uniq_fnames)}
img_pool = torch.stack([per_image_feats[f2i[fn]] for fn in uniq_fnames])
correct_pool_idx = torch.tensor([fn_to_pool[fn] for fn in fnames])
print(f"\nTest trials: {len(fnames)} | unique test images: {len(uniq_fnames)}")

def kway(eeg_f, pool, correct_idx, k, normalize):
    if normalize:
        eeg_f = F.normalize(eeg_f, dim=-1); pool = F.normalize(pool, dim=-1)
    P = pool.size(0); k = min(k, P); all_idx = set(range(P))
    hit = tot = 0
    for i in range(eeg_f.size(0)):
        corr = correct_idx[i].item()
        sel = random.sample(list(all_idx - {corr}), k-1) + [corr]
        logits = eeg_f[i] @ pool[sel].T
        pred = sel[torch.argmax(logits).item()]
        hit += (pred == corr); tot += 1
    return hit / tot

random.seed(0)
print("\n=== IMAGE-LEVEL retrieval (correct image among test images) ===")
for k in [2, 4, 10, 50, 100, 200]:
    acc = kway(eeg_feats, img_pool, correct_pool_idx, k, normalize)
    kk = min(k, img_pool.size(0))
    print(f"  v{k} (eff {kk}-way): {acc:.4f}   chance={1/kk:.4f}")
