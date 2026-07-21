"""
Alljoined1 dataset for Retrieval/train_unified.py.

Matches the six-tuple __getitem__ contract used by EEGImageNetDataset:
    (x_eeg, label, "", txt_feat, fname, img_feat)

Key differences from the EEG-ImageNet version, all in Alljoined's favor:
  - per-IMAGE features (not category prototypes): real image-level retrieval
  - label indexes the unique-image pool, so evaluate() does N-way image retrieval
  - EEG already channels-first (64, 334); z-scored per channel using TRAIN stats

Feature source: alljoined_ViT-H-14_features.pt (built by extract_alljoined_clip.py),
a dict {'coco_ids': LongTensor, 'img_features': FloatTensor (M,1024)}, RAW/unnormalized
to match the THINGS pipeline (the contrastive loss normalizes).

Retrieval pool contract: self.img_features is the (P, 1024) pool of unique images
in THIS split, and label i means "trial's true image is pool row i". train_unified's
engine reads self.img_features as the candidate pool, same as EEG-ImageNet.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

import alljoined_loader as L


class AlljoinedDataset(Dataset):
    def __init__(self, subject_id, features_path, train=True, n_times=250,
                 pooled=False,
                 pad_to=None, drop_channels=None, verbose=True):
        """
        subject_id    : which Alljoined subject (gate test uses one)
        features_path : path to alljoined_ViT-H-14_features.pt
        train         : train split vs test split
        n_times       : crop/pad time dim to this (dataset native = 334)
        pad_to        : pad channel dim up to this (ATMS wants 63 for THINGS; here
                        Alljoined is 64 native, so pad_to is normally None)
        drop_channels : indices to drop (for LaBraM montage mapping); None = keep 64
        """
        if pooled:
            data = L.load_pooled(test_subject=subject_id, verbose=verbose)
        else:
            data = L.load_subject(subject_id=subject_id, verbose=verbose)
        eeg = data["train_eeg"] if train else data["test_eeg"]           # (N,64,334)
        coco_ids = data["train_coco_ids"] if train else data["test_coco_ids"]  # (N,)

        # load the ViT-H-14 features and align to this split's coco_ids
        feat_blob = torch.load(features_path, weights_only=False)
        feat_ids = feat_blob["coco_ids"].numpy()
        feat_mat = feat_blob["img_features"].float()                     # (M,1024)
        id_to_row = {int(cid): r for r, cid in enumerate(feat_ids)}

        # keep only trials whose image has a feature (all should, but be safe)
        keep = np.array([int(c) in id_to_row for c in coco_ids])
        if not keep.all():
            missing = int((~keep).sum())
            print(f"[warn] {missing} trials dropped: image feature missing")
        eeg = eeg[keep]
        coco_ids = np.asarray(coco_ids)[keep]

        # build the per-image feature POOL for this split, in trial order.
        # each unique image = one pool row; label points a trial at its image.
        uniq_ids = list(dict.fromkeys(int(c) for c in coco_ids))  # stable unique
        pool_index = {cid: i for i, cid in enumerate(uniq_ids)}
        self.img_features = torch.stack(
            [feat_mat[id_to_row[cid]] for cid in uniq_ids])        # (P,1024)
        # placeholder text features (image-only retrieval; loss ignores these),
        # pool-shaped to match img_features like eegimagenet_dataset.py
        self.text_features = torch.zeros_like(self.img_features)
        self.labels = torch.tensor(
            [pool_index[int(c)] for c in coco_ids], dtype=torch.long)  # (N,)

        # --- channel montage handling (mirror EEG-ImageNet path) ---
        if drop_channels is not None:
            keep_ch = [i for i in range(eeg.shape[1]) if i not in set(drop_channels)]
            eeg = eeg[:, keep_ch, :]

        # --- time crop/pad ---
        if eeg.shape[2] > n_times:
            eeg = eeg[:, :, :n_times]
        elif eeg.shape[2] < n_times:
            pad = np.zeros((eeg.shape[0], eeg.shape[1], n_times - eeg.shape[2]),
                           dtype=eeg.dtype)
            eeg = np.concatenate([eeg, pad], axis=2)

        # --- channel pad (ATMS on THINGS expects 63; Alljoined native 64) ---
        if pad_to is not None and eeg.shape[1] < pad_to:
            pad = np.zeros((eeg.shape[0], pad_to - eeg.shape[1], eeg.shape[2]),
                           dtype=eeg.dtype)
            eeg = np.concatenate([eeg, pad], axis=1)

        self.data = torch.from_numpy(eeg).float()                   # (N,C,T)
        # text features: none for COCO image retrieval; feed zeros of matching width
        self._txt = torch.zeros(feat_mat.shape[1], dtype=torch.float32)

        if verbose:
            print(f"Alljoined {'train' if train else 'test'}  "
                  f"EEG {tuple(self.data.shape)}  pool {tuple(self.img_features.shape)}  "
                  f"(retrieval is {self.img_features.shape[0]}-way)")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        label = self.labels[index]
        img_feat = self.img_features[label]
        # six-tuple contract: (x_eeg, label, text_str, txt_feat, fname, img_feat)
        txt_feat = self.text_features[label]
        return x, label, "", txt_feat, "", img_feat
