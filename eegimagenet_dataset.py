"""EEG-ImageNet dataset (70-category subset), category-level retrieval.
Mirrors THINGS EEGDataset for the Retrieval engine:
  self.img_features  : [70,1024] category-prototype pool (mean of each category's images);
                       label (category index) indexes into it for eval retrieval.
  self.text_features : placeholder zeros [70,1024].
  __getitem__ returns (x_eeg, label, "", txt_feat_row, fname, img_feat_row); img_feat_row is
                       the trial's SPECIFIC image embedding (training loss), label is CATEGORY idx.
modes: loso (subject != exclude = train) / intra (single subject, split by test_frac)."""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class EEGImageNetDataset(Dataset):
    def __init__(self, pth_paths, features_path,
                 exclude_subject=None, train=True, mode='loso',
                 n_times=250, crop=(40, 440), test_frac=0.2, seed=42, pad_to=None, drop_channels=None):
        self.train = train; self.n_times = n_times; self.crop = crop; self.pad_to = pad_to; self.drop_channels = drop_channels

        blob = torch.load(features_path, map_location='cpu', weights_only=False)
        per_image_feats = blob['img_features']
        fname_to_idx = blob['fname_to_idx']

        trials = []
        for p in pth_paths:
            d = torch.load(p, map_location='cpu', weights_only=False)
            trials.extend(d['dataset'])
        n_before = len(trials)
        trials = [t for t in trials if t['image'] in fname_to_idx]
        print(f"Kept {len(trials)}/{n_before} trials (image in feature bank)")

        self._labels_vocab = sorted({t['label'] for t in trials})
        self._label_to_int = {lab: i for i, lab in enumerate(self._labels_vocab)}
        n_cat = len(self._labels_vocab)

        dim = per_image_feats.shape[1]
        sums = torch.zeros(n_cat, dim); counts = torch.zeros(n_cat)
        seen_img = {}
        for t in trials:
            lab = self._label_to_int[t['label']]; fn = t['image']
            if fn in seen_img:
                continue
            seen_img[fn] = lab
            sums[lab] += per_image_feats[fname_to_idx[fn]]
            counts[lab] += 1
        counts = counts.clamp(min=1).unsqueeze(1)
        self.img_features = (sums / counts)
        self.text_features = torch.zeros(n_cat, dim)
        assert self.img_features.shape[0] == n_cat, "category pool size mismatch"
        print(f"Category pool: {self.img_features.shape}  ({n_cat} categories)")

        self._per_image_feats = per_image_feats
        self._fname_to_idx = fname_to_idx

        if mode == 'loso':
            if exclude_subject is None:
                raise ValueError("loso mode requires exclude_subject")
            self.trials = [t for t in trials if (t['subject'] != exclude_subject) == train]
        elif mode == 'intra':
            if exclude_subject is None:
                raise ValueError("intra mode: pass target subject as exclude_subject")
            sub_trials = [t for t in trials if t['subject'] == exclude_subject]
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(sub_trials), generator=g).tolist()
            n_test = int(len(sub_trials) * test_frac)
            test_set = set(perm[:n_test])
            self.trials = [t for i, t in enumerate(sub_trials) if (i not in test_set) == train]
        else:
            raise ValueError(f"unknown mode {mode}")

        print(f"EEGImageNet[{mode}/{'train' if train else 'test'}] "
              f"subject={exclude_subject}: {len(self.trials)} trials, {n_cat} categories")

    def __len__(self):
        return len(self.trials)

    def _prep_eeg(self, eeg):
        a, b = self.crop
        x = eeg[:, a:b].float()
        if self.drop_channels is not None:
            keep = [i for i in range(x.shape[0]) if i not in self.drop_channels]
            x = x[keep]
        # z-score each channel (EEG-ImageNet is in volts, std ~9e-6; encoders expect ~unit variance)
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        x = (x - mu) / sd
        if x.shape[1] != self.n_times:
            x = F.interpolate(x.unsqueeze(0), size=self.n_times,
                              mode='linear', align_corners=False).squeeze(0)
        if self.pad_to is not None and x.shape[0] < self.pad_to:
            pad = torch.zeros(self.pad_to - x.shape[0], x.shape[1])
            x = torch.cat([x, pad], dim=0)
        return x

    def __getitem__(self, index):
        t = self.trials[index]
        x = self._prep_eeg(t['eeg_data'])
        label = self._label_to_int[t['label']]
        fname = t['image']
        img_feat = self._per_image_feats[self._fname_to_idx[fname]]
        txt_feat = self.text_features[label]
        return x, label, "", txt_feat, fname, img_feat
