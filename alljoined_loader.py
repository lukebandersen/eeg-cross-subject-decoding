"""
Alljoined1 loader for the cross-subject asymmetry project.

Reads the Alljoined/05_125 parquet shards, builds an image-level retrieval
dataset for ONE subject:
  - groups trials by coco_id
  - averages EEG across the (up to 4) repeats of each image
  - applies per-channel z-score (fixes the volt-scale issue: values are ~1e-6)
  - keeps the dataset's own train/test split so the held-out images never leak

Schema confirmed from the files:
  columns = ['EEG','subject_id','session','block','trial','73k_id','coco_id','curr_time']
  EEG shape per row = (64, 334)  # channels-first, 512 Hz, ~650 ms window

Scaling to all 8 subjects later = call load_subject() in a loop, or set
subject_id=None in _load_split to keep everyone.
"""

import glob
import os
import numpy as np
import pyarrow.parquet as pq

DATA_DIR = r"C:/Users/mwolff3/Desktop/alljoined1/data"
N_CHANNELS = 64
N_TIMES = 334


def _read_shards(split):
    """Load all rows from the train or test parquet shards as a dict of arrays."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No {split} parquet files under {DATA_DIR}")
    tables = [pq.read_table(f) for f in files]
    # concatenate columns across shards
    cols = {}
    for name in tables[0].column_names:
        parts = [t.column(name).to_pylist() for t in tables]
        cols[name] = [x for part in parts for x in part]
    return cols


def _to_eeg_array(eeg_list):
    """Turn the list-of-lists EEG column into a float32 (N, 64, 334) array."""
    arr = np.asarray(eeg_list, dtype=np.float32)
    # sanity: each row must be (64, 334)
    if arr.ndim != 3 or arr.shape[1:] != (N_CHANNELS, N_TIMES):
        raise ValueError(f"Unexpected EEG shape {arr.shape}, expected (N,{N_CHANNELS},{N_TIMES})")
    return arr


def _group_and_average(eeg, coco_ids):
    """Average EEG across repeats of the same coco_id.

    Returns:
        img_ids: (M,) unique coco_ids
        img_eeg: (M, 64, 334) mean EEG per image
        n_reps:  (M,) how many trials were averaged for each image
    """
    coco_ids = np.asarray(coco_ids)
    uniq = np.unique(coco_ids)
    img_eeg = np.zeros((len(uniq), N_CHANNELS, N_TIMES), dtype=np.float32)
    n_reps = np.zeros(len(uniq), dtype=np.int32)
    for i, cid in enumerate(uniq):
        mask = coco_ids == cid
        img_eeg[i] = eeg[mask].mean(axis=0)
        n_reps[i] = int(mask.sum())
    return uniq, img_eeg, n_reps


def _zscore_per_channel(eeg, mean=None, std=None):
    """Per-channel z-score. Fit stats on train, reuse them on test.

    eeg: (N, 64, 334). Normalizes each channel across (trials, time).
    """
    if mean is None or std is None:
        # stats per channel, pooled over trials and time
        mean = eeg.mean(axis=(0, 2), keepdims=True)   # (1,64,1)
        std = eeg.std(axis=(0, 2), keepdims=True)      # (1,64,1)
        std = np.where(std < 1e-12, 1.0, std)          # guard against dead channels
    return (eeg - mean) / std, mean, std


def _load_split(split, subject_id):
    cols = _read_shards(split)
    eeg = _to_eeg_array(cols["EEG"])
    sid = np.asarray(cols["subject_id"])
    coco = np.asarray(cols["coco_id"])
    if subject_id is not None:
        keep = sid == subject_id
        eeg, coco = eeg[keep], coco[keep]
        if len(eeg) == 0:
            raise ValueError(f"No trials for subject_id={subject_id} in {split}")
    return eeg, coco


def load_subject(subject_id=6, verbose=True):
    """Build the image-level retrieval dataset for one subject.

    Returns a dict with train/test EEG grouped and averaged per image,
    z-scored with train-fit stats, plus the coco_ids you'll need to fetch
    CLIP features for.
    """
    # --- train ---
    tr_eeg, tr_coco = _load_split("train", subject_id)
    tr_ids, tr_img_eeg, tr_reps = _group_and_average(tr_eeg, tr_coco)
    tr_img_eeg, mean, std = _zscore_per_channel(tr_img_eeg)

    # --- test (reuse train stats, never refit) ---
    te_eeg, te_coco = _load_split("test", subject_id)
    te_ids, te_img_eeg, te_reps = _group_and_average(te_eeg, te_coco)
    te_img_eeg, _, _ = _zscore_per_channel(te_img_eeg, mean, std)

    if verbose:
        print(f"subject {subject_id}")
        print(f"  train: {len(tr_eeg)} trials -> {len(tr_ids)} unique images "
              f"(avg {tr_reps.mean():.1f} reps/image)")
        print(f"  test:  {len(te_eeg)} trials -> {len(te_ids)} unique images "
              f"(avg {te_reps.mean():.1f} reps/image)")
        overlap = set(tr_ids) & set(te_ids)
        print(f"  train/test image overlap: {len(overlap)} (should be 0)")
        print(f"  EEG after z-score: mean {tr_img_eeg.mean():.3f} std {tr_img_eeg.std():.3f}")

    return {
        "train_eeg": tr_img_eeg,          # (M_tr, 64, 334)
        "train_coco_ids": tr_ids,         # (M_tr,)
        "test_eeg": te_img_eeg,           # (M_te, 64, 334)
        "test_coco_ids": te_ids,          # (M_te,)
        "zscore_mean": mean,
        "zscore_std": std,
    }


if __name__ == "__main__":
    # gate-test default: subject 6 (present in the sample you inspected)
    data = load_subject(subject_id=6)
    # quick shape report for the next step (CLIP feature matching)
    print("\nready for retrieval:")
    print("  train EEG:", data["train_eeg"].shape)
    print("  test EEG:", data["test_eeg"].shape)
    print("  unique test images (retrieval pool size):", len(data["test_coco_ids"]))
