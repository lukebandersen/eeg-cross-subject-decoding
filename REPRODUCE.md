# REPRODUCE.md

Exact steps from a fresh clone to the paper's numbers, using the repo's real
layout (core code in `Retrieval/`, orchestration at root). Paths are given
literally where they matter.

---

## 0. Requirements

- CUDA GPU (developed on RTX 4080, 16 GB).
- Python 3.11, conda recommended.
- ~50 GB free disk for data + checkpoints.
- THINGS-EEG2 dataset (step 2).

---

## 1. Environment

```bash
conda create -n BCI python=3.11 -y
conda activate BCI
pip install -r requirements.txt
```

Pins that matter:
- `braindecode==0.8` - the specialist wrappers use 0.8-era args
  (`in_chans`, `n_classes`, `input_window_samples`). Do NOT upgrade to 1.x.
- `torch==2.5`, CUDA 12.x.

Check the GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. Data placement

Place THINGS-EEG2 so the scripts find it at these repo-relative paths (these are
the defaults the code uses):

```
./EEG_Image_decode/Preprocessed_data_250Hz/sub-01 ... sub-10   preprocessed EEG
./image_set/training_images/                                    stimuli (train)
./image_set/test_images/                                        stimuli (test)
./emb_eeg/ViT-H-14_features_train.pt                            CLIP features
./emb_eeg/ViT-H-14_features_test.pt
```

(Different locations? pass `--data_path`, `--img_dir_training`,
`--img_dir_test`, `--clip_features_dir` explicitly.)

---

## 3. Verify before computing

```bash
bash run_all_gates.sh
```

Confirms env, GPU, that every encoder instantiates and trains 2 epochs on real
data, and that CBraMod's pretrained weights actually loaded (not random). Expect
`ALL GATES GREEN`. A CBraMod SKIP just means you have not run step 5 yet - fine
for the core ATMS-vs-LaBraM result.

---

## 4. The core result (Paper 1): ATMS vs LaBraM

Two encoders, two protocols, ten subjects. Note the LOSO invocation: pass ALL
ten subjects to `--subjects` and hold one out with `--test_subjects`.

```bash
# intra-subject (one model per subject)
for s in $(seq -f "sub-%02g" 1 10); do
  python Retrieval/train_unified.py --encoder_type ATMS --mode intra \
    --subjects $s --epochs 300 --seed 42 \
    --data_path ./EEG_Image_decode/Preprocessed_data_250Hz \
    --img_dir_training ./image_set/training_images \
    --img_dir_test ./image_set/test_images \
    --clip_features_dir ./emb_eeg --n_chans 63 --n_times 250
done

# leave-one-subject-out (hold out $s, train on the other nine)
for s in $(seq -f "sub-%02g" 1 10); do
  python Retrieval/train_unified.py --encoder_type ATMS --mode loso \
    --subjects sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10 \
    --test_subjects $s --epochs 300 --seed 42 \
    --data_path ./EEG_Image_decode/Preprocessed_data_250Hz \
    --img_dir_training ./image_set/training_images \
    --img_dir_test ./image_set/test_images \
    --clip_features_dir ./emb_eeg --n_chans 63 --n_times 250
done
```

Repeat with `--encoder_type LaBraM_ATMS`, then seeds 43 and 44.

**Retention** = LOSO v200 top-1 / intra v200 top-1, per encoder.
Expected three-seed means: ATMS 33.7% +/- 3.1, LaBraM 93.3% +/- 1.2.

> **Why 300 epochs.** Early stopping (patience 10) lets each encoder stop at its
> own convergence: ATMS ~40, LaBraM ~22, EEGConformer ~200. A fixed 40-cap
> converges ATMS/LaBraM but undertrains EEGConformer (v200 0.010 at ep40 vs 0.075
> at ep200). Generous cap + early stopping = fixed protocol, no hidden per-encoder
> tuning.

---

## 5. The full panel (all six encoders)

Use the sweep scripts instead of hand-looping. They cover the whole matrix
(encoders x modes x subjects) with resume support.

```bash
bash setup_cbramod.sh          # one-time: 2nd foundation model
bash local_sweep.sh            # everything outstanding (resumable, Ctrl-C safe)
bash local_sweep.sh --status   # progress table
```

`experiment_matrix.tsv` is the definitive run list. On a Slurm cluster use
`retrieval_sweep.sbatch` (array job) - the folds run in parallel, which
EEGConformer LOSO needs (~130 GPU-hours serially).

---

## 6. Classification transfer probe

Tests task-generality. Reuses the intra checkpoints from step 4.

```bash
python probe_loso_classify.py --encoder ATMS        --subjects 1-10 --out probe_loso_ATMS.csv
python probe_loso_classify.py --encoder LaBraM_ATMS --subjects 1-10 --out probe_loso_LaBraM.csv
```

Three baseline-fair metrics (chance-corrected retention, normalized drop, rank
transfer). Result: significant but attenuated (cc-retention 0.234 vs 0.154,
p=0.012), null on rank transfer.

---

## 7. Where results land

```
outputs/retrieval/{encoder}/{subject}/{timestamp}/{encoder}_{mode}_{subject}.csv
models/contrast/{encoder}/{subject}/{timestamp}/best.pth
```

Each CSV has per-epoch train/val loss and v2..v200 accuracies; the final row's
v200 top-1 is the number used for retention. Collate with `aggregate_all.py`.

---

## Common issues

- **`torch.cat(): expected a non-empty list of Tensors`** (LOSO): you passed a
  single subject to `--subjects` in loso mode. LOSO needs the full pool in
  `--subjects` and the held-out one in `--test_subjects`.
- **`ModuleNotFoundError: models.loss`** after adding CBraMod: run from the repo
  root; the CBraMod wrapper isolates its `models/` import but relies on cwd.
- **`EEGNetv4` ImportError**: braindecode >=1.x renamed it. Pin 0.8 (recommended)
  or run `patch_braindecode_compat.py`.
- **CBraMod trains but numbers look random**: weights did not load. `run_all_gates.sh`
  gate 3 checks this; do not trust CBraMod numbers if it fails.
