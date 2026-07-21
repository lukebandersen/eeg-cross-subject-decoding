# Cross-Subject Generalization in EEG-to-Image Decoding

Specialized encoders overfit to subjects; foundation encoders hold. This
repository contains the code, experiments, and manuscripts for an NSF REU
project studying how EEG-to-image decoders generalize across people, and what
that implies for how the field should measure cross-subject performance.

Built on the EEG_Image_decode / ATM codebase (see `LICENSE`); the cross-subject
comparison, the encoder panel, the second foundation model (CBraMod), and the
experiment tooling are this project's additions.

**Core finding.** On EEG-to-image retrieval (THINGS-EEG2), a specialized
from-scratch encoder (ATMS, 3.2M params) attains higher within-subject accuracy
but collapses across subjects, retaining only 33.7% of its performance. A
fine-tuned foundation encoder (LaBraM, 6.4M params) retains 93.3%. The two are
statistically *tied* on raw cross-subject accuracy, so the finding is the
**degradation asymmetry**, not a raw ranking, and it is why we argue
cross-subject EEG work should report chance-corrected **retention** rather than
raw accuracy.

---

## Start here

| If you want to... | Go to |
|---|---|
| Understand the finding | the Paper 1 manuscript (`root.tex` / its PDF) |
| Reproduce results | `REPRODUCE.md` |
| Run the sweep | `run_all_gates.sh` then `local_sweep.sh` (or `retrieval_sweep.sbatch` on a cluster) |
| See the model + training code | `Retrieval/` |
| See the full project story | the technical report + `project_journal.md` |

---

## Layout (what's actually where)

The core decoding code lives in **`Retrieval/`**. The repo root holds the
orchestration scripts, dataset loaders, and aggregation tools. Several top-level
directories (`Generation/`, `LaBraM/`, `EEG-preprocessing/`, `MEG-preprocessing/`)
are inherited from the upstream codebase and are not central to the cross-subject
study.

```
Retrieval/                    <- THE CORE CODE
  train_unified.py              main entry point (intra / loso / joint)
  eeg_encoders.py               encoder registry (ATMS, LaBraM, EEGNet,
                                Conformer, ShallowFBCSPNet, CBraMod, ...)
  cbramod_encoder.py            2nd foundation model (standalone CBraMod wrapper)
  labram_encoder.py             LaBraM wrapper
  retrieval_engine.py           contrastive train loop + v2..v200 retrieval
  diag_image_level.py           image-level diagnostic (used for the negatives)

(root) experiment orchestration
  run_all_gates.sh              pre-flight verification - RUN THIS FIRST
  local_sweep.sh                run the full matrix on one GPU (resumable)
  retrieval_sweep.sbatch        Slurm array version (cluster)
  setup_cbramod.sh              one-time CBraMod repo + weights setup
  run_*.sh                      per-study sweep launchers (intra/loso/seed43/44)
  aggregate_*.py                collate per-run CSVs into summary tables

(root) data + features
  eegdatasets.py                THINGS-EEG2 dataset + loaders
  alljoined_loader.py, alljoined_dataset.py, extract_alljoined_clip.py
                                Alljoined (documented negative)
  eegimagenet_dataset.py, extract_eegimagenet_clip.py, wnids.txt
                                EEG-ImageNet (documented negative)
  emb_eeg/                      pre-extracted CLIP image features (.pt)
  image_set/                    THINGS stimuli (training_images / test_images)
  EEG_Image_decode/             preprocessed EEG (Preprocessed_data_250Hz/)

(root) probes + setup
  probe_transfer.py             frozen-encoder classification transfer probe
  probe_loso_classify.py        LOSO-trained classification probe
  patch_braindecode_compat.py, patch_register_cbramod.py, fix_cbramod_import.py
                                environment/compat helpers
  requirements.txt, environment.yml, setup.sh

(root) outputs (git-ignored)
  outputs/retrieval/            per-run result CSVs
  models/contrast/              trained checkpoints
  *_logs/                       run logs

third_party/CBraMod/            official CBraMod repo + pretrained weights
_archive/                       archived backups/logs/dead-ends (git-ignored)

Inherited from upstream (not central to this study):
  Generation/  LaBraM/  EEG-preprocessing/  MEG-preprocessing/  REU-PIPELINE/
```

---

## The encoders

| Encoder | Class | Params | Notes |
|---|---|---|---|
| ATMS | specialized | 3.20M | from scratch |
| EEGNet | specialized | 1.19M | braindecode |
| EEGConformer | specialized | 0.64M | braindecode; ~200 epochs to converge |
| ShallowFBCSPNet | specialized | 0.89M | braindecode |
| LaBraM-Base | foundation | 6.45M | pretrained, fine-tuned |
| CBraMod | foundation | 6.97M | pretrained; standalone repo (see setup) |

Four specialists vs two foundation models: each side is plural, so the finding
is a property of the *class*, not of any single model.

---

## The datasets (and why only one works)

Image-level retrieval needs **repeated presentations** and **image-level
labels**. Only the THINGS family provides both:

| Dataset | Retrieval | Why |
|---|---|---|
| THINGS-EEG2 | **works** | 4-80 reps, image labels (primary) |
| THINGS-EEG1 | no | single-presentation |
| MSS | no | category-only labels |
| EEG-ImageNet | no | per-trial signal too weak (proven chance) |
| Alljoined | no | weak signal, even pooled (proven chance) |
| THINGS-MEG | works | only 4 subjects; cross-modality |

Full mapping in `project_journal.md` and the technical report.

---

## Quick start

```bash
pip install -r requirements.txt
bash setup_cbramod.sh          # one-time, for the 2nd foundation model
bash run_all_gates.sh          # verify everything before spending compute
bash local_sweep.sh            # run the sweep (resumable; Ctrl-C safe)
bash local_sweep.sh --status   # progress table
```

See `REPRODUCE.md` for data placement and the exact per-experiment commands.

---

## A note on braindecode

Pinned to **0.8**. The specialized-encoder wrappers use 0.8-era arguments
(`in_chans`, `n_classes`, `input_window_samples`) removed in 1.x. Do not upgrade;
CBraMod comes from its own repo (`setup_cbramod.sh`) precisely so the pin can
stay. `patch_braindecode_compat.py` exists if you are forced onto 1.x.

---

## Citation

Cite the Paper 1 manuscript. Conducted as part of the NSF Research Experiences
for Undergraduates program at the University of Wyoming.
