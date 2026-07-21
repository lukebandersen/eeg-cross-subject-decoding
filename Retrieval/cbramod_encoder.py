#!/usr/bin/env python
"""
cbramod_encoder.py -- second foundation model integration: CBraMod (ICLR 2025).

STANDALONE. Uses the OFFICIAL CBraMod repo, NOT braindecode.

WHY NOT braindecode's CBraMod:
  braindecode >=1.6 ships a CBraMod class, but the HF repo it points at
  ("braindecode/CBraMod") is ARCHITECTURE-ONLY and distributes NO pretrained
  weights. Using it would have silently given a randomly-initialised "foundation
  model" -- exactly the failure this project has been bitten by before.
  Worse, upgrading braindecode 0.8 -> 1.6 BREAKS every Block B wrapper in
  eeg_encoders.py (they use the removed kwargs in_chans / n_classes /
  input_window_samples / add_log_softmax). add_log_softmax's removal would
  SILENTLY change the Block B embeddings.
  => We keep braindecode 0.8 and take CBraMod from its own MIT-licensed repo,
     which is where the real pretrained weights live.

SETUP (run scripts/setup_cbramod.sh, or do it manually):
    git clone https://github.com/wjq-learning/CBraMod.git third_party/CBraMod
    # download pretrained_weights.pth from https://huggingface.co/weighting666/CBraMod
    # place at: third_party/CBraMod/pretrained_weights/pretrained_weights.pth
    pip install einops        # the repo needs it

OFFICIAL USAGE THIS WRAPS (from the repo README):
    from models.cbramod import CBraMod
    model = CBraMod()
    model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth'))
    model.proj_out = nn.Identity()          # -> returns encoder features
    mock_eeg = torch.randn((8, 22, 4, 200)) # (batch, chans, segments, points_per_patch)

SHAPE PATH FOR THINGS-EEG2:
    (B, 63, 250) @250Hz            input, 1.0 s
      -> rFFT resample to 200 Hz   (B, 63, 200)
      -> reshape into 1s patches   (B, 63, 1, 200)   [200 points = 1 patch]
      -> CBraMod (proj_out=Identity)                 (B, 63, 1, 200)
      -> per-channel reduce 200->16                  (B, 63, 1, 16)
      -> flatten 1008 -> project                     (B, 1024)

PARAM BUDGET (deliberate): a naive Flatten(12600)->Linear(1024) head costs
12.9M and would dwarf the ~4M backbone, confounding "foundation vs specialist"
with sheer size. The per-channel reduction keeps the head ~1M so the total sits
near LaBraM-Base's 6.45M and the comparison stays about the backbone.

PIPELINE CONTRACT (documented at the top of eeg_encoders.py -- all three are
required, retrieval_engine.py reads the first two off the model):
    model.logit_scale  -- learnable temperature (nn.Parameter)
    model.loss_func    -- ClipLoss instance
    model(eeg)         -- (B, 63, 250) -> (B, 1024)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Locate the official CBraMod repo. Override with env CBRAMOD_REPO if you put
# it elsewhere. Import is OPTIONAL: a missing repo must never break the encoder
# registry for the encoders that do not need it.
# ---------------------------------------------------------------------------
_DEFAULT_REPO_CANDIDATES = [
    os.environ.get("CBRAMOD_REPO", ""),
    "third_party/CBraMod",
    "../third_party/CBraMod",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "CBraMod"),
]

CBRAMOD_AVAILABLE = False
_CBRAMOD_IMPORT_ERROR = None
_CBraMod_cls = None
CBRAMOD_REPO_PATH = None


def _import_cbramod_isolated(repo_path):
    """
    Import CBraMod WITHOUT poisoning this project's `models` package.

    THE TRAP (hit for real, do not undo this):
    The CBraMod repo ships its own top-level package named `models`, and
    cbramod.py does `from models.criss_cross_transformer import ...`. This repo
    ALSO has a `models` package (models/loss.py -> ClipLoss). Naively doing
    sys.path.insert(0, repo) makes `models` resolve to CBRAMOD's package for the
    rest of the process, so `from models.loss import ClipLoss` explodes with
    "No module named 'models.loss'" and takes the whole pipeline with it.

    Fix: borrow the `models` name only for the duration of the import, then put
    everything back exactly as it was. The returned class keeps working because
    its own module-level imports already resolved.
    """
    saved_mods = {k: v for k, v in sys.modules.items()
                  if k == "models" or k.startswith("models.")}
    saved_path = list(sys.path)
    try:
        for k in list(sys.modules):
            if k == "models" or k.startswith("models."):
                del sys.modules[k]
        sys.path.insert(0, repo_path)
        from models.cbramod import CBraMod as cls  # type: ignore
        return cls
    finally:
        # drop CBraMod's `models` so it cannot shadow this project's
        for k in list(sys.modules):
            if k == "models" or k.startswith("models."):
                del sys.modules[k]
        sys.modules.update(saved_mods)   # restore the project's, if any
        sys.path[:] = saved_path         # restore path exactly


for _cand in _DEFAULT_REPO_CANDIDATES:
    if not _cand:
        continue
    _cand = os.path.abspath(_cand)
    if os.path.isdir(os.path.join(_cand, "models")):
        try:
            _CBraMod_cls = _import_cbramod_isolated(_cand)
            CBRAMOD_AVAILABLE = True
            CBRAMOD_REPO_PATH = _cand
            break
        except Exception as _e:
            _CBRAMOD_IMPORT_ERROR = _e

if not CBRAMOD_AVAILABLE and _CBRAMOD_IMPORT_ERROR is None:
    _CBRAMOD_IMPORT_ERROR = FileNotFoundError(
        "CBraMod repo not found. Looked in: "
        + ", ".join(c for c in _DEFAULT_REPO_CANDIDATES if c)
    )

THINGS_SFREQ = 250
CBRAMOD_SFREQ = 200
CBRAMOD_PATCH = 200          # points per patch = 1 s @ 200 Hz
CBRAMOD_EMB = 200            # encoder feature width per patch
DEFAULT_WEIGHTS = "pretrained_weights/pretrained_weights.pth"


def fft_resample(x: torch.Tensor, n_out: int) -> torch.Tensor:
    """Resample the last dim via rFFT (differentiable, GPU-friendly).
    Mirrors the Alljoined loader convention so both paths agree."""
    n_in = x.shape[-1]
    if n_in == n_out:
        return x
    X = torch.fft.rfft(x, dim=-1)
    n_keep = min(X.shape[-1], n_out // 2 + 1)
    Xr = torch.zeros(*X.shape[:-1], n_out // 2 + 1, dtype=X.dtype, device=x.device)
    Xr[..., :n_keep] = X[..., :n_keep]
    return torch.fft.irfft(Xr, n=n_out, dim=-1) * (n_out / n_in)


class CBraMod_Encoder(nn.Module):
    def __init__(
        self,
        n_chans: int = 63,
        n_times: int = 250,
        out_dim: int = 1024,
        pretrained: bool = True,
        weights_path: str | None = None,
        freeze_backbone: bool = False,
        drop_prob: float = 0.1,
        chan_reduce: int = 16,
    ):
        super().__init__()
        if not CBRAMOD_AVAILABLE:
            raise ImportError(
                "CBraMod_Encoder needs the official CBraMod repo.\n"
                "  Run: bash scripts/setup_cbramod.sh\n"
                "  (clones https://github.com/wjq-learning/CBraMod and fetches weights)\n"
                "  Or set CBRAMOD_REPO=/path/to/CBraMod\n"
                f"  Original error: {_CBRAMOD_IMPORT_ERROR}"
            )

        self.n_chans = n_chans
        self.n_times_in = n_times
        self.out_dim = out_dim

        self.backbone = _CBraMod_cls()
        # proj_out is the pretraining reconstruction head; Identity -> features.
        self.backbone.proj_out = nn.Identity()

        if pretrained:
            wp = weights_path or os.path.join(CBRAMOD_REPO_PATH or "", DEFAULT_WEIGHTS)
            if not os.path.exists(wp):
                raise FileNotFoundError(
                    f"CBraMod pretrained weights not found at: {wp}\n"
                    "  Download from https://huggingface.co/weighting666/CBraMod\n"
                    "  (file: pretrained_weights.pth) or run scripts/setup_cbramod.sh.\n"
                    "  REFUSING to run a 'foundation model' on random weights."
                )
            sd = torch.load(wp, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            model_keys = set(self.backbone.state_dict().keys())
            matched = model_keys & set(sd.keys())
            frac = len(matched) / max(len(model_keys), 1)
            res = self.backbone.load_state_dict(sd, strict=False)
            missing = list(getattr(res, "missing_keys", []))
            if frac < 0.80:
                raise RuntimeError(
                    f"CBraMod weight load FAILED VERIFICATION: only {frac:.0%} of keys "
                    f"matched (need >=80%). Refusing to run on partially-random weights. "
                    f"Missing (first 5): {missing[:5]}"
                )
            self._load_frac = frac

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        n_patches = 1                                  # 1.0 s -> one 200-pt patch
        flat = n_chans * n_patches * chan_reduce       # 63 * 1 * 16 = 1008
        self.reduce = nn.Linear(CBRAMOD_EMB, chan_reduce)
        self.proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(flat),
            nn.Linear(flat, out_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        # --- required interface (see eeg_encoders.py docstring) -------------
        # Imported here rather than at module scope: this file is imported BY
        # eeg_encoders.py before it pulls in ClipLoss, and the isolated CBraMod
        # import above temporarily borrows the `models` namespace. A local
        # import sidesteps both ordering hazards.
        from models.loss import ClipLoss
        # 1/0.07 is the CLIP default and what ATMS/LaBraM_ATMS use. Matching it
        # keeps the contrastive temperature identical across the comparison, so
        # any difference is the backbone, not the loss scaling.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """(B, 63, 250) @250Hz -> (B, 1024). Extra args ignored for drop-in compat."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, chans, times), got {tuple(x.shape)}")
        if x.shape[1] != self.n_chans:
            raise ValueError(
                f"channel mismatch: built for {self.n_chans}, got {x.shape[1]}"
            )
        x = fft_resample(x, CBRAMOD_SFREQ)             # (B, C, 200)
        B, C, T = x.shape
        n_patches = T // CBRAMOD_PATCH
        if n_patches < 1:
            raise ValueError(f"need >= {CBRAMOD_PATCH} points after resample, got {T}")
        x = x.reshape(B, C, n_patches, CBRAMOD_PATCH)  # (B, C, S, 200)
        feat = self.backbone(x)                        # (B, C, S, 200)
        if feat.dim() != 4:
            raise RuntimeError(
                f"expected (B,C,S,P) from CBraMod, got {tuple(feat.shape)}. "
                "Is proj_out still nn.Identity()?"
            )
        feat = self.reduce(feat)                       # (B, C, S, chan_reduce)
        return self.proj(feat)                         # (B, 1024)


def verify_pretrained_load(verbose: bool = True) -> bool:
    """
    Prove the weights are REAL, not random. Compares a pretrained backbone
    against a fresh random one: if they match, the load did nothing.
    Run this before ANY CBraMod training.
    """
    if not CBRAMOD_AVAILABLE:
        if verbose:
            print("SKIP: CBraMod repo not present.", _CBRAMOD_IMPORT_ERROR)
        return False
    torch.manual_seed(0)
    rnd = CBraMod_Encoder(pretrained=False)
    pre = CBraMod_Encoder(pretrained=True)
    rp = dict(rnd.backbone.named_parameters())
    pp = dict(pre.backbone.named_parameters())
    shared = [k for k in rp if k in pp and rp[k].shape == pp[k].shape]
    if not shared:
        if verbose:
            print("VERIFY FAIL: no comparable parameters.")
        return False
    n_diff = sum(1 for k in shared if not torch.allclose(rp[k], pp[k]))
    frac = n_diff / len(shared)
    if verbose:
        print(f"weights loaded from: {CBRAMOD_REPO_PATH}/{DEFAULT_WEIGHTS}")
        print(f"key match on load  : {getattr(pre, '_load_frac', float('nan')):.0%}")
        print(f"params differing from random: {n_diff}/{len(shared)} ({frac:.0%})")
    ok = frac > 0.9
    if verbose:
        print("VERIFY PASS: pretrained weights are real." if ok else
              "VERIFY FAIL: weights look random -- load did NOT take effect.")
    return ok


if __name__ == "__main__":
    print("=== CBraMod_Encoder check ===")
    print(f"repo available: {CBRAMOD_AVAILABLE}  path={CBRAMOD_REPO_PATH}")
    if not CBRAMOD_AVAILABLE:
        print(f"  {_CBRAMOD_IMPORT_ERROR}")
        print("  Run: bash scripts/setup_cbramod.sh")
        raise SystemExit(2)
    # NOTE: run from the repo root so `models.loss` (ClipLoss) resolves.
    m = CBraMod_Encoder(pretrained=False)
    x = torch.randn(2, 63, 250)
    with torch.no_grad():
        y = m(x)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"params={n:.2f}M  in={tuple(x.shape)}  out={tuple(y.shape)}")
    assert y.shape == (2, 1024), f"BAD SHAPE {tuple(y.shape)}"
    print("architecture OK. Now run verify_pretrained_load().")
