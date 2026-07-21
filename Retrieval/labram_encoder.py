"""
LaBraM_ATMS: adapter making pretrained LaBraM-Base a drop-in encoder for
train_unified.py (retrieval).  Interface matches ATMS:
    forward(eeg (B,63,250), subject_ids)  ->  (B, 1024) features
    .logit_scale  (nn.Parameter)
    .loss_func    (ClipLoss)
subject_ids is accepted but ignored (LaBraM uses channel indices, not subjects).
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly

# --- import the validated scaffold (real weights + official channel vocab) ---
_SCAFFOLD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labram_scaffold")
_LABRAM_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LaBraM")
for _p in (_SCAFFOLD, _LABRAM_REPO):
    if _p not in sys.path:
        sys.path.append(_p)

import importlib.util as _ilu
# Load the scaffold's models.py by explicit path. The name "models" is ambiguous
# (scaffold has models.py; the ATMS repo has a models/ package). Under train_unified.py
# the repo package wins, so a bare "import models" grabs the wrong one — hence explicit load.
_scaf_path = os.path.join(_SCAFFOLD, "models.py")
_scaf_spec = _ilu.spec_from_file_location("labram_scaffold_models", _scaf_path)
_scaf = _ilu.module_from_spec(_scaf_spec)
_scaf_spec.loader.exec_module(_scaf)
LaBraMBaseEncoder = _scaf.LaBraMBaseEncoder
get_input_chans   = _scaf.get_input_chans
inject_lora       = _scaf.inject_lora
# Load ClipLoss from the ATMS repo's models/ package by explicit path.
_loss_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "loss.py")
_spec = _ilu.spec_from_file_location("atms_loss", _loss_path)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ClipLoss = _mod.ClipLoss

# THINGS-EEG2 63-channel montage order (matches your data's ch_names)
_ALLJOINED_CHANNELS_64 = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']

_THINGS_CHANNELS = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5',
    'CP1','Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz','C4',
    'T8','FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7',
    'FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6',
    'P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','F2','AF4','AF8']

_EEGIMAGENET_CHANNELS_60 = ['Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz',
    'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7',
    'C5','C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4',
    'CP6','TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO5','PO3','POz',
    'PO4','PO6','PO8','O1','Oz','O2']

_CKPT = os.path.join(_LABRAM_REPO, "checkpoints", "labram-base.pth")


class ProjectionHead(nn.Module):
    """200 -> 1024, L2-normalized (matches ViT-H-14 target dim)."""
    def __init__(self, in_dim=200, out_dim=1024, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class LaBraM_ATMS(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250,
                 out_dim=1024, lora_rank=8, lora_alpha=16,
                 freeze_backbone=True, channel_names=None, **kwargs):
        super().__init__()
        self.encoder = LaBraMBaseEncoder(
            patch_size=200, embed_dim=200, depth=12, num_heads=10,
            pretrained_ckpt=_CKPT, pooling="mean")
        assert self.encoder.backend == "official", \
            "LaBraM fell back to random weights — check repo path / init_values."
        # Full fine-tune (all LaBraM params trainable). Matches the intra
        # full-FT baseline (v200-top1=0.175). To switch to peft-LoRA, restore
        # the get_peft_model block targeting ["fc1","fc2"].
        print("[LaBraM_ATMS] full fine-tune (all params trainable).")
        self.proj = ProjectionHead(in_dim=200, out_dim=out_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
        # precompute channel indices once (official vocab, validated)
        _chs = channel_names if channel_names is not None else _THINGS_CHANNELS
        self._input_chans = get_input_chans(_chs)
        print(f'[LaBraM_ATMS] using {len(_chs)} channels for input_chans')

    def _resample_patchify(self, eeg):
        # eeg: (B, 63, 250) @250Hz -> (B, 63, 200) @200Hz -> (B, 63, 1, 200)
        if eeg.shape[-1] != 200:
            arr = resample_poly(eeg.detach().cpu().numpy(), 4, 5, axis=-1)
            eeg = torch.from_numpy(arr).to(next(self.parameters()).device).float()
        B, C, T = eeg.shape
        return eeg.reshape(B, C, 1, T)

    def forward(self, eeg, subject_ids=None):
        x = self._resample_patchify(eeg)
        feats = self.encoder(x, self._input_chans)   # (B, 200)
        return self.proj(feats)                        # (B, 1024), L2-normed
