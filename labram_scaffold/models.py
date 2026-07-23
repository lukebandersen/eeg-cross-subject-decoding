"""
models.py
=========
LaBraM-Base backbone integration for ATMS triple-contrastive alignment.

Pipeline:  EEG (B, C, T@200Hz)
             -> patchify  (B, C, A, 200)            [A = T // 200]
             -> 30% input-patch masking (train aug)
             -> LaBraM-Base encoder (mean pooling)  -> (B, 200)
             -> projection "linear bottleneck"      -> (B, 768) L2-normalised
                                                       == z_EEG

The encoder has two backends, chosen automatically:
  * OFFICIAL  : the real 935963004/LaBraM model + labram-base.pth weights.
                This is the path you use for results. Channel indices come from
                LaBraM's own utils so the pretrained channel-embedding table
                aligns with your electrodes.
  * FALLBACK  : a small architecture-compatible stand-in (random weights) so the
                whole pipeline runs for shape/plumbing checks before you clone
                the repo or download weights. NOT pretrained -- never report
                numbers from it.

A loud warning is printed whenever the fallback is used.
"""
from __future__ import annotations

import math
import re
import warnings
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Channel-name -> LaBraM input_chans indices
# =============================================================================
_LOCAL_STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2',
]


def get_input_chans(ch_names: Sequence[str]) -> List[int]:
    """Map electrode names -> LaBraM channel indices, prepending 0 for the
    [CLS]/class token (matching the official `get_input_chans`).

    Tries the official LaBraM utils first; falls back to the local list with a
    warning. Matching is case-insensitive (THINGS uses 'Fp1', LaBraM 'FP1').
    """
    try:
        import ast as _ast, os as _os
        _lu = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "LaBraM", "utils.py")
        _src = open(_lu, encoding="utf-8").read()
        vocab = None
        for _n in _ast.walk(_ast.parse(_src)):
            if isinstance(_n, _ast.Assign):
                for _t in _n.targets:
                    if isinstance(_t, _ast.Name) and _t.id == "standard_1020":
                        vocab = _ast.literal_eval(_n.value)
        if vocab is None:
            raise ValueError("standard_1020 not found in LaBraM/utils.py")
        official = True
    except Exception:
        vocab = _LOCAL_STANDARD_1020
        official = False

    upper = [v.upper() for v in vocab]
    idx = [0]
    missing = []
    for ch in ch_names:
        key = ch.upper()
        if key in upper:
            idx.append(upper.index(key) + 1)
        else:
            missing.append(ch)
            idx.append(0)
    if missing:
        warnings.warn(
            f"[channels] {len(missing)} electrode(s) not in "
            f"{'official' if official else 'LOCAL fallback'} vocabulary: "
            f"{missing}. They were mapped to the class-token index and will "
            f"carry no spatial information. Fix the `channels:` list or add "
            f"the official LaBraM repo to PYTHONPATH."
        )
    if not official:
        warnings.warn(
            "[channels] Using LOCAL fallback electrode vocabulary. This is fine "
            "for smoke tests but for pretrained weights you MUST expose the "
            "official LaBraM `utils.standard_1020` so indices align with the "
            "pretrained channel-embedding table."
        )
    return idx


# =============================================================================
# LoRA
# =============================================================================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * update


def inject_lora(module: nn.Module, targets: Sequence[str], rank: int,
                alpha: int, dropout: float) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in targets):
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            count += 1
        else:
            count += inject_lora(child, targets, rank, alpha, dropout)
    return count


# =============================================================================
# Input-patch masking (augmentation only)
# =============================================================================
class PatchMasking(nn.Module):
    def __init__(self, ratio: float, patch_len: int, mode: str = "learned",
                 apply_in_eval: bool = False):
        super().__init__()
        assert 0.0 <= ratio < 1.0
        assert mode in ("learned", "zero")
        self.ratio = ratio
        self.mode = mode
        self.apply_in_eval = apply_in_eval
        if mode == "learned":
            self.mask_token = nn.Parameter(torch.zeros(patch_len))
            nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ratio == 0.0 or (not self.training and not self.apply_in_eval):
            return x
        B, C, A, P = x.shape
        n_tok = C * A
        n_mask = int(round(self.ratio * n_tok))
        if n_mask == 0:
            return x
        flat = x.reshape(B, n_tok, P)
        noise = torch.rand(B, n_tok, device=x.device)
        mask_idx = noise.argsort(dim=1)[:, :n_mask]
        token = (self.mask_token if self.mode == "learned"
                 else torch.zeros(P, device=x.device, dtype=x.dtype))
        flat = flat.clone()
        bidx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, n_mask)
        flat[bidx, mask_idx] = token.to(flat.dtype)
        return flat.reshape(B, C, A, P)


# =============================================================================
# Fallback LaBraM-compatible backbone (random weights; smoke only)
# =============================================================================
class _TemporalConvPatch(nn.Module):
    def __init__(self, patch_len: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(patch_len, embed_dim), nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        B, C, A, P = x.shape
        return self.proj(x.reshape(B, C * A, P))


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.qkv = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h = self.qkv(h)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class _FallbackLaBraM(nn.Module):
    def __init__(self, patch_len, embed_dim, depth, num_heads, n_chan_vocab=128):
        super().__init__()
        self.patch_embed = _TemporalConvPatch(patch_len, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.chan_embed = nn.Embedding(n_chan_vocab, embed_dim)
        self.temporal_embed = nn.Parameter(torch.zeros(1, 256, embed_dim))
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.temporal_embed, std=0.02)

    def forward_features(self, x, input_chans=None, pooling="mean"):
        B, C, A, P = x.shape
        tok = self.patch_embed(x)
        if input_chans is not None:
            ch = torch.as_tensor(input_chans[1:], device=x.device)
            ch = ch.clamp(max=self.chan_embed.num_embeddings - 1)
            ch = ch.repeat_interleave(A).unsqueeze(0)
            tok = tok + self.chan_embed(ch)
        tok = tok + self.temporal_embed[:, :tok.size(1)]
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, tok], dim=1)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return h[:, 0] if pooling == "cls" else h[:, 1:].mean(dim=1)


# =============================================================================
# Encoder wrapper (chooses official vs fallback)
# =============================================================================
class LaBraMBaseEncoder(nn.Module):
    def __init__(self, patch_size=200, embed_dim=200, depth=12, num_heads=10,
                 pretrained_ckpt: Optional[str] = None, pooling: str = "mean"):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pooling = pooling
        self.backend, self.model = self._build(
            patch_size, embed_dim, depth, num_heads, pretrained_ckpt)

    def _build(self, patch_size, embed_dim, depth, num_heads, ckpt):
        try:
            import modeling_finetune
            from timm.models import create_model
            model = create_model(
                "labram_base_patch200_200",
                pretrained=False,
                num_classes=0,
                drop_rate=0.0, drop_path_rate=0.1,
                use_mean_pooling=(self.pooling == "mean"),
                init_values=0.1,
            )
            if ckpt:
                self._load_official_ckpt(model, ckpt)
            else:
                warnings.warn("[encoder] OFFICIAL LaBraM built but no checkpoint "
                              "given (model.pretrained_ckpt is null) -> random "
                              "weights.")
            return "official", model
        except Exception as e:
            warnings.warn(
                "=" * 70 + "\n"
                f"[encoder] Official LaBraM unavailable ({type(e).__name__}: {e}).\n"
                "Falling back to an architecture-compatible RANDOM-WEIGHT stand-in.\n"
                "Use this ONLY for shape/plumbing checks. For real results, clone\n"
                "https://github.com/935963004/LaBraM, put it on PYTHONPATH, and set\n"
                "model.pretrained_ckpt to labram-base.pth.\n" + "=" * 70
            )
            return "fallback", _FallbackLaBraM(patch_size, embed_dim, depth, num_heads)

    @staticmethod
    def _load_official_ckpt(model: nn.Module, ckpt_path: str) -> None:
        sd = torch.load(ckpt_path, map_location="cpu")
        for k in ("model", "module", "state_dict"):
            if isinstance(sd, dict) and k in sd:
                sd = sd[k]
                break
        sd = {re.sub(r"^(student\.|module\.|backbone\.)", "", k): v
              for k, v in sd.items()}
        model_keys = set(model.state_dict().keys())
        load_keys = set(sd.keys())
        matched = model_keys & load_keys
        result = model.load_state_dict(sd, strict=False)
        print(f"[encoder] loaded {ckpt_path}: matched={len(matched)}/"
              f"{len(model_keys)} | missing={len(result.missing_keys)} | "
              f"unexpected={len(result.unexpected_keys)}")
        if len(matched) < 0.5 * len(model_keys):
            warnings.warn("[encoder] <50% of weights matched -- the checkpoint "
                          "key names likely differ from this model. Verify the "
                          "LaBraM version / variant.")

    def forward(self, x_patched: torch.Tensor, input_chans) -> torch.Tensor:
        if self.backend == "official":
            feats = self.model.forward_features(
                x_patched, input_chans=torch.as_tensor(
                    input_chans, device=x_patched.device))
            if feats.dim() == 3:
                feats = (feats[:, 0] if self.pooling == "cls"
                         else feats[:, 1:].mean(dim=1))
            return feats
        return self.model.forward_features(x_patched, input_chans, self.pooling)

    @property
    def blocks(self):
        return getattr(self.model, "blocks", None)


# =============================================================================
# Projection head ("linear bottleneck" 200 -> 768)
# =============================================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Sequential(nn.LayerNorm(in_dim),
                                     nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# =============================================================================
# Full EEG model
# =============================================================================
class ATMSLaBraM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m, mk, lo = cfg["model"], cfg["masking"], cfg["lora"]
        self.patch_size = m["patch_size"]

        self.masking = PatchMasking(
            ratio=mk["ratio"], patch_len=m["patch_size"],
            mode=mk["mode"], apply_in_eval=mk["apply_in_eval"])

        self.encoder = LaBraMBaseEncoder(
            patch_size=m["patch_size"], embed_dim=m["embed_dim"],
            depth=m["depth"], num_heads=m["num_heads"],
            pretrained_ckpt=m["pretrained_ckpt"], pooling=m["pooling"])

        self.target_names = ("image", "text", "canny")
        self.proj_mode = m["projection"].get("mode", "shared")
        assert self.proj_mode in ("shared", "separate")
        proj_kwargs = dict(
            in_dim=m["embed_dim"], out_dim=m["projection"]["out_dim"],
            hidden_dim=m["projection"]["hidden_dim"],
            dropout=m["projection"]["dropout"])
        if self.proj_mode == "separate":
            self.proj = nn.ModuleDict(
                {name: ProjectionHead(**proj_kwargs) for name in self.target_names})
        else:
            self.proj = ProjectionHead(**proj_kwargs)

        if lo["enabled"]:
            n = inject_lora(self.encoder.model, lo["targets"],
                            lo["rank"], lo["alpha"], lo["dropout"])
            print(f"[lora] injected adapters into {n} attention Linear layer(s).")
            if lo["freeze_backbone"]:
                self._freeze_backbone_keep_lora()

        self._freeze_first_n_blocks(cfg.get("freeze_first_n_blocks", 0))

    def _freeze_backbone_keep_lora(self):
        for name, p in self.encoder.named_parameters():
            if "lora_" not in name:
                p.requires_grad_(False)

    def _freeze_first_n_blocks(self, n):
        blocks = self.encoder.blocks
        if not n or blocks is None:
            return
        for i in range(min(n, len(blocks))):
            for p in blocks[i].parameters():
                p.requires_grad_(False)
        print(f"[freeze] froze first {min(n, len(blocks))} transformer block(s) "
              f"(incl. their LoRA adapters).")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_summary(self):
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"trainable {tr:,} / {tot:,} ({100*tr/max(tot,1):.2f}%)"

    def _patchify(self, eeg):
        B, C, T = eeg.shape
        assert T % self.patch_size == 0, (
            f"time length {T} is not a multiple of patch_size {self.patch_size}; "
            f"resample to 200 Hz first (preprocessing.py).")
        A = T // self.patch_size
        return eeg.reshape(B, C, A, self.patch_size)

    def forward(self, eeg, input_chans):
        x = self._patchify(eeg)
        x = self.masking(x)
        feats = self.encoder(x, input_chans)
        if self.proj_mode == "separate":
            return {name: head(feats) for name, head in self.proj.items()}
        z = self.proj(feats)
        return {name: z for name in self.target_names}


def build_model(cfg: dict) -> ATMSLaBraM:
    model = ATMSLaBraM(cfg)
    print(f"[model] backend={model.encoder.backend} | "
          f"proj={model.proj_mode} | {model.param_summary()}")
    return model
