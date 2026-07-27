#!/usr/bin/env python
"""
patch_freeze_backbone.py -- wire a real --freeze_backbone flag end to end.

WHY: the frozen-backbone experiment distinguishes the two live explanations for
the LaBraM (~90%) vs CBraMod (~54%) retention gap:

  - fine-tuning DESTROYS invariant features  -> frozen retention jumps up
  - the features are LESS invariant to begin  -> frozen retention stays low

CBraMod's wrapper already honours freeze_backbone (real requires_grad=False on
self.backbone). LaBraM's flag is DEAD: accepted, never used. And nothing
upstream can set either flag. This patch:

  1. train_unified.py : add --freeze_backbone (store_true) and put it into the
                        kwargs dict that build_encoder forwards to the encoder.
  2. labram_encoder.py: make the flag real -- freeze self.encoder, leave the
                        projection head / logit_scale / loss trainable so the
                        frozen FEATURES are still mapped into CLIP space (a
                        frozen head would test random projections, not the
                        pretrained representation). CBraMod already does this.

Only self.encoder / self.backbone is frozen; the projection head stays
trainable in both, so the two LEMs freeze symmetrically. Idempotent; writes
.bak files. Run from the repo root:  python patch_freeze_backbone.py
"""
import os
import re
import shutil
import sys

TU = "Retrieval/train_unified.py"
LE = "Retrieval/labram_encoder.py"


def patch_train_unified(kwargs_var):
    s = open(TU, encoding="utf-8").read()
    if "--freeze_backbone" in s:
        print(f"{TU}: already has --freeze_backbone")
        return
    # add the CLI arg next to the other training args (anchor on --seed line)
    anchor = "p.add_argument('--seed',       type=int,   default=42)"
    if anchor not in s:
        sys.exit(f"{TU}: could not find the --seed argument to anchor on.")
    s = s.replace(
        anchor,
        anchor + "\n    p.add_argument('--freeze_backbone', action='store_true',\n"
                 "                   help='freeze the pretrained backbone; train "
                 "only the projection head. For LaBraM_ATMS and CBraMod_Encoder.')")
    # inject freeze_backbone into the kwargs dict that build_encoder forwards
    # (the dict is populated shortly before the build_encoder(...) call)
    # match the marker's own indentation so we never emit a mis-indented line
    mline = next((ln for ln in s.splitlines() if "model = build_encoder(" in ln), None)
    if mline is None:
        sys.exit(f"{TU}: could not find the build_encoder call.")
    indent = mline[:len(mline) - len(mline.lstrip())]
    marker = mline
    inject = (f"{indent}{kwargs_var}['freeze_backbone'] = args.freeze_backbone\n"
              f"{mline}")
    s = s.replace(marker, inject, 1)

    shutil.copy(TU, TU + ".bak_freeze")
    open(TU, "w", encoding="utf-8").write(s)
    print(f"patched {TU}  (backup: {TU}.bak_freeze)")


def patch_labram():
    s = open(LE, encoding="utf-8").read()
    if "backbone FROZEN" in s:
        print(f"{LE}: already has a real freeze")
        return
    old = '''        print("[LaBraM_ATMS] full fine-tune (all params trainable).")'''
    if old not in s:
        sys.exit(f"{LE}: could not find the 'full fine-tune' print to replace.")
    new = '''        if freeze_backbone:
            for _p in self.encoder.parameters():
                _p.requires_grad = False
            print("[LaBraM_ATMS] backbone FROZEN (encoder requires_grad=False; "
                  "projection head stays trainable).")
        else:
            print("[LaBraM_ATMS] full fine-tune (all params trainable).")'''
    s = s.replace(old, new, 1)
    shutil.copy(LE, LE + ".bak_freeze")
    open(LE, "w", encoding="utf-8").write(s)
    print(f"patched {LE}  (backup: {LE}.bak_freeze)")


def main():
    if not (os.path.exists(TU) and os.path.exists(LE)):
        sys.exit("Run from the repo root (Retrieval/ must be visible).")

    # find the kwargs dict name passed to build_encoder as **NAME
    tu = open(TU, encoding="utf-8").read()
    m = re.search(r"\*\*(\w+),?\s*\)", tu[tu.find("model = build_encoder("):])
    if not m:
        sys.exit("Could not determine the **kwargs dict name in the "
                 "build_encoder call. Paste `grep -n _enc_kwargs` output.")
    kwargs_var = m.group(1)
    print(f"kwargs dict forwarded to build_encoder: {kwargs_var}")

    patch_train_unified(kwargs_var)
    patch_labram()

    print("\nRun the frozen experiment:")
    print("  python train_unified.py --mode loso --encoder_type CBraMod_Encoder \\")
    print("      --freeze_backbone --seed 42 [ ...usual data/epoch args ]")
    print("  python train_unified.py --mode loso --encoder_type LaBraM_ATMS \\")
    print("      --freeze_backbone --seed 42 [ ... ]")
    print("\nThe log must print 'backbone FROZEN' for each. If it says 'full "
          "fine-tune', the flag did not thread through -- stop and check.")


if __name__ == "__main__":
    main()
