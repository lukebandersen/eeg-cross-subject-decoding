#!/usr/bin/env python
"""
patch_register_cbramod.py -- register CBraMod_Encoder in the pipeline.

Idempotent. Wires the second foundation model into eeg_encoders.py so that
`--encoder_type CBraMod_Encoder` works exactly like every other encoder.

USAGE:  python scripts/patch_register_cbramod.py Retrieval/eeg_encoders.py
        (also copy src/cbramod_encoder.py next to eeg_encoders.py first)

WHAT IT DOES
  1. adds:  from cbramod_encoder import CBraMod_Encoder
  2. adds 'CBraMod_Encoder': CBraMod_Encoder to ENCODER_REGISTRY
  3. adds it to NORMALIZE_FEAT_ENCODERS (matching LaBraM's convention, since
     both are foundation models feeding the same contrastive head)
"""
import os
import sys

IMPORT_LINE = "from cbramod_encoder import CBraMod_Encoder  # second foundation model (LEM)"
MARKER = "CBraMod_Encoder"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "Retrieval/eeg_encoders.py"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found."); sys.exit(1)
    s = open(path, encoding="utf-8").read()

    if MARKER in s:
        print("already registered -- no change."); return

    open(path + ".bak_cbramod", "w", encoding="utf-8").write(s)

    # 1) import: place right after the braindecode import block
    anchor = "from braindecode.models import"
    idx = s.find(anchor)
    if idx == -1:
        print("WARNING: braindecode import not found; appending import at top.")
        s = IMPORT_LINE + "\n" + s
    else:
        eol = s.find("\n", idx)
        # skip past a possible try/except shim added by patch_braindecode_compat
        tail = s[eol:eol + 400]
        if "except ImportError" in tail:
            blk_end = s.find("\n", s.find("as EEGNetv4", eol))
            eol = blk_end if blk_end != -1 else eol
        s = s[:eol + 1] + IMPORT_LINE + "\n" + s[eol + 1:]

    # 2) registry
    reg = "ENCODER_REGISTRY = {"
    i = s.find(reg)
    if i == -1:
        print("ERROR: ENCODER_REGISTRY not found."); sys.exit(2)
    j = s.find("\n", i)
    s = s[:j + 1] + "    'CBraMod_Encoder':         CBraMod_Encoder,\n" + s[j + 1:]

    # 3) normalize-features set (match LaBraM's handling)
    for name in ("NORMALIZE_FEAT_ENCODERS",):
        k = s.find(name)
        if k != -1:
            br = s.find("{", k)
            if br != -1 and s.find("}", br) != -1:
                s = s[:br + 1] + "'CBraMod_Encoder', " + s[br + 1:]

    open(path, "w", encoding="utf-8").write(s)
    print(f"registered CBraMod_Encoder in {path}  (backup: {path}.bak_cbramod)")
    print("Next: python -c \"from eeg_encoders import build_encoder; "
          "m=build_encoder('CBraMod_Encoder'); print(sum(p.numel() for p in m.parameters())/1e6,'M')\"")


if __name__ == "__main__":
    main()
