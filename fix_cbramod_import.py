#!/usr/bin/env python
"""
fix_cbramod_import.py -- REPAIR the hard CBraMod import.

WHY THIS EXISTS (my mistake, documented so it isn't repeated):
patch_register_cbramod.py inserted a HARD import:

    from cbramod_encoder import CBraMod_Encoder

and cbramod_encoder.py raised ImportError at module level when braindecode <1.6.
Result: eeg_encoders.py failed to import AT ALL, so EEGNet, EEGConformer,
ShallowFBCSPNet, ATMS and LaBraM all died for a dependency none of them use.
One optional model took down the entire registry. That is exactly backwards.

THIS SCRIPT makes the import tolerant. After running it:
  - braindecode 0.8  -> everything works; CBraMod_Encoder simply is not registered
  - braindecode 1.6+ -> CBraMod_Encoder registers and works as intended

Idempotent. Safe to run twice.

USAGE:
    python fix_cbramod_import.py Retrieval/eeg_encoders.py
"""
import os
import sys

HARD = "from cbramod_encoder import CBraMod_Encoder  # second foundation model (LEM)"

SOFT = '''# --- second foundation model (optional; needs braindecode>=1.6) -------------
# Tolerant import: braindecode 0.8 has no CBraMod, and that must NOT break the
# rest of this registry. CBraMod_Encoder is registered only if importable.
try:
    from cbramod_encoder import CBraMod_Encoder
    _HAS_CBRAMOD = True
except Exception as _cbramod_err:
    CBraMod_Encoder = None
    _HAS_CBRAMOD = False
# ---------------------------------------------------------------------------'''

HARD_REG = "    'CBraMod_Encoder':         CBraMod_Encoder,\n"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "Retrieval/eeg_encoders.py"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found."); sys.exit(1)

    s = open(path, encoding="utf-8").read()

    if "_HAS_CBRAMOD" in s:
        print("already repaired -- no change.")
        return

    changed = False

    # 1) soften the import
    if HARD in s:
        s = s.replace(HARD, SOFT)
        changed = True
        print("  softened the CBraMod import")
    else:
        print("  (hard import line not found; may already differ)")

    # 2) make the registry entry conditional: remove the unconditional line and
    #    append a guarded registration AFTER the dict literal closes.
    if HARD_REG in s:
        s = s.replace(HARD_REG, "")
        changed = True
        print("  removed unconditional registry entry")

    if "_HAS_CBRAMOD and" not in s and "if _HAS_CBRAMOD:" not in s:
        # find end of ENCODER_REGISTRY dict and append guarded registration
        i = s.find("ENCODER_REGISTRY = {")
        if i == -1:
            print("ERROR: ENCODER_REGISTRY not found."); sys.exit(2)
        depth = 0
        j = s.find("{", i)
        k = j
        while k < len(s):
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        eol = s.find("\n", k)
        guarded = (
            "\n\n# Register the second foundation model only if its dependency is present.\n"
            "if _HAS_CBRAMOD:\n"
            "    ENCODER_REGISTRY['CBraMod_Encoder'] = CBraMod_Encoder\n"
        )
        s = s[:eol + 1] + guarded + s[eol + 1:]
        changed = True
        print("  added guarded registration after ENCODER_REGISTRY")

    # 3) clean the NORMALIZE_FEAT_ENCODERS insertion if it was added
    s = s.replace("{'CBraMod_Encoder', ", "{")
    s = s.replace("'CBraMod_Encoder', ", "")

    if not changed:
        print("nothing to change.")
        return

    open(path + ".bak_fiximport", "w", encoding="utf-8").write(
        open(path, encoding="utf-8").read()
    )
    open(path, "w", encoding="utf-8").write(s)
    print(f"repaired {path}  (backup: {path}.bak_fiximport)")
    print("\nVerify with:")
    print("  python -c \"import sys; sys.path.insert(0,'Retrieval'); "
          "from eeg_encoders import ENCODER_REGISTRY; print(sorted(ENCODER_REGISTRY))\"")


if __name__ == "__main__":
    main()
