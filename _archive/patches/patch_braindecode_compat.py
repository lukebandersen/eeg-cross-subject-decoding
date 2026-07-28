#!/usr/bin/env python
"""
patch_braindecode_compat.py -- make eeg_encoders.py work across braindecode versions.

VERIFIED ISSUE: braindecode >=1.x renamed EEGNetv4 -> EEGNet. The repo imports
`EEGNetv4`, which raises ImportError on newer braindecode (confirmed on 1.6.1).
EEGConformer, ShallowFBCSPNet, ATCNet, EEGITNet keep their names.

This patch replaces the hard import with a version-tolerant one. Idempotent:
running it twice is safe (it detects the already-patched form).

USAGE:  python scripts/patch_braindecode_compat.py /path/to/eeg_encoders.py
        (defaults to ./Retrieval/eeg_encoders.py if no path given)
"""
import sys, os, re

OLD = "from braindecode.models import ATCNet, EEGConformer, EEGITNet, EEGNetv4, ShallowFBCSPNet"
NEW = (
"from braindecode.models import ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet\n"
"try:  # braindecode <1.x\n"
"    from braindecode.models import EEGNetv4\n"
"except ImportError:  # braindecode >=1.x renamed EEGNetv4 -> EEGNet\n"
"    from braindecode.models import EEGNet as EEGNetv4"
)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "Retrieval/eeg_encoders.py"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Pass the path to eeg_encoders.py."); sys.exit(1)
    s = open(path, encoding="utf-8").read()
    if "except ImportError:  # braindecode >=1.x renamed" in s:
        print("already patched -- no change."); return
    if OLD not in s:
        print("WARNING: exact import line not found. Check braindecode import in", path)
        print("Expected:", OLD); sys.exit(2)
    bak = path + ".bak_braindecode"
    open(bak, "w", encoding="utf-8").write(s)
    s = s.replace(OLD, NEW)
    open(path, "w", encoding="utf-8").write(s)
    print(f"patched {path}  (backup: {bak})")

if __name__ == "__main__":
    main()
