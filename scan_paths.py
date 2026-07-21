#!/usr/bin/env python
"""Scan repo for hardcoded Windows paths that break on ARCC. REPORTS ONLY."""
import os, re, glob
ROOT = os.path.expanduser("~/Desktop/EEG_Image_decode-develop")
PATTERNS = [
    (re.compile(r'C:[/\\]', re.I), 'Windows drive path'),
    (re.compile(r'/c/Users', re.I), 'Git-Bash mount path'),
    (re.compile(r'mwolff3'), 'hardcoded username'),
    (re.compile(r'~/Desktop/EEG'), 'hardcoded ~/Desktop path'),
]
SCAN_FILES = ['eegimagenet_dataset.py','extract_eegimagenet_clip.py',
    'run_eegimagenet_sweep.sh','aggregate_seed44.py','Retrieval/train_unified.py',
    'Retrieval/labram_encoder.py','Retrieval/run.sh']
SCAN_FILES += [os.path.relpath(p, ROOT) for p in glob.glob(f"{ROOT}/*.sh")]
SCAN_FILES += [os.path.relpath(p, ROOT) for p in glob.glob(f"{ROOT}/Retrieval/diag_*.py")]
SCAN_FILES = sorted(set(SCAN_FILES))
total = files_hit = 0
for rel in SCAN_FILES:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path): continue
    lines = open(path, encoding='utf-8', errors='ignore').readlines()
    hits = []
    for i, line in enumerate(lines, 1):
        for pat, desc in PATTERNS:
            if pat.search(line):
                hits.append((i, desc, line.strip()[:90])); break
    if hits:
        files_hit += 1; total += len(hits)
        print(f"\n### {rel}  ({len(hits)} path(s))")
        for ln, desc, txt in hits[:12]:
            print(f"  L{ln}: [{desc}] {txt}")
        if len(hits) > 12: print(f"  ... and {len(hits)-12} more")
print(f"\n{'='*60}\nSUMMARY: {total} hardcoded paths across {files_hit} files\n{'='*60}")
