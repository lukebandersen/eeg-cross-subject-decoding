#!/usr/bin/env python
"""
run_config.py -- record what actually produced a run, permanently, per run.

THE PROBLEM THIS CLOSES
-----------------------
Run configuration lives nowhere in the output. The results CSV has
epoch/loss/accuracy; the path has encoder/subject/timestamp. Seed, epoch budget,
learning rate, channel count, and the code version are all unrecorded, so the
only way to know whether two runs are comparable is to remember which sweep was
which.

That has already produced one wrong table (the locked pair resolved to seed 44
while the panel encoders were seed 42, giving ATMS 31.2% next to a published
33.7%) and one near-miss (panel encoders trained to convergence at a 300-epoch
cap while the locked pair used a fixed 40). Both were caught by memory. The next
one -- lr, n_times, avg_trials, a changed eval -- would not be.

Dumping the resolved args covers every axis at once, including axes nobody has
thought of yet, instead of adding one column per bug.

USAGE  (in train_unified.py, immediately after the run directory is created):

    from run_config import dump_run_config
    dump_run_config(run_dir, args)

That is the whole integration. It never raises: a failure to record config must
not kill a training run, so every failure path degrades to a written file with
an "unknown" field rather than an exception.
"""
import json
import os
import subprocess
import sys
import time


def _git(*cmd, cwd=None):
    """Run a git command, returning None on any failure (git missing, not a
    repo, detached, permissions). Never raises."""
    try:
        out = subprocess.run(("git",) + cmd, cwd=cwd, capture_output=True,
                             text=True, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def git_provenance(cwd=None):
    """Commit + dirty flag. A commit hash alone is misleading when the working
    tree has uncommitted changes: it names code that is not what ran."""
    commit = _git("rev-parse", "--short", "HEAD", cwd=cwd)
    if commit is None:
        return {"git_commit": "unknown", "git_dirty": "unknown",
                "git_branch": "unknown"}
    status = _git("status", "--porcelain", cwd=cwd)
    return {
        "git_commit": commit,
        # dirty means the recorded commit does NOT fully describe the run
        "git_dirty": bool(status) if status is not None else "unknown",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd) or "unknown",
    }


def dump_run_config(run_dir, args, extra=None, filename="run_config.json"):
    """Write the resolved configuration for this run into run_dir.

    args   : the argparse Namespace (or any object with __dict__, or a dict)
    extra  : optional dict of anything else worth pinning (dataset path hashes,
             number of trials loaded, resolved device, ...)

    Returns the path written, or None if it could not be written. Never raises.
    """
    try:
        if isinstance(args, dict):
            cfg = dict(args)
        else:
            cfg = dict(vars(args))
    except Exception:
        cfg = {"_args_unreadable": True}

    cfg.update(git_provenance())
    cfg["_recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg["_python"] = sys.version.split()[0]
    try:
        import torch
        cfg["_torch"] = torch.__version__
        cfg["_cuda"] = torch.version.cuda
        cfg["_gpu"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else None)
    except Exception:
        pass
    try:
        import braindecode
        cfg["_braindecode"] = braindecode.__version__
    except Exception:
        pass

    if extra:
        try:
            cfg.update(extra)
        except Exception:
            pass

    try:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        with open(path, "w") as fh:
            # default=str so a Path, device, or tensor never breaks the dump
            json.dump(cfg, fh, indent=2, sort_keys=True, default=str)
        return path
    except Exception as e:
        # last resort: say so on stdout, but do not take the run down
        print(f"[run_config] WARNING: could not write config: "
              f"{type(e).__name__}: {e}")
        return None


def load_run_config(run_dir, filename="run_config.json"):
    """Read a run's config, or None if absent/unreadable (legacy runs)."""
    path = os.path.join(run_dir, filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


if __name__ == "__main__":
    # smoke test
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--encoder_type", default="ATMS")
    a = p.parse_args([])
    out = dump_run_config("/tmp/_rc_smoke", a, extra={"n_trials": 132320})
    print("wrote:", out)
    print(json.dumps(load_run_config("/tmp/_rc_smoke"), indent=2)[:400])
