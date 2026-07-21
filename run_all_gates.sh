#!/bin/bash
# =============================================================================
# run_all_gates.sh -- ONE COMMAND, ONE VERDICT.
#
# WHERE TO RUN THIS:
#   On your LOCAL machine (the RTX 4080), in Git Bash, FROM THE REPO ROOT:
#       cd ~/Desktop/EEG_Image_decode-develop
#       bash run_all_gates.sh
#   NOT on ARCC. The whole point is to prove things work before you upload.
#
# WHAT IT DOES: runs every pre-sendoff gate in dependency order and prints a
# single PASS/FAIL summary. Nothing goes to the cluster until this is green.
#
#   Gate 0  env + GPU visible to torch
#   Gate 1  braindecode compat patch applied (EEGNetv4 -> EEGNet rename)
#   Gate 2  every encoder INSTANTIATES through the real registry
#   Gate 3  CBraMod pretrained weights are REAL, not random  [needs internet]
#   Gate 4  Block B encoders TRAIN on real THINGS data (2 epochs, sub-01)
#   Gate 5  LOSO classification probe runs (2 subjects, smoke)
#
# Gates are independent: a failure is reported and the run continues, so you get
# the full picture in one pass instead of whack-a-mole.
#
# FLAGS:
#   --quick     skip Gates 4 and 5 (the slow ones); structure checks only
#   --gate N    run only gate N
# =============================================================================
set -uo pipefail

QUICK=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --gate)  shift; ONLY="$1" ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
  shift
done

# ---- paths: edit here if your layout differs -------------------------------
: "${THINGS_DATA:=./EEG_Image_decode/Preprocessed_data_250Hz}"
: "${THINGS_IMG_TRAIN:=./image_set/training_images}"
: "${THINGS_IMG_TEST:=./image_set/test_images}"
FEATS="./emb_eeg"
export THINGS_DATA THINGS_IMG_TRAIN THINGS_IMG_TEST

RESULTS=()
note()  { echo ""; echo "=============================================================="; echo " $1"; echo "=============================================================="; }
record(){ RESULTS+=("$1|$2"); }
skip()  { [ -n "$ONLY" ] && [ "$ONLY" != "$1" ]; }

# ---------------------------------------------------------------- Gate 0 ----
if ! skip 0; then
note "GATE 0: environment + GPU"
python - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available()
    print("torch", torch.__version__, "| cuda:", ok, "|", torch.cuda.get_device_name(0) if ok else "NO GPU")
    sys.exit(0 if ok else 1)
except Exception as e:
    print("FAIL:", e); sys.exit(1)
PY
[ $? -eq 0 ] && { echo "GATE 0 PASS"; record 0 PASS; } || { echo "GATE 0 FAIL"; record 0 FAIL; }
fi

# ---------------------------------------------------------------- Gate 1 ----
if ! skip 1; then
note "GATE 1: braindecode compat (EEGNetv4 -> EEGNet rename)"
python - <<'PY'
import sys
try:
    import braindecode
    from braindecode import models as M
    v = braindecode.__version__
    has_old = hasattr(M, "EEGNetv4"); has_new = hasattr(M, "EEGNet")
    print(f"braindecode {v} | EEGNetv4={has_old} EEGNet={has_new}")
    if not has_old and not has_new:
        print("FAIL: neither EEGNetv4 nor EEGNet found"); sys.exit(1)
    # does the repo's encoder module import cleanly?
    sys.path.insert(0, "Retrieval"); sys.path.insert(0, ".")
    import eeg_encoders  # noqa
    print("eeg_encoders imports OK")
    sys.exit(0)
except ImportError as e:
    if "EEGNetv4" in str(e):
        print("FAIL: the EEGNetv4 rename bites here.")
        print("  FIX: python scripts/patch_braindecode_compat.py Retrieval/eeg_encoders.py")
    else:
        print("FAIL:", e)
    sys.exit(1)
except Exception as e:
    print("FAIL:", type(e).__name__, e); sys.exit(1)
PY
[ $? -eq 0 ] && { echo "GATE 1 PASS"; record 1 PASS; } || { echo "GATE 1 FAIL"; record 1 FAIL; }
fi

# ---------------------------------------------------------------- Gate 2 ----
if ! skip 2; then
note "GATE 2: every encoder instantiates through the real registry"
python - <<'PY'
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "Retrieval"); sys.path.insert(0, ".")
import torch
try:
    from eeg_encoders import build_encoder
except Exception as e:
    print("FAIL importing build_encoder:", e); sys.exit(1)

try:
    from eeg_encoders import ENCODER_REGISTRY
    have_cbramod = 'CBraMod_Encoder' in ENCODER_REGISTRY
except Exception:
    have_cbramod = False

want = ['ATMS','LaBraM_ATMS','EEGNetv4_Encoder','EEGConformer_Encoder',
        'ShallowFBCSPNet_Encoder']
if have_cbramod:
    want.append('CBraMod_Encoder')
else:
    print("  SKIP CBraMod_Encoder (needs braindecode>=1.6; yours is older)")
    print("       Block B does NOT need it. This is not a failure.")
x = torch.randn(2, 63, 250)
bad = 0
for n in want:
    try:
        m = build_encoder(n, n_chans=63, n_times=250); m.eval()
        with torch.no_grad():
            try: out = m(x, torch.zeros(2, dtype=torch.long))
            except TypeError: out = m(x)
        p = sum(t.numel() for t in m.parameters())/1e6
        ok = hasattr(out,'shape') and out.shape[-1] == 1024
        print(f"  {'OK  ' if ok else 'BAD '} {n:26s} {p:6.2f}M out={tuple(out.shape)}")
        if not ok: bad += 1
    except FileNotFoundError as e:
        # CBraMod registered but weights not downloaded -> not a code failure.
        print(f"  SKIP {n:26s} weights not downloaded (see Gate 3)")
    except Exception as e:
        print(f"  FAIL {n:26s} {type(e).__name__}: {str(e)[:90]}")
        bad += 1
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && { echo "GATE 2 PASS"; record 2 PASS; } || { echo "GATE 2 FAIL (see above)"; record 2 FAIL; }
fi

# ---------------------------------------------------------------- Gate 3 ----
if ! skip 3; then
note "GATE 3: CBraMod pretrained weights are REAL (not random)"
echo "  [needs the official CBraMod repo + weights: bash setup_cbramod.sh]"
python - <<'PY'
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "Retrieval"); sys.path.insert(0, ".")
try:
    from cbramod_encoder import CBRAMOD_AVAILABLE, CBRAMOD_REPO_PATH, verify_pretrained_load
except Exception as e:
    print("cannot import cbramod_encoder:", e)
    print("  (place src/cbramod_encoder.py in Retrieval/)")
    sys.exit(2)
if not CBRAMOD_AVAILABLE:
    print("SKIP: official CBraMod repo not found.")
    print("      Run: bash setup_cbramod.sh   (clones repo + fetches weights)")
    print("      Block B is unaffected by this.")
    sys.exit(2)
print(f"  repo: {CBRAMOD_REPO_PATH}")
try:
    sys.exit(0 if verify_pretrained_load() else 1)
except Exception as e:
    msg = str(e)
    if "weights not found" in msg or "FileNotFound" in type(e).__name__:
        print("SKIP: CBraMod repo is present but the pretrained weights are not.")
        print("      Download pretrained_weights.pth from:")
        print("        https://huggingface.co/weighting666/CBraMod")
        print("      -> third_party/CBraMod/pretrained_weights/pretrained_weights.pth")
        print("      Block B is unaffected; CBraMod simply cannot run until then.")
        sys.exit(2)
    print("FAIL:", type(e).__name__, msg[:200])
    sys.exit(1)
PY
rc=$?
if [ $rc -eq 0 ]; then echo "GATE 3 PASS"; record 3 PASS
elif [ $rc -eq 2 ]; then echo "GATE 3 SKIP (CBraMod unavailable -- not a blocker for Block B)"; record 3 SKIP
else echo "GATE 3 FAIL -- do NOT trust any CBraMod number until this passes"; record 3 FAIL; fi
fi

# ---------------------------------------------------------------- Gate 4 ----
if ! skip 4 && [ $QUICK -eq 0 ]; then
note "GATE 4: Block B encoders TRAIN on real data (2 epochs, sub-01)"
fails=0
ENCS="EEGNetv4_Encoder EEGConformer_Encoder ShallowFBCSPNet_Encoder"
if python -c "import sys; sys.path.insert(0,'Retrieval'); from eeg_encoders import ENCODER_REGISTRY; sys.exit(0 if 'CBraMod_Encoder' in ENCODER_REGISTRY else 1)" 2>/dev/null; then
  ENCS="$ENCS CBraMod_Encoder"
else
  echo "  (CBraMod_Encoder not registered -- skipping it; needs braindecode>=1.6)"
fi
for ENC in $ENCS; do
  echo ""
  echo "  ---- $ENC ----"
  LOG="gate4_${ENC}.log"
  python Retrieval/train_unified.py \
    --encoder_type "$ENC" --mode intra --dataset things \
    --data_path "$THINGS_DATA" --img_dir_training "$THINGS_IMG_TRAIN" \
    --img_dir_test "$THINGS_IMG_TEST" --clip_features_dir "$FEATS" \
    --n_chans 63 --n_times 250 --subjects sub-01 \
    --epochs 2 --batch_size 128 --lr 2e-4 --val_ratio 0.1 > "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    if grep -q "weights not found" "$LOG"; then
      echo "    SKIP: pretrained weights not downloaded (see Gate 3). Not a code failure."
    else
      echo "    CRASHED. tail:"; tail -6 "$LOG"; fails=$((fails+1))
    fi
  else
    v2=$(grep -oE "v2=[0-9.]+" "$LOG" | tail -1 | cut -d= -f2)
    echo "    trained OK  final v2=${v2:-NA} (chance=0.50)"
  fi
done
[ $fails -eq 0 ] && { echo ""; echo "GATE 4 PASS (no crashes)"; record 4 PASS; } || { echo ""; echo "GATE 4 FAIL ($fails crashed)"; record 4 FAIL; }
fi

# ---------------------------------------------------------------- Gate 5 ----
if ! skip 5 && [ $QUICK -eq 0 ]; then
note "GATE 5: LOSO classification probe runs (2-subject smoke)"
python probe_loso_classify.py --encoder ATMS --subjects 1-2 \
       --out gate5_probe_smoke.csv > gate5.log 2>&1
if [ $? -eq 0 ]; then
  tail -4 gate5.log; echo "GATE 5 PASS"; record 5 PASS
else
  echo "  FAILED. tail:"; tail -8 gate5.log; echo "GATE 5 FAIL"; record 5 FAIL
fi
fi

# ---------------------------------------------------------------- verdict ---
note "VERDICT"
green=1
for r in "${RESULTS[@]}"; do
  g="${r%%|*}"; s="${r##*|}"
  printf "  Gate %-2s %s\n" "$g" "$s"
  [ "$s" = "FAIL" ] && green=0
done
echo ""
if [ $green -eq 1 ] && [ ${#RESULTS[@]} -gt 0 ]; then
  echo "  ALL GATES GREEN -- safe to stage to ARCC."
  echo "  Next: upload repo+data, then sbatch slurm/preflight.sbatch on the cluster."
else
  echo "  NOT READY. Fix the FAIL gates above before sendoff."
  echo "  (Gate 3 FAIL = CBraMod may be running on random weights. Blocker.)"
  echo "  (Gate 3 SKIP = fine: run setup_cbramod.sh to enable it; Block B unaffected.)"
fi
