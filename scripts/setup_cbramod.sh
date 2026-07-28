#!/bin/bash
# =============================================================================
# setup_cbramod.sh -- one-time prep for the second foundation model.
#
# Clones the OFFICIAL CBraMod repo (MIT) and puts the pretrained weights where
# cbramod_encoder.py expects them. Does NOT touch braindecode: your 0.8 pin and
# every Block B wrapper stay exactly as they are.
#
# RUN FROM THE REPO ROOT:
#     bash setup_cbramod.sh
#
# WHY NOT braindecode's CBraMod: braindecode>=1.6 has the architecture, but its
# HF repo is ARCHITECTURE-ONLY -- no pretrained weights. And upgrading
# braindecode 0.8 -> 1.6 would break EEGNetv4/EEGConformer/EEGITNet/
# ShallowFBCSPNet/ATCNet/ATM_E, which pass the removed kwargs in_chans,
# n_classes, input_window_samples, add_log_softmax. The last one silently
# changes the embeddings. So: official repo, no upgrade.
# =============================================================================
set -uo pipefail

REPO_DIR="third_party/CBraMod"
WEIGHTS_DIR="${REPO_DIR}/pretrained_weights"
WEIGHTS="${WEIGHTS_DIR}/pretrained_weights.pth"
HF_URL="https://huggingface.co/weighting666/CBraMod"

echo "=============================================================="
echo " CBraMod setup (second foundation model)"
echo "=============================================================="

# ---- 1. einops (the repo needs it) ----
echo ""
echo "[1/4] einops"
python -c "import einops" 2>/dev/null && echo "  already installed" || {
  echo "  installing..."; pip install einops || { echo "  FAILED"; exit 1; }
}

# ---- 2. clone ----
echo ""
echo "[2/4] official repo -> ${REPO_DIR}"
if [ -d "${REPO_DIR}/models" ]; then
  echo "  already present"
else
  mkdir -p third_party
  git clone --depth 1 https://github.com/wjq-learning/CBraMod.git "${REPO_DIR}" || {
    echo "  CLONE FAILED. Clone manually:"
    echo "    git clone https://github.com/wjq-learning/CBraMod.git ${REPO_DIR}"
    exit 1
  }
  echo "  cloned"
fi

# ---- 3. weights ----
echo ""
echo "[3/4] pretrained weights"
mkdir -p "${WEIGHTS_DIR}"
if [ -f "${WEIGHTS}" ]; then
  sz=$(du -h "${WEIGHTS}" | cut -f1)
  echo "  already present (${sz})"
else
  echo "  MISSING. Download 'pretrained_weights.pth' from:"
  echo "      ${HF_URL}"
  echo "  and place it at:"
  echo "      ${WEIGHTS}"
  echo ""
  echo "  Option A (huggingface_hub, no braindecode involved):"
  echo "      pip install huggingface_hub"
  echo "      python -c \"from huggingface_hub import hf_hub_download; import shutil; \\"
  echo "        p=hf_hub_download('weighting666/CBraMod','pretrained_weights.pth'); \\"
  echo "        shutil.copy(p,'${WEIGHTS}'); print('saved')\""
  echo ""
  echo "  Option B: download in a browser from the HF page above."
  echo ""
  echo "  Re-run this script once the file is in place."
fi

# ---- 4. verify ----
echo ""
echo "[4/4] verify"
if [ ! -f "${WEIGHTS}" ]; then
  echo "  SKIPPED (weights not downloaded yet)."
  echo ""
  echo "  Architecture-only check:"
  CBRAMOD_REPO="$(pwd)/${REPO_DIR}" python Retrieval/cbramod_encoder.py 2>&1 | tail -3
  exit 0
fi

export CBRAMOD_REPO="$(pwd)/${REPO_DIR}"
python - <<'PY'
import sys
sys.path.insert(0, "Retrieval"); sys.path.insert(0, ".")
from cbramod_encoder import verify_pretrained_load
ok = verify_pretrained_load()
sys.exit(0 if ok else 1)
PY
rc=$?
echo ""
if [ $rc -eq 0 ]; then
  echo "=============================================================="
  echo " CBraMod READY. Add to your shell (or run_all_gates picks the"
  echo " default path up automatically):"
  echo "     export CBRAMOD_REPO=\"$(pwd)/${REPO_DIR}\""
  echo "=============================================================="
else
  echo "=============================================================="
  echo " VERIFY FAILED -- do NOT train CBraMod until this passes."
  echo " A 'foundation model' on random weights produces meaningless"
  echo " numbers that look perfectly plausible."
  echo "=============================================================="
fi
exit $rc
