#!/usr/bin/env bash
#SBATCH --job-name=agri_figs_scav
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:1
#SBATCH --exclude=nova26-gpu-1,nova26-gpu-2
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:15:00
#SBATCH --output=agri_figs_scav_%j.out
#SBATCH --error=agri_figs_scav_%j.out
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=tirtho@iastate.edu
set -euo pipefail
export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
LOCAL="${TMPDIR:-/tmp}/agri_figs_${SLURM_JOB_ID}"
echo "=== node $(hostname) | $(date) | requeue-safe scavenger ==="; nvidia-smi -L || true
PYBASE="$(command -v python3.11 || command -v python3)"
$PYBASE -m venv "$LOCAL"; source "$LOCAL/bin/activate"
export PIP_CACHE_DIR="$LOCAL/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1
# PIN torch to a cu126 build: works on V100 (cc7.0) AND Ampere+ — avoids the
# non-deterministic default that pulled cu130 and crashed on the V100.
pip install --no-input --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 torchvision==0.23.0 2>&1 | tail -3
pip install --no-input open_clip_torch numpy pillow pandas scikit-learn matplotlib 2>&1 | tail -3
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available(),
      "| dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
PY
export AGRI_CLIP_CKPT="$AGRI_ROOT/weights/open_clip_vitb32_laion2b.bin"
export AGRI_FIGS="$AGRI_ROOT/figures"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== [1/2] pilot embed + perception/decision ($(date)) ==="; python src/pilot_figs.py
echo "=== [2/2] render all figures ($(date)) ==="; python src/make_figs.py
echo "=== DONE $(date) ==="; ls -la figures/*.png 2>/dev/null | wc -l; ls figures/*.png 2>/dev/null
