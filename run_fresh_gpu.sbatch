#!/usr/bin/env bash
#SBATCH --job-name=agri_fresh_gpu
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --exclude=nova26-gpu-1,nova26-gpu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=agri_fresh_gpu_%j.out
#SBATCH --error=agri_fresh_gpu_%j.out
set -euo pipefail

export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
rm -f FRESH_*.txt

# ---- fresh venv on NODE-LOCAL disk (fast imports, unlike the /work venv) ----
LOCAL="${TMPDIR:-/tmp}/agri_${SLURM_JOB_ID}"
echo "=== node $(hostname) | $(date) | local env: $LOCAL ==="
nvidia-smi -L || true
PYBASE="$(command -v python3.11 || command -v python3)"
echo "base python: $PYBASE ($($PYBASE --version 2>&1))"
$PYBASE -m venv "$LOCAL"
source "$LOCAL/bin/activate"
export PIP_CACHE_DIR="$LOCAL/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1
python -m pip install --upgrade pip >/dev/null 2>&1 || true

echo "=== pip installing torch + deps into local env ($(date)) ==="
if pip install --no-input torch open_clip_torch numpy pillow pandas scikit-learn matplotlib 2>&1 | tail -8; then
  echo "pip-install: OK" > "$AGRI_ROOT/FRESH_PIP_DONE.txt"
else
  echo "pip-install: FAILED (likely no internet on compute node)" > "$AGRI_ROOT/FRESH_PIP_FAILED.txt"; exit 1
fi

python -c "import torch,os; open(os.path.join(os.environ['AGRI_ROOT'],'FRESH_TORCH_OK.txt'),'w').write('torch %s cuda=%s ndev=%s\n'%(torch.__version__,torch.cuda.is_available(),torch.cuda.device_count()))"
cat "$AGRI_ROOT/FRESH_TORCH_OK.txt"

export AGRI_CLIP_CKPT="$AGRI_ROOT/weights/open_clip_vitb32_laion2b.bin"
export AGRI_FIGS="$AGRI_ROOT/figures"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=== [1/3] embedding ($(date)) ==="; python src/run_embed.py
echo "=== [2/3] perception + replay ($(date)) ==="; python src/exp.py
echo "=== [3/3] figures ($(date)) ==="; python src/make_figs.py
echo "=== DONE $(date) ==="; ls -la figures/*.png 2>/dev/null | wc -l
