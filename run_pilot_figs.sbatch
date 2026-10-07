#!/usr/bin/env bash
#SBATCH --job-name=agri_pilot_figs
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --exclude=nova26-gpu-1,nova26-gpu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=agri_pilot_figs_%j.out
#SBATCH --error=agri_pilot_figs_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu
set -euo pipefail
export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
LOCAL="${TMPDIR:-/tmp}/agri_pf_${SLURM_JOB_ID}"
echo "=== node $(hostname) | $(date) ==="; nvidia-smi -L || true
PYBASE="$(command -v python3.11 || command -v python3)"
$PYBASE -m venv "$LOCAL"; source "$LOCAL/bin/activate"
export PIP_CACHE_DIR="$LOCAL/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1
pip install --no-input torch open_clip_torch numpy pillow pandas scikit-learn matplotlib 2>&1 | tail -3
export AGRI_CLIP_CKPT="$AGRI_ROOT/weights/open_clip_vitb32_laion2b.bin"
export AGRI_FIGS="$AGRI_ROOT/figures"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "=== [1/2] pilot embed + perception/decision ($(date)) ==="; python src/pilot_figs.py
echo "=== [2/2] render all figures ($(date)) ==="; python src/make_figs.py
echo "=== DONE $(date) ==="; ls -la figures/*.png 2>/dev/null | wc -l; ls figures/*.png 2>/dev/null
