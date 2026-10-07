#!/usr/bin/env bash
#SBATCH --job-name=agri_full_scav
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:1
#SBATCH --exclude=nova26-gpu-1,nova26-gpu-2
#SBATCH --requeue
#SBATCH --dependency=afterany:16629543
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=agri_full_scav_%j.out
#SBATCH --error=agri_full_scav_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=tirtho@iastate.edu
set -euo pipefail
export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
# The pilot job subsets seq_frames.csv / seq_crops.csv (backing up to *.full).
# Restore the FULL-scope lists before the full embed so we embed all sequences.
[ -f seq_frames.csv.full ] && cp -f seq_frames.csv.full seq_frames.csv
[ -f seq_crops.csv.full ]  && cp -f seq_crops.csv.full  seq_crops.csv
LOCAL="${TMPDIR:-/tmp}/agri_full_${SLURM_JOB_ID}"
echo "=== node $(hostname) | $(date) | FULL scope | requeue-safe scavenger ==="; nvidia-smi -L || true
PYBASE="$(command -v python3.11 || command -v python3)"
$PYBASE -m venv "$LOCAL"; source "$LOCAL/bin/activate"
export PIP_CACHE_DIR="$LOCAL/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1
# PIN torch cu126 (V100 + Ampere safe) — avoids the cu130 default that crashed.
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
# Separate figure dir so the pilot's figures/ set is preserved -> you get BOTH.
export AGRI_FIGS="$AGRI_ROOT/figures_full"; mkdir -p "$AGRI_FIGS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
echo "=== [1/3] embed FULL (all patches + all seq crops) ($(date)) ==="; python src/run_embed.py
echo "=== [2/3] perception + decision replay ($(date)) ==="; python src/exp.py
echo "=== [3/3] render figures ($(date)) ==="; python src/make_figs.py
echo "=== DONE $(date) ==="; ls -la "$AGRI_FIGS"/*.png 2>/dev/null | wc -l; ls "$AGRI_FIGS"/*.png 2>/dev/null
