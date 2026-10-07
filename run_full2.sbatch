#!/usr/bin/env bash
#SBATCH --job-name=agrimorph_full2
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=agrimorph_full2_%j.out
#SBATCH --error=agrimorph_full2_%j.out
set -euo pipefail

export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=""     # CPU: torch import + embed; slow /work FS is the bottleneck, not compute

export AGRI_CLIP_CKPT="$AGRI_ROOT/weights/open_clip_vitb32_laion2b.bin"
export AGRI_FIGS="$AGRI_ROOT/figures"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== node $(hostname) | start $(date) ==="
# single torch import for BOTH embedding stages (features + seq crops)
echo "=== [1/3] consolidated CLIP embedding ($(date)) ==="; python src/run_embed.py
echo "=== [2/3] perception + decision replay ($(date)) ==="; python src/exp.py
echo "=== [3/3] all poster figures ($(date)) ==="; python src/make_figs.py
echo "=== collect ($(date)) ==="; mkdir -p results
cp -f results_*.json results_*.csv perception_out.npz traces.npy clip_text.npy patches_all.csv results/ 2>/dev/null || true
echo "=== DONE $(date) ==="; ls -la figures/*.png 2>/dev/null
