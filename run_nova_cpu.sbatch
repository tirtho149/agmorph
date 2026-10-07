#!/usr/bin/env bash
#SBATCH --job-name=agrimorph_cpu
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=agrimorph_cpu_%j.out
#SBATCH --error=agrimorph_cpu_%j.out
set -euo pipefail

export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"

# Self-activate the project venv (torch/open_clip/sklearn/matplotlib live here).
source /work/mech-ai-scratch/tirtho/.venv/bin/activate

# Offline: use the pre-fetched LAION ViT-B/32 checkpoint, no HF download.
export AGRI_CLIP_CKPT="$AGRI_ROOT/weights/open_clip_vitb32_laion2b.bin"
export AGRI_FIGS="$AGRI_ROOT/figures"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TOKENIZERS_PARALLELISM=false

echo "=== node $(hostname) | $(date) ==="
python - <<'P'
import torch, open_clip, sklearn, matplotlib
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("open_clip", open_clip.__version__, "| sklearn", sklearn.__version__, "| mpl", matplotlib.__version__)
P

echo "=== [1/4] CLIP image + text embeddings for patches ==="
python src/features.py

echo "=== [2/4] CLIP embeddings for test-sequence crops (4 views/frame) ==="
python src/seq_feats.py --embed

echo "=== [3/4] perception metrics + decision replay ==="
python src/exp.py

echo "=== [4/4] poster figures ==="
python src/make_figs.py

echo "=== collect results ==="
mkdir -p results
cp -f results_*.json results_*.csv perception_out.npz traces.npy clip_text.npy results/ 2>/dev/null || true
cp -f patches_all.csv results/ 2>/dev/null || true
tar czf results.tgz results figures
echo "DONE $(date) -> results.tgz + figures/"
ls -la figures/