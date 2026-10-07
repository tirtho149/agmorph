#!/usr/bin/env bash
#SBATCH --job-name=agrimorph_gpu
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=agrimorph_gpu_%j.out
#SBATCH --error=agrimorph_gpu_%j.out
set -euo pipefail

export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate

export AGRI_CLIP_CKPT="$AGRI_ROOT/weights/open_clip_vitb32_laion2b.bin"
export AGRI_FIGS="$AGRI_ROOT/figures"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== node $(hostname) | start $(date) ==="
nvidia-smi -L || echo "(no nvidia-smi)"
# device probe (writes fast on a working A100 -> confirms torch import is healthy)
python -c "import os,torch; open(os.path.join(os.environ['AGRI_ROOT'],'DEVICE.txt'),'w').write('cuda=%s cudaver=%s ndev=%s\n'%(torch.cuda.is_available(),torch.version.cuda,torch.cuda.device_count()))"

echo "=== [1/4] CLIP patch embeddings ($(date)) ==="; python src/features.py
echo "=== [2/4] CLIP seq-crop embeddings ($(date)) ==="; python src/seq_feats.py --embed
echo "=== [3/4] perception + decision replay ($(date)) ==="; python src/exp.py
echo "=== [4/4] all poster figures ($(date)) ==="; python src/make_figs.py
echo "=== collect ($(date)) ==="; mkdir -p results
cp -f results_*.json results_*.csv perception_out.npz traces.npy clip_text.npy patches_all.csv results/ 2>/dev/null || true
echo "=== DONE $(date) ==="; ls -la figures/*.png 2>/dev/null
