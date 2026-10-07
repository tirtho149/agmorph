#!/usr/bin/env bash
#SBATCH --job-name=agrimorph_all
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
# CPU-only: torch cu128 (Jun-13 venv) hangs on CUDA init on the free newer GPUs
# (Blackwell rtx_pro_6000 nova26, Hopper h200 nova25); compatible A100s (nova21) are saturated.
# 17.5k ViT-B/32 embeds on 32 cores ~15-25 min, so CPU is the deterministic path here.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=agrimorph_all_%j.out
#SBATCH --error=agrimorph_all_%j.out
set -euo pipefail

export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=""   # force CPU — avoids the cu128 CUDA-init hang on newer GPU nodes

export AGRI_CLIP_CKPT="$AGRI_ROOT/weights/open_clip_vitb32_laion2b.bin"
export AGRI_FIGS="$AGRI_ROOT/figures"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1              # flush prints immediately so progress is visible

echo "=== node $(hostname) | start $(date) ==="
echo "--- nvidia-smi ---"; nvidia-smi -L || echo "(no nvidia-smi)"
# device probe: exits immediately so DEVICE.txt is flushed to disk within seconds
# (SLURM block-buffers stdout, so this file — not the .out — is the fast GPU check)
python -c "import os,torch; open(os.path.join(os.environ['AGRI_ROOT'],'DEVICE.txt'),'w').write('cuda=%s cudaver=%s ndev=%s\n'%(torch.cuda.is_available(),torch.version.cuda,torch.cuda.device_count()))"
python -c "import torch,open_clip,sklearn,matplotlib as m; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'| open_clip',open_clip.__version__,'| sklearn',sklearn.__version__,'| mpl',m.__version__)"

echo "=== [1/4] CLIP image + text embeddings for patches ($(date)) ==="
python src/features.py

echo "=== [2/4] CLIP embeddings for test-sequence crops ($(date)) ==="
python src/seq_feats.py --embed

echo "=== [3/4] perception metrics + decision replay ($(date)) ==="
python src/exp.py

echo "=== [4/4] all poster figures ($(date)) ==="
python src/make_figs.py

echo "=== collect results ($(date)) ==="
mkdir -p results
cp -f results_*.json results_*.csv perception_out.npz traces.npy clip_text.npy patches_all.csv results/ 2>/dev/null || true
tar czf results.tgz results figures
echo "=== DONE $(date) ==="
echo "--- figures ---"; ls -la figures/
echo "--- results ---"; ls -la results/