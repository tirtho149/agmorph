#!/usr/bin/env bash
#SBATCH --job-name=agri_pilot
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --exclude=nova26-gpu-1,nova26-gpu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:28:00
#SBATCH --output=agri_pilot_%j.out
#SBATCH --error=agri_pilot_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu
set -euo pipefail
export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"; rm -f PILOT_RESULTS.json
LOCAL="${TMPDIR:-/tmp}/agri_pilot_${SLURM_JOB_ID}"
echo "=== node $(hostname) | $(date) ==="; nvidia-smi -L || true
PYBASE="$(command -v python3.11 || command -v python3)"
$PYBASE -m venv "$LOCAL"; source "$LOCAL/bin/activate"
export PIP_CACHE_DIR="$LOCAL/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1
pip install --no-input torch open_clip_torch numpy pillow pandas scikit-learn 2>&1 | tail -3
export AGRI_CLIP_CKPT="$AGRI_ROOT/weights/open_clip_vitb32_laion2b.bin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
echo "=== running pilot ($(date)) ==="
python src/pilot.py
echo "=== DONE $(date) ==="; cat PILOT_RESULTS.json 2>/dev/null

# email the actual pilot numbers (best-effort; Slurm --mail also notifies)
if [ -f PILOT_RESULTS.json ]; then
  { echo "To: tirtho@iastate.edu"
    echo "Subject: AgriMorph PILOT numbers ready (job ${SLURM_JOB_ID})"
    echo "Content-Type: text/plain"
    echo ""
    echo "AgriMorph perception pilot finished on $(hostname) at $(date)."
    echo "Balanced accuracy = zero-shot CLIP vs 5-member ensemble probe, per split."
    echo ""
    cat PILOT_RESULTS.json
    echo ""
    echo "Full results dir: $AGRI_ROOT/  (see PROGRESS.md)"
  } | /usr/sbin/sendmail -t || echo "sendmail failed (Slurm --mail END still notifies)"
fi
