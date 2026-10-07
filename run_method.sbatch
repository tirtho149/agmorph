#!/usr/bin/env bash
#SBATCH --job-name=agri_method
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=agri_method_%j.out
#SBATCH --error=agri_method_%j.out
set -euo pipefail
export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export AGRI_FIGS="$AGRI_ROOT/figures"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "=== node $(hostname) | start $(date) ==="
# matplotlib-only, NO torch/sklearn — the SAGE-parity methodology flow needs no data
python -c "import sys; sys.path.insert(0,'src'); import poster_extra as pe; pe.fig_methodology(); print('methodology figure done', flush=True)"
echo "=== DONE $(date) ==="; ls -la figures/*.png 2>/dev/null
