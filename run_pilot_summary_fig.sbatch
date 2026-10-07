#!/usr/bin/env bash
#SBATCH --job-name=agri_pilotfig
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=agri_pilotfig_%j.out
#SBATCH --error=agri_pilotfig_%j.out
set -euo pipefail
export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
module purge 2>/dev/null || true
module load py-matplotlib/3.10.1-py311-luy2jpy py-pandas/2.2.3-py311-girnx7x
export AGRI_FIGS="$AGRI_ROOT/figures"; export MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "=== node $(hostname) | $(date) | $(python --version 2>&1) ==="
python src/pilot_summary_fig.py
echo "=== DONE $(date) ==="; ls -la figures/r7_pilot_perception.* 2>/dev/null
