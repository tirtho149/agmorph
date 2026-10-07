#!/usr/bin/env bash
#SBATCH --job-name=agri_method_spack
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=agri_method_spack_%j.out
#SBATCH --error=agri_method_spack_%j.out
set -euo pipefail
export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
# spack matplotlib (autoloads a compatible numpy) — imports in <1s, no /work venv, no torch
module purge 2>/dev/null || true
module load py-matplotlib/3.10.1-py311-luy2jpy
export AGRI_FIGS="$AGRI_ROOT/figures"; export MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "=== node $(hostname) | $(date) | python $(python --version 2>&1) ==="
time python -c "import sys; sys.path.insert(0,'src'); import poster_extra as pe; pe.fig_methodology(); print('METHODOLOGY DONE', flush=True)"
echo "=== DONE $(date) ==="; ls -la figures/r0_methodology_flow.* 2>/dev/null
