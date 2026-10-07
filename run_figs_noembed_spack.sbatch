#!/usr/bin/env bash
#SBATCH --job-name=agri_noembed
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=agri_noembed_%j.out
#SBATCH --error=agri_noembed_%j.out
set -euo pipefail
export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
module purge 2>/dev/null || true
module load py-matplotlib/3.10.1-py311-luy2jpy py-pandas/2.2.3-py311-girnx7x
export AGRI_FIGS="$AGRI_ROOT/figures"; export MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "=== node $(hostname) | $(date) | $(python --version 2>&1) ==="
# figs that need only CSVs + patch images (NO torch embeddings): morphology samples + dataset split
python - <<'PY'
import sys; sys.path.insert(0, "src")
import figs_rugd as fr
for name, fn in (("r1 morphology_samples", fr.fig_morph_samples), ("r6 dataset_split", fr.fig_data_stats)):
    try:
        fn(); print("OK", name, flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print("FAIL", name, repr(e), flush=True)
PY
echo "=== DONE $(date) ==="; ls -la figures/r1_*.png figures/r6_*.png 2>/dev/null
