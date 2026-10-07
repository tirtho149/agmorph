#!/usr/bin/env bash
#SBATCH --job-name=agrimorph_framefigs
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=agrimorph_framefigs_%j.out
#SBATCH --error=agrimorph_framefigs_%j.out
set -euo pipefail

export AGRI_ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$AGRI_ROOT"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export AGRI_FIGS="$AGRI_ROOT/figures"

echo "=== full-scene RGB figures (need RUGD frames+annotations) ==="
python - <<'P'
import sys, os
sys.path.insert(0, "src")
import figs_rugd as fr
for name, f in (("raw_frames_terrain_to_body", fr.fig_raw_frames),
                ("full_frame_filmstrip", fr.fig_filmstrip)):
    try:
        f(); print("OK", name, flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print("FAIL", name, repr(e), flush=True)
print("figures dir:", fr.OUT)
P
echo "=== refresh results tarball ==="
tar czf results.tgz results figures 2>/dev/null || tar czf results.tgz figures
ls -la figures/