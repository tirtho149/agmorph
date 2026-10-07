# AgriMorph poster figures — progress

_Last updated: 2026-09-28, session with Claude._

## Goal
Produce 8 SAGE-parity poster figures for **AgriMorph** (terrain-adaptive robot: frozen
CLIP ViT-B/32 → 5-member bootstrap linear-probe ensemble → epistemic-gated body choice
on RUGD off-road runs). Plus a quick **pilot** reporting headline perception numbers.

## Root-cause of the whole ordeal
The `/work` Lustre filesystem is **severely I/O-degraded** for small-file reads:
- `import numpy` ≈ 40 s, `import torch` from the `/work` venv **hangs** (46 min, 6 s CPU used).
- Reading images off `/work` ≈ **0.58 img/s** (200-image read test timed out at 120 s).
- Metadata (`ls`) is fine — it's *content reads* that crawl.

### What did NOT work
- `/work` venv (`torch 2.10+cu128`): import hangs on every node tried (Blackwell nova26,
  H200 nova25, CPU swift node). Not GPU-specific — it's the FS.
- Spack `py-torch` modules: **silently refuse to load** ("No modules loaded"); combining
  multiple `py-*` modules conflicts (different pinned numpy).

### What WORKS ✅
- **Fresh venv on node-local disk** inside the job (`/tmp/...`): `pip install torch ...`
  (compute nodes have internet) → torch imports instantly, `cuda=True`. This is the fix.
- **Spack `py-matplotlib` (solo)** for torch-free figures: imports in <1 s.
- **Parallel image loader** (`features.embed_images`, 64 threads) to hide `/work` read
  latency — turns ~0.6 img/s serial into tens/s.

## Figures (8 total)
Rendered to `figures/` (600-dpi PNG + vector PDF).

| # | figure | needs embeddings? | status |
|---|--------|-------------------|--------|
| r0 | `methodology_flow` (2-stage: curation+training / embodied deployment, icons, trace) | no | ✅ done (spack) |
| r1 | `morphology_samples` (5 terrain→body classes, real patches) | no | ✅ done (spack) |
| r6 | `dataset_split` (train/cal/test patch counts) | no | ✅ done (spack) |
| r2b | `results_table` (decision-level utility + safety) | yes | ⏳ pending embeddings |
| r2 | `quantitative_results` | yes | ⏳ pending embeddings |
| r4 | `confusion` | yes | ⏳ pending embeddings |
| r5 | `creek_timeline` | yes | ⏳ pending embeddings |
| r3 | `qualitative_filmstrip_crop` | yes | ⏳ pending embeddings |

## Jobs (Slurm)
- **Full figures:** `run_fresh_gpu.sbatch` → job `16619371` (fresh node-local venv + GPU +
  parallel loader; runs embed → exp → make_figs). Walltime 50 min.
- **Pilot numbers:** `run_pilot_gpu.sbatch` → job `16619397` (subsample ~2.5k patches,
  perception-only). Emails `tirtho@iastate.edu` on completion (Slurm `--mail` + sendmail
  of the numbers). Walltime 28 min. Output → `PILOT_RESULTS.json`.
- Both were `PD` (GPU-contended); scheduler worst-case ~9 h but short walltimes qualify
  for backfill (earlier run backfilled in ~1 min).

## Key files
- `src/poster_extra.py` — methodology flow + results table (SAGE style, icons).
- `src/features.py` — CLIP embedding (now parallel image loader).
- `src/run_embed.py` — consolidated single-torch-import embedding.
- `src/exp.py` — perception + decision replay (sklearn imported lazily).
- `src/pilot.py` — pilot perception numbers → `PILOT_RESULTS.json`.
- `src/make_figs.py` / `src/figs_rugd.py` — the figure set.

## How to check status / re-run
```bash
cd /work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle
squeue -u tirtho | grep agri            # job status
cat PILOT_RESULTS.json                   # pilot numbers (when done)
ls figures/*.png                         # finished figures
sbatch run_fresh_gpu.sbatch              # re-run full pipeline
sbatch run_pilot_gpu.sbatch              # re-run pilot (emails on done)
```
