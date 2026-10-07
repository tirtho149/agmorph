# AgriMorph

Terrain-adaptive morphing robot: a **frozen CLIP** terrain perceptron with an
**epistemic-gated** body-selection policy, evaluated on real off-road sequences
from the **RUGD** dataset.

A single forward-facing camera patch is embedded by a frozen vision encoder,
classified into a coarse *terrain group*, and — when the model is confident
enough — used to pick the robot *morphology* (body) that is best matched to that
terrain. Ensemble disagreement supplies the epistemic uncertainty that gates
whether the robot commits to a reconfiguration or holds its current body.

## Model

| Stage | Model / method |
|-------|----------------|
| **Vision encoder** | **CLIP ViT-B/32** (`open_clip`), pretrained `laion2b_s34b_b79k`, **frozen** — used only for L2-normalized image embeddings. |
| **Zero-shot baseline** | CLIP text prompts per terrain group (`a photo of gravel road`, `dense bushes`, `rocks`, `dirt trail`, `tree foliage`, …) scored against image embeddings. |
| **Terrain classifier** | **5-member bootstrap ensemble of logistic-regression linear probes** on the frozen embeddings (per-member `C ∈ {0.5,1,2,4,8}`, class-balanced, bootstrap-resampled). |
| **Uncertainty** | **BALD mutual information** (predictive entropy − mean member entropy) across the ensemble → epistemic signal used for a risk-coverage gate (threshold `τ` = 85th-percentile MI on a held-out calibration sequence). |
| **Embodiment policy** | Frame-by-frame replay over RUGD sequences; body chosen from a per-terrain **utility table** `U` with a reconfiguration switch cost, reactive-delay and predictive-horizon variants. |

### Classes

- **Terrain groups:** `firm`, `dense`, `rough`, `soft`, `canopy`
- **Robot morphologies:** `Wheel`, `Snake`, `Quadruped`, `Crawler`, `Extended`

### Data splits (RUGD sequences)

- **Train:** park-1/2, trail, trail-3/4/6/9/10/11/12
- **Calibration** (sets gate threshold `τ`): trail-14
- **ID test:** park-8, trail-5/7/13/15
- **OOD test:** creek

## Headline results

Full perception run (`results_perception.json`), balanced accuracy:

| Split | Zero-shot | Linear-probe ensemble |
|-------|-----------|------------------------|
| In-distribution | 0.73 | **0.91** |
| Out-of-distribution (creek) | 0.57 | **0.66** |

Ensemble-disagreement flags perception mistakes (ID AUROC ≈ 0.71 for
error detection). Pilot numbers in `PILOT_RESULTS.json`.

## Code layout

```
src/
  features.py   frozen CLIP ViT-B/32 embedding (parallel image loader for /work I/O)
  run_embed.py  consolidated single-torch-import embedding pass
  exp.py        perception (zero-shot vs ensemble) + epistemic gate + decision replay
  pilot.py      subsampled perception pilot -> PILOT_RESULTS.json
  build_patches.py / rare_patches.py   terrain patch extraction from RUGD frames
  seq_feats.py  per-sequence features for the decision replay
  make_figs.py / figs_rugd.py / poster_figs.py / poster_extra.py / pilot_*fig*.py   figure set
run_*.sbatch    Slurm launchers (GPU / CPU / pilot / figures)
PROGRESS.md     run log and HPC/Lustre notes
```

## Running

The pipeline is embedding → experiment → figures. On the cluster:

```bash
# env knobs
export AGRI_ROOT=/path/to/data            # holds patches_all.csv + frames
export AGRI_CLIP_CKPT=/path/to/clip.bin   # optional local CLIP ckpt (offline nodes)

sbatch run_fresh_gpu.sbatch   # embed -> exp -> make_figs (fresh node-local torch venv + GPU)
sbatch run_pilot_gpu.sbatch   # subsampled perception pilot -> PILOT_RESULTS.json
```

> **Not in git (large data/weights):** `dl/` (raw RUGD frames), `weights/`
> (CLIP checkpoint), `seqcrops/`, `patches/`, `RUGD_annotations/`, and all
> `*.npy/*.npz/*.zip` caches. Regenerate embeddings with the scripts above, or
> point `AGRI_ROOT`/`AGRI_CLIP_CKPT` at local copies.
