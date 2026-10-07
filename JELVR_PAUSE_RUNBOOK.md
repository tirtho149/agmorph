# Runbook — pause JELVR to run AgriMorph, then resume (safely)

_Context 2026-09-28: AgriMorph GPU jobs are queued behind priority; JELVR jobs
(`16592018`, `16592019`, `jelvr_leafnet`) hold 2× A100 on `nova21-gpu-9`, running 3.6 days._

## Which mechanism (and why)
- **`scontrol suspend` / `resume`** — the ONLY non-terminating option here.
  Cluster has `PreemptMode=GANG,SUSPEND`, so suspend is supported. Suspend = SIGSTOP:
  the process is frozen, **GPU memory is retained**, and `resume` continues exactly
  where it paused. JELVR is *not* terminated.
- **`scontrol requeue` — DO NOT USE.** The JELVR sbatch
  (`FLORA/sage_lvr/run_leafnet_arm_nova.sbatch`) trains with `--save_every_epoch` but
  has **no `--resume`** flag, so requeue restarts training from epoch 0 → loses 3.6 days.

## Risks
1. Suspending frees the GPU to the **whole queue** — another user's higher-priority job
   may grab it instead of AgriMorph. (Pausing JELVR ≠ guaranteeing AgriMorph runs.)
2. **Stuck-suspended risk:** if the resume step never runs (session dies, forgotten),
   JELVR sits paused forever, wasting the allocation. ALWAYS pair suspend with a
   guaranteed resume (see the safeguard below).
3. VRAM coexistence: if AgriMorph lands on the same physical GPU, its ~2 GB CLIP-inference
   footprint should fit alongside JELVR's frozen context on an 80 GB A100, but if not,
   *AgriMorph* OOMs (JELVR is unharmed — its VRAM is never freed).

## Procedure
```bash
JELVR="16592018 16592019"          # the two jelvr_leafnet job IDs (verify first!)
squeue -u tirtho | grep jelvr      # confirm they are still R on nova21-gpu-9

# 1) PAUSE (does not terminate; VRAM kept)
scontrol suspend $JELVR
squeue -u tirtho | grep jelvr      # STATE should show S (suspended)

# 2) run AgriMorph (already-queued jobs will now compete for the freed GPUs)
#    e.g. sbatch run_pilot_gpu.sbatch ; sbatch run_fresh_gpu.sbatch

# 3) RESUME as soon as AgriMorph is done (or immediately if it didn't get the GPU)
scontrol resume $JELVR
squeue -u tirtho | grep jelvr      # STATE back to R
```

## MANDATORY safeguard — guaranteed auto-resume
Run this detached so JELVR resumes even if the session dies (resumes when AgriMorph
finishes OR after a hard 60-min cap, whichever first):
```bash
JELVR="16592018 16592019"; AGRI=<agri_job_id>
nohup bash -c '
  for i in $(seq 1 180); do            # up to 60 min
    st=$(squeue -j '"$AGRI"' -h -o %t 2>/dev/null)
    [ -z "$st" ] && break              # agri finished -> resume now
    sleep 20
  done
  scontrol resume '"$JELVR"'
' >/tmp/jelvr_autoresume.log 2>&1 &
```

## Recommendation
Given JELVR = 3.6 days of work and AgriMorph pilot = ~15 min, the safer default is to
**let AgriMorph backfill on its own** (short walltime, any-GPU). Only suspend if you
accept risks 1–2 above AND run the mandatory auto-resume safeguard.
