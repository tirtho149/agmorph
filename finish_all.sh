#!/usr/bin/env bash
# Autonomous end-to-end finisher for the AgriMorph poster run.
# Safe to run detached: logs to finish_all.log, writes STATUS_* markers.
set -uo pipefail
ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
cd "$ROOT"
LOG="$ROOT/finish_all.log"
EXPECT=5671298594
OLD_JOB=16611949
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
rm -f STATUS_DONE STATUS_FAIL
: > "$LOG"
say "=== finish_all starting ==="

# 1) wait for the orphaned assembly (cat) to finish frames.zip
say "waiting for frames.zip assembly to reach $EXPECT bytes ..."
for i in $(seq 1 240); do
  sz=$(stat -c%s dl/frames.zip 2>/dev/null || echo 0)
  running=$(pgrep -fc 'cat dl/part_0' || true); running=${running:-0}
  [ "$sz" -ge "$EXPECT" ] && break
  if [ "$running" -eq 0 ] && [ "$sz" -lt "$EXPECT" ]; then
    say "assembler gone with partial ($sz); re-catting parts ..."
    if [ "$(ls dl/part_* 2>/dev/null | wc -l)" -eq 16 ]; then cat dl/part_* > dl/frames.zip; fi
    break
  fi
  sleep 10
done
sz=$(stat -c%s dl/frames.zip 2>/dev/null || echo 0)
say "frames.zip size = $sz (expected $EXPECT)"

# 2) verify integrity
if [ "$sz" -ne "$EXPECT" ]; then say "SIZE MISMATCH"; fi
if unzip -t dl/frames.zip >/dev/null 2>&1; then
  say "zip integrity OK"
else
  say "ZIP INTEGRITY FAILED — aborting frame extraction (crop figs still produced by job)"
  FRAMES_OK=0
fi
FRAMES_OK=${FRAMES_OK:-1}

# 3) extract the 3 sequences the frame figures use
if [ "$FRAMES_OK" = 1 ]; then
  say "extracting creek / park-1 / trail-7 frames ..."
  unzip -oq dl/frames.zip \
    "RUGD_frames-with-annotations/creek/*" \
    "RUGD_frames-with-annotations/park-1/*" \
    "RUGD_frames-with-annotations/trail-7/*" -d . && say "extract done" || say "extract error"
  for s in creek park-1 trail-7; do
    say "  $s frames: $(ls RUGD_frames-with-annotations/$s 2>/dev/null | wc -l)"
  done
  # 4) free disk / stop future I/O pressure
  say "removing zip + parts to free space"
  rm -f dl/frames.zip dl/part_*
fi

# 5) cancel the wedged compute job (superseded by run_all)
if squeue -h -j "$OLD_JOB" >/dev/null 2>&1 && [ -n "$(squeue -h -j "$OLD_JOB" 2>/dev/null)" ]; then
  say "cancelling wedged job $OLD_JOB"
  scancel "$OLD_JOB" 2>/dev/null || true
  sleep 5
fi

# 6) submit the consolidated job on a now-quiet filesystem
say "submitting run_all.sbatch"
SUB=$(sbatch run_all.sbatch)
say "$SUB"
JOB=$(echo "$SUB" | grep -oE '[0-9]+' | tail -1)
echo "$JOB" > CONSOLIDATED_JOBID
say "consolidated job id = $JOB"

# 7) wait for it to finish
say "waiting for job $JOB to complete ..."
for i in $(seq 1 720); do   # up to ~4h at 20s cadence
  st=$(squeue -h -j "$JOB" -o "%T" 2>/dev/null)
  [ -z "$st" ] && { say "job $JOB left the queue"; break; }
  sleep 20
done

# 8) verify outputs
say "=== verification ==="
NF=$(ls figures/*.png 2>/dev/null | wc -l)
say "figure PNGs produced: $NF"
ls -la figures/ 2>/dev/null | tee -a "$LOG"
say "--- results files ---"
ls -la results/ 2>/dev/null | tee -a "$LOG"
say "--- tail of job log ---"
tail -30 agrimorph_all_${JOB}.out 2>/dev/null | tee -a "$LOG"

if [ "$NF" -ge 6 ] && [ -f results/results_perception.json ] && [ -f results/results_decision.csv ]; then
  say "=== ALL DONE: $NF figures + results present ==="
  touch STATUS_DONE
else
  say "=== INCOMPLETE: figures=$NF (see log) ==="
  touch STATUS_FAIL
fi
say "=== finish_all finished ==="
