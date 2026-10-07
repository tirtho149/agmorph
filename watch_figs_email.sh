#!/usr/bin/env bash
# Watch the scavenger figures job; email a verified success/failure summary.
JID=16629543
ROOT="/work/mech-ai-scratch/tirtho/agrimorph_gpu_bundle"
TO="tirtho@iastate.edu"
started=0
for i in $(seq 1 240); do   # up to ~4h
  st=$(squeue -j "$JID" -h -o "%T" 2>/dev/null)
  if [ "$st" = "RUNNING" ] && [ "$started" = "0" ]; then
    started=1
    node=$(squeue -j "$JID" -h -o "%R" 2>/dev/null)
    { echo "To: $TO"
      echo "Subject: AgriMorph figures — GOT GPU (job $JID on $node)"
      echo "Content-Type: text/plain"; echo
      echo "The AgriMorph embedding-figures job started on $node at $(date)."
      echo "Rendering r2/r2b/r3/r4/r5 (+refresh). You'll get a second email when done."
    } | /usr/sbin/sendmail -t 2>/dev/null || true
  fi
  [ -z "$st" ] && break
  sleep 60
done
# final state
state=$(sacct -j "$JID" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
npng=$(ls "$ROOT"/figures/*.png 2>/dev/null | wc -l)
if echo "$state" | grep -qi COMPLETED; then
  subj="AgriMorph figures — SUCCESSFULLY DONE (job $JID)"
  body="All AgriMorph figures rendered successfully at $(date). State=$state."
else
  subj="AgriMorph figures — job $JID ended: $state"
  body="Job $JID ended with state=$state at $(date). Check agri_figs_scav_${JID}.out."
fi
{ echo "To: $TO"; echo "Subject: $subj"; echo "Content-Type: text/plain"; echo
  echo "$body"; echo
  echo "Figures directory ($npng PNGs):"
  ls -1 "$ROOT"/figures/*.png 2>/dev/null
  echo
  echo "Tail of job log:"
  tail -15 "$ROOT/agri_figs_scav_${JID}.out" 2>/dev/null
} | /usr/sbin/sendmail -t 2>/dev/null || true
echo "watcher done: state=$state pngs=$npng"
