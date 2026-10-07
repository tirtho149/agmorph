"""Pilot-scope figure inputs: embed a SUBSAMPLE of patches + a subset of sequences
(creek + 2 id runs, full frames for the temporal figures), then run the perception +
decision experiment so make_figs can render all figures fast. Writes the same files
exp.py normally writes, just at pilot scale.

NOTE: overwrites patches_all.csv / seq_frames.csv / seq_crops.csv with pilot subsets
(originals backed up to *.full). Re-run the full pipeline to restore full scope.
"""
import os, sys, json, time, shutil
import numpy as np, pandas as pd
sys.path.insert(0, "src")
ROOT = os.environ["AGRI_ROOT"]
import features as ft
from exp import G, TRAIN, CAL, ID_TEST, OOD_TEST

t0 = time.time()

# ---- 1) patch subsample (include CAL so exp can set the gate threshold τ) ----
df = pd.read_csv(f"{ROOT}/patches.csv")
keep = set(TRAIN) | set(CAL) | set(ID_TEST) | set(OOD_TEST)
df = df[df.seq.isin(keep)].copy()
df["_ps"] = df.seq.map(lambda s: "train" if s in TRAIN else "cal" if s in CAL
                       else "id" if s in ID_TEST else "ood")
caps = {"train": 300, "cal": 200, "id": 200, "ood": 300}
sub = pd.concat([d.sample(min(len(d), caps[s]), random_state=0)
                 for (_, s), d in df.groupby(["group", "_ps"])]).reset_index(drop=True)
print(f"[{time.time()-t0:.0f}s] patch subsample: {len(sub)} "
      f"({dict(sub._ps.value_counts())})", flush=True)
_cf, _ct, _pa = f"{ROOT}/clip_feats.npy", f"{ROOT}/clip_text.npy", f"{ROOT}/patches_all.csv"
if os.path.exists(_cf) and os.path.exists(_pa) and \
        len(np.load(_cf)) == len(pd.read_csv(_pa)):
    # checkpoint: reuse patch embeddings from a prior (interrupted / pre-empted) run
    print(f"[{time.time()-t0:.0f}s] reusing cached clip_feats "
          f"({len(np.load(_cf))} patches) -- skipping patch embed", flush=True)
    if not os.path.exists(_ct):
        np.save(_ct, ft.embed_text())
else:
    sub.drop(columns="_ps").to_csv(_pa, index=False)
    X = ft.embed_images([f"{ROOT}/patches/{f}" for f in sub.file])
    np.save(_cf, X); np.save(_ct, ft.embed_text())
    print(f"[{time.time()-t0:.0f}s] patches embedded {X.shape}", flush=True)

# ---- 2) sequence subset: creek (ood) + 2 id runs, FULL frames (temporal figs) ----
for f in ("seq_frames.csv", "seq_crops.csv"):
    if not os.path.exists(f"{ROOT}/{f}.full"):
        shutil.copy(f"{ROOT}/{f}", f"{ROOT}/{f}.full")
sf = pd.read_csv(f"{ROOT}/seq_frames.csv.full")
keep_seqs = list(OOD_TEST) + ID_TEST[:2]
sf2 = sf[sf.seq.isin(keep_seqs)].reset_index(drop=True)
VIEWS = 4
crops = [f"{r.seq}_{r.frame[:-4]}_v{k}.jpg" for r in sf2.itertuples() for k in range(VIEWS)]
sf2.to_csv(f"{ROOT}/seq_frames.csv", index=False)
pd.Series(crops).to_csv(f"{ROOT}/seq_crops.csv", index=False, header=["file"])
print(f"[{time.time()-t0:.0f}s] seq subset: {len(sf2)} frames over {keep_seqs}", flush=True)
import seq_feats
seq_feats.embed()
print(f"[{time.time()-t0:.0f}s] seq crops embedded", flush=True)

# ---- 3) perception + decision replay (writes results_*.json/csv, traces.npy) ----
import exp
ms = exp.main()
exp.decisions(ms, json.load(open(f"{ROOT}/results_perception.json"))["tau"])
print(f"[{time.time()-t0:.0f}s] PILOT FIGURE INPUTS READY", flush=True)
