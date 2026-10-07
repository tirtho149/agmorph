"""Consolidated CLIP embedding: import torch/open_clip ONCE (the /work FS makes
each fresh torch import cost tens of minutes), then embed patches + seq crops in
a single process."""
import os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.environ["AGRI_ROOT"]

t0 = time.time()
print(f"[{time.time()-t0:6.1f}s] importing features (builds frozen CLIP once)...", flush=True)
import features                     # top-level: loads torch, open_clip, builds ViT-B/32
print(f"[{time.time()-t0:6.1f}s] model ready", flush=True)

# ---- patches ----
df = pd.read_csv(f"{ROOT}/patches.csv")
if os.path.exists(f"{ROOT}/patches_rare.csv"):
    df = pd.concat([df, pd.read_csv(f"{ROOT}/patches_rare.csv")], ignore_index=True)
    df.to_csv(f"{ROOT}/patches_all.csv", index=False)
else:
    df.to_csv(f"{ROOT}/patches_all.csv", index=False)
print(f"[{time.time()-t0:6.1f}s] embedding {len(df)} patches...", flush=True)
X = features.embed_images([f"{ROOT}/patches/{f}" for f in df.file])
np.save(f"{ROOT}/clip_feats.npy", X)
np.save(f"{ROOT}/clip_text.npy", features.embed_text())
print(f"[{time.time()-t0:6.1f}s] patches done -> clip_feats {X.shape}", flush=True)

# ---- sequence crops (reuses the already-imported features module) ----
import seq_feats
print(f"[{time.time()-t0:6.1f}s] embedding sequence crops...", flush=True)
seq_feats.embed()
print(f"[{time.time()-t0:6.1f}s] EMBED DONE", flush=True)
