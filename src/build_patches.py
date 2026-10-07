"""Cut single-terrain image patches from RUGD and group them by the body AgriMorph would pick.

RUGD class -> terrain group -> morphology (the design rule used throughout the poster):
  firm      asphalt, concrete, gravel, grass   -> m1 Wheel      (fast transit on firm ground)
  dense     bush                               -> m2 Snake      (under-canopy navigation)
  rough     rock, rock-bed, log                -> m3 Quadruped  (step over uneven ground)
  soft      dirt, sand, mulch, water           -> m4 Crawler    (low ground pressure)
  canopy    tree                               -> m5 Extended   (reach into canopy)
"""
import os, sys, json, zipfile, io
import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.environ.get("AGRI_ROOT", os.path.expanduser("~/Desktop/Research/IROS2026_data"))
GROUPS = {"firm": [10, 23, 11, 3], "dense": [19], "rough": [21, 14, 15], "soft": [1, 2, 13, 6], "canopy": [4]}
GNAMES = list(GROUPS)
MORPH = {"firm": "Wheel", "dense": "Snake", "rough": "Quadruped", "soft": "Crawler", "canopy": "Extended"}
P, STRIDE, PURITY, MAXPER = 160, 80, 0.85, 2

cmap = {}
for line in open(f"{ROOT}/RUGD_annotations/RUGD_annotation-colormap.txt"):
    i, name, r, g, b = line.split()
    cmap[(int(r), int(g), int(b))] = int(i)
lut = np.full(256 ** 3, -1, np.int16)
for (r, g, b), i in cmap.items():
    gid = next((k for k, g_ in enumerate(GROUPS.values()) if i in g_), -1)
    lut[(r << 16) | (g << 8) | b] = gid


def group_map(ann_path):
    a = np.asarray(Image.open(ann_path).convert("RGB")).astype(np.int32)
    return lut[(a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]]


def main(frame_step=4):
    rng = np.random.default_rng(0)
    out = f"{ROOT}/patches"; os.makedirs(out, exist_ok=True)
    rows = []
    seqs = sorted(d for d in os.listdir(f"{ROOT}/RUGD_annotations") if os.path.isdir(f"{ROOT}/RUGD_annotations/{d}"))
    for seq in seqs:
        files = sorted(f for f in os.listdir(f"{ROOT}/RUGD_annotations/{seq}") if f.endswith(".png"))[::frame_step]
        for fn in files:
            gm = group_map(f"{ROOT}/RUGD_annotations/{seq}/{fn}")
            img = Image.open(f"{ROOT}/RUGD_frames-with-annotations/{seq}/{fn}").convert("RGB")
            H, W = gm.shape
            cands = {}
            for y in range(0, H - P + 1, STRIDE):
                for x in range(0, W - P + 1, STRIDE):
                    win = gm[y:y + P, x:x + P]
                    vals, cnt = np.unique(win, return_counts=True)
                    k = cnt.argmax()
                    if vals[k] >= 0 and cnt[k] / win.size >= PURITY:
                        cands.setdefault(int(vals[k]), []).append((y, x))
            for g, locs in cands.items():
                for j in rng.permutation(len(locs))[:MAXPER]:
                    y, x = locs[j]
                    name = f"{seq}_{fn[:-4]}_{y}_{x}.jpg"
                    img.crop((x, y, x + P, y + P)).save(f"{out}/{name}", quality=92)
                    rows.append(dict(file=name, seq=seq, frame=fn, y=y, x=x, group=GNAMES[g]))
        print(seq, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(f"{ROOT}/patches.csv", index=False)


if __name__ == "__main__":
    main()
