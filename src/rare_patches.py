"""Extra patches of the rare groups (dense, rough) from the training sequences, every frame."""
import os, sys
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import build_patches as bp
from exp import TRAIN, CAL

RARE = [bp.GNAMES.index("dense"), bp.GNAMES.index("rough")]
rows, rng = [], np.random.default_rng(1)
have = set(pd.read_csv(f"{bp.ROOT}/patches.csv").file)
for seq in TRAIN + CAL:
    for fn in sorted(f for f in os.listdir(f"{bp.ROOT}/RUGD_annotations/{seq}") if f.endswith(".png")):
        gm = bp.group_map(f"{bp.ROOT}/RUGD_annotations/{seq}/{fn}")
        if not np.isin(gm, RARE).any():
            continue
        img = None
        for y in range(0, gm.shape[0] - bp.P + 1, 40):
            for x in range(0, gm.shape[1] - bp.P + 1, 40):
                win = gm[y:y + bp.P, x:x + bp.P]
                for g in RARE:
                    if (win == g).mean() >= bp.PURITY and rng.random() < 0.5:
                        name = f"{seq}_{fn[:-4]}_{y}_{x}.jpg"
                        if name in have:
                            continue
                        img = img or Image.open(f"{bp.ROOT}/RUGD_frames-with-annotations/{seq}/{fn}").convert("RGB")
                        img.crop((x, y, x + bp.P, y + bp.P)).save(f"{bp.ROOT}/patches/{name}", quality=92)
                        rows.append(dict(file=name, seq=seq, frame=fn, y=y, x=x, group=bp.GNAMES[g])); have.add(name)
d = pd.DataFrame(rows); d.to_csv(f"{bp.ROOT}/patches_rare.csv", index=False)
print(d.groupby(["group"]).size())
