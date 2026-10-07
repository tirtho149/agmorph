"""Per-frame 'ground ahead' crops along the held-out RUGD sequences, plus look-again views."""
import os, sys
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import build_patches as bp
from exp import ID_TEST, OOD_TEST

ROOT = bp.ROOT
P = 160
VIEWS = [(300, 264), (300, 184), (300, 344), (360, 264)]   # (y, x): ahead-centre, left, right, nearer


def main():
    rows, crops = [], []
    os.makedirs(f"{ROOT}/seqcrops", exist_ok=True)
    for seq in ID_TEST + OOD_TEST:
        fs = sorted(f for f in os.listdir(f"{ROOT}/RUGD_annotations/{seq}") if f.endswith(".png"))
        for t, fn in enumerate(fs):
            gm = bp.group_map(f"{ROOT}/RUGD_annotations/{seq}/{fn}")
            y0, x0 = VIEWS[0]
            win = gm[y0:y0 + P, x0:x0 + P]; v = win[win >= 0]
            if v.size < 0.5 * win.size:
                continue
            gt = int(np.bincount(v, minlength=5).argmax())
            img = Image.open(f"{ROOT}/RUGD_frames-with-annotations/{seq}/{fn}").convert("RGB")
            for k, (y, x) in enumerate(VIEWS):
                name = f"{seq}_{fn[:-4]}_v{k}.jpg"
                img.crop((x, y, x + P, y + P)).save(f"{ROOT}/seqcrops/{name}", quality=92)
                crops.append(name)
            rows.append(dict(seq=seq, t=t, frame=fn, gt=gt))
        print(seq, len(rows), flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{ROOT}/seq_frames.csv", index=False)
    pd.Series(crops).to_csv(f"{ROOT}/seq_crops.csv", index=False, header=["file"])
    print("cropped", len(df), "frames")


def embed():
    import features as ft
    df = pd.read_csv(f"{ROOT}/seq_frames.csv"); crops = pd.read_csv(f"{ROOT}/seq_crops.csv").file.tolist()
    X = ft.embed_images([f"{ROOT}/seqcrops/{c}" for c in crops])
    np.save(f"{ROOT}/seq_feats.npy", X.reshape(len(df), len(VIEWS), -1))
    print("done", X.shape)


if __name__ == "__main__":
    embed() if "--embed" in sys.argv else main()
