"""Generate the AgriMorph poster figures that the bundled data supports.

Runnable on CPU after run_gpu.sh has produced the results files.  The full-scene
RGB figures in figs_rugd.py (fig_raw_frames / fig_filmstrip / *_TODO) need the raw
RUGD frames, which are not in the compute bundle; here we build a crop-based
qualitative filmstrip from the already-extracted seqcrops instead.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import poster_figs as pf
import figs_rugd as fr
import poster_extra as pe
from exp import G, MORPH

ROOT = fr.ROOT
GLAB, GCOL = fr.GLAB, fr.GCOL
CARD, INK, MUTED = pf.CARD, pf.INK, pf.MUTED


def fig_crop_filmstrip(seq="creek", n=6):
    """Qualitative strip on the unseen creek run using the ground-ahead crop
    (seqcrops view 0) for each frame, annotated with pred / gate / chosen body."""
    tr = np.load(f"{ROOT}/traces.npy", allow_pickle=True).item()[seq]
    gt, pred, mi, gate, ag = tr["gt"], tr["pred"], tr["mi"], tr["gate"], tr["AgriMorph"]
    frames = tr["frames"]
    # pick a window rich in gate events and body switches
    score = np.convolve((gate != "go").astype(float) + (np.diff(ag, prepend=ag[0]) != 0),
                        np.ones(n * 3), "same")
    c = int(np.argmax(score))
    idx = np.linspace(max(c - n * 3 // 2, 0), min(c + n * 3 // 2, len(gt) - 1), n).astype(int)
    fig, axs = plt.subplots(1, n, figsize=(18, 5.4), gridspec_kw={"wspace": 0.06})
    for ax, t in zip(axs, idx):
        crop = f"{ROOT}/seqcrops/{seq}_{frames[t][:-4]}_v0.jpg"
        ax.imshow(Image.open(crop)); ax.axis("off")
        col = CARD if gate[t] != "go" else "white"
        for spname, sp in ax.spines.items():
            sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(5)
        ax.set_xticks([]); ax.set_yticks([])
        ok = pred[t] == gt[t]
        ax.set_title(f"t = {t}", fontsize=14, color=MUTED)
        txt = (f"true: {GLAB[G[gt[t]]]}\npred: {GLAB[G[pred[t]]]} {'OK' if ok else 'X'}\n"
               f"disagree: {mi[t]:.2f}\ngate: {dict(go='commit', look='look again', hold='HOLD')[gate[t]]}\n"
               f"body: {MORPH[ag[t]]}")
        ax.text(0.5, -0.06, txt, transform=ax.transAxes, ha="center", va="top", fontsize=13,
                color=INK, linespacing=1.35)
    fig.text(0.5, 1.02, "AgriMorph on the unseen creek run (RUGD) — ground-ahead crop; "
             "red border = epistemic gate fired on high ensemble disagreement",
             ha="center", fontsize=15, color=MUTED, style="italic")
    pf.save(fig, "r3_qualitative_filmstrip_crop")


def _have_frames(*seqs):
    return all(os.path.isdir(f"{ROOT}/RUGD_frames-with-annotations/{s}") for s in seqs)


FIGS = [
    ("methodology flow", pe.fig_methodology),
    ("results table", pe.fig_results_table),
    ("morphology samples", fr.fig_morph_samples),
    ("quantitative results", fr.fig_quant),
    ("confusion matrices", fr.fig_confusion),
    ("creek timeline", fr.fig_timeline),
    ("dataset split", fr.fig_data_stats),
    ("crop filmstrip", fig_crop_filmstrip),
]

# Full-scene RGB figures — only if the raw RUGD frames were staged.
if _have_frames("park-1", "trail-7", "creek"):
    FIGS.append(("raw frames (RGB+overlay)", fr.fig_raw_frames))
if _have_frames("creek"):
    FIGS.append(("full-frame filmstrip", fr.fig_filmstrip))

if __name__ == "__main__":
    ok, fail = [], []
    for name, f in FIGS:
        try:
            f(); ok.append(name)
        except Exception as e:
            import traceback; traceback.print_exc()
            fail.append((name, repr(e)))
    print("\n=== FIGURE SUMMARY ===")
    print("OK  :", ", ".join(ok))
    if fail:
        print("FAIL:", "; ".join(f"{n} ({e})" for n, e in fail))
    print("figures dir:", pf.OUT)
    if fail:
        raise SystemExit(1)
