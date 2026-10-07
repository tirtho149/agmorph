"""Render the AgriMorph pilot headline-numbers figure straight from
PILOT_RESULTS.json (no embeddings / no torch / no GPU needed).

Built to Nature figure specifications:
  - double-column width 183 mm (7.2 in), depth to fit;
  - 5-7 pt sans-serif (Arial/Helvetica, DejaVu fallback);
  - line weights >= 0.25 pt; colourblind-safe blue/orange;
  - lowercase bold panel letters (a, b); NO title on the figure
    (description lives in the caption, written to figures/r7_pilot_perception.caption.txt);
  - fonts embedded as editable TrueType in the vector PDF (fonttype 42);
  - 600-dpi PNG + vector PDF.

Panels: (a) frozen-CLIP linear-probe perception, ID vs OOD, across the headline
metrics; (b) epistemic gate AUROC for mis-classification detection (ID vs OOD).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import poster_figs as pf

# ---- palette: colourblind-safe blue / orange (Nature-safe pairing) ----------
INK   = "#222222"
MUTED = "#6E6E6E"
LINE  = "#D9D9D9"
BLUE  = "#3B6EA5"   # in-distribution
ORANGE = "#E08214"  # out-of-distribution

MM = 1.0 / 25.4  # mm -> inch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 0.75,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # editable embedded TrueType
})

ROOT = os.environ.get("AGRI_ROOT", os.getcwd())
OUT = os.environ.get("AGRI_FIGS", os.path.join(ROOT, "figures"))
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(ROOT, "PILOT_RESULTS.json")))
ID, OOD = R["id"], R["ood"]

METRICS = [("zeroshot_bacc", "Zero-shot\nbal. acc"),
           ("probe_bacc",    "Probe\nbal. acc"),
           ("probe_f1",      "Probe\nmacro-F1"),
           ("probe_acc",     "Probe\naccuracy")]
CHANCE = 0.20   # 5 morphology classes -> balanced-accuracy chance

# double-column width, modest depth
fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(183 * MM, 78 * MM),
    gridspec_kw={"width_ratios": [2.55, 1.0]})

def _labels(ax, bars, fs=6):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015,
                f"{b.get_height():.2f}", ha="center", va="bottom",
                fontsize=fs, fontweight="bold", color=INK)

# ---- Panel a: perception metrics, grouped ID vs OOD -------------------------
keys = [k for k, _ in METRICS]
x = np.arange(len(keys)); w = 0.38
bA = axL.bar(x - w/2, [ID[k]  for k in keys], w, color=BLUE,   ec=INK, lw=0.5,
             label="In-distribution")
bB = axL.bar(x + w/2, [OOD[k] for k in keys], w, color=ORANGE, ec=INK, lw=0.5,
             label="Out-of-distribution")
_labels(axL, bA); _labels(axL, bB)

axL.axhline(CHANCE, ls=(0, (4, 3)), lw=0.6, color=MUTED, zorder=1)
axL.text(len(keys)-0.5, CHANCE+0.015, "chance (5 classes)",
         ha="right", va="bottom", fontsize=5.5, color=MUTED, style="italic")
axL.set_xticks(x); axL.set_xticklabels([lab for _, lab in METRICS])
axL.set_ylim(0, 1.22); axL.set_ylabel("Score")
axL.set_yticks(np.arange(0, 1.01, 0.2))
axL.set_yticklabels([f"{t:.1f}" for t in np.arange(0, 1.01, 0.2)])
axL.tick_params(length=2.5)
axL.spines[["top", "right"]].set_visible(False)
axL.legend(frameon=False, loc="upper center", ncol=2,
           columnspacing=1.4, handletextpad=0.5, handlelength=1.2)
axL.grid(axis="y", color=LINE, lw=0.4, zorder=0)
axL.set_axisbelow(True)

# ---- Panel b: epistemic gate AUROC ------------------------------------------
gb = axR.bar([0, 1], [ID["auroc_mi"], OOD["auroc_mi"]], 0.55,
             color=[BLUE, ORANGE], ec=INK, lw=0.5)
_labels(axR, gb, fs=6.5)
axR.axhline(0.5, ls=(0, (4, 3)), lw=0.6, color=MUTED, zorder=1)
axR.text(1.45, 0.515, "chance", ha="right", va="bottom",
         fontsize=5.5, color=MUTED, style="italic")
axR.set_xticks([0, 1]); axR.set_xticklabels(["ID", "OOD"])
axR.set_ylim(0, 1.0); axR.set_yticks(np.arange(0, 1.01, 0.25))
axR.set_ylabel("AUROC (misclf. detection)")
axR.tick_params(length=2.5)
axR.spines[["top", "right"]].set_visible(False)
axR.grid(axis="y", color=LINE, lw=0.4, zorder=0)
axR.set_axisbelow(True)

# ---- lowercase bold panel letters (Nature style) ----------------------------
for ax, lab in ((axL, "a"), (axR, "b")):
    ax.text(-0.02, 1.06, lab, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="right")

fig.tight_layout(w_pad=2.0)
for ext, dpi in (("png", 600), ("pdf", None)):
    fig.savefig(os.path.join(OUT, f"r7_pilot_perception.{ext}"),
                dpi=dpi, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("saved r7_pilot_perception (Nature-spec)", flush=True)

# ---- caption (title/description belong here, not on the figure) -------------
sub = R["pilot_subsample"]
cap = (
    "Figure R7 | AgriMorph pilot perception on RUGD off-road terrain. "
    "A frozen CLIP ViT-B/32 encoder feeds a five-member bootstrap linear-probe "
    "ensemble that predicts terrain-adaptive body morphology (five classes). "
    f"(a) Balanced accuracy, macro-F1 and accuracy on in-distribution runs "
    f"(ID, n={ID['n']} patches) and held-out out-of-distribution runs "
    f"(OOD, n={OOD['n']}); zero-shot CLIP balanced accuracy is shown for reference. "
    f"Dashed line: five-class balanced-accuracy chance (0.20). "
    f"(b) Area under the ROC curve for epistemic mis-classification detection by "
    f"the probe-ensemble disagreement gate; dashed line: chance (0.50). "
    f"Probe threshold and calibration set from a pilot subsample "
    f"(train n={sub['train']}). Zero-shot bal. acc: ID {ID['zeroshot_bacc']:.2f}, "
    f"OOD {OOD['zeroshot_bacc']:.2f}."
)
open(os.path.join(OUT, "r7_pilot_perception.caption.txt"), "w").write(cap + "\n")
print("saved caption", flush=True)
