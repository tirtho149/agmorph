"""SAGE-parity poster figures for AgriMorph: a boxed perception->embodiment
methodology flow and a clean rendered results table.

Styled to match the SAGE crop-disease poster: warm earthy palette, rounded
soft-tint boxes with a coloured section band, a CARD-red table header with
alternating row shading, and the AgriMorph row highlighted.
"""
import os, sys, json
import numpy as np           # pandas imported lazily in fig_results_table so the
                             # torch-free methodology figure can render with just matplotlib+numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

sys.path.insert(0, os.path.dirname(__file__))
import poster_figs as pf

ROOT = os.environ.get("AGRI_ROOT", pf.ROOT)
INK, MUTED, LINE, SOFT, CARD, GOLD, BLUE = pf.INK, pf.MUTED, pf.LINE, pf.SOFT, pf.CARD, pf.GOLD, pf.BLUE
MORPH = ["Wheel", "Snake", "Quadruped", "Crawler", "Extended"]
GLYPHS = pf.GLYPHS
TCOL = ["#C9C4A6", "#7FB069", "#6B6B6B", "#8A5A2B", "#1F5F3A"]   # firm, dense, rough, soft, canopy
TGRP = ["firm", "dense", "rough", "soft", "canopy"]

# soft tints for the flow boxes (perception = cool, gate = gold, embodiment = card)
PERC = "#EAF1EC"; PROBE = "#E7EEF4"; GATEC = "#FBF1DC"; EMB = "#F6E7E6"
DATAT = "#EFEAE0"   # curation-row tint


FIGH = 9.6            # methodology figure height (two stacked rows -> fills the poster panel)
ASPECT = 17.0 / FIGH  # main axes fills the figure (add_axes([0,0,1,1])); x-unit : y-unit


def _box(ax, x, y, w, h, tint, title, sub, badge=None):
    """Rounded soft-tint box; icon occupies the top, title+subtitle the lower half."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc=tint, ec=INK, lw=1.6, mutation_aspect=0.6, zorder=3))
    ax.text(x + w / 2, y + h * 0.33, title, ha="center", va="center", fontsize=15,
            fontweight="bold", color=INK, zorder=8)
    ax.text(x + w / 2, y + h * 0.135, sub, ha="center", va="center", fontsize=10.8,
            color=MUTED, style="italic", zorder=8, linespacing=1.15)
    if badge:
        ax.text(x + w - 0.008, y + h - 0.012, badge, ha="right", va="top", fontsize=10.5,
                color=CARD, fontweight="bold", zorder=8)


def _arrow(ax, x0, y0, x1, y1, label=None, curve=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle="-|>", mutation_scale=18, lw=2.0, color=INK,
                 connectionstyle=f"arc3,rad={curve}", zorder=2))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.03, label, ha="center", va="bottom",
                fontsize=10.5, color=MUTED, style="italic", zorder=4)


def _icon_ax(ax, cx, cy, vs):
    """Square (equal-aspect) inset centred at (cx, cy); vs = visual size in y-fraction."""
    w = vs / ASPECT
    ia = ax.inset_axes([cx - w / 2, cy - vs / 2, w, vs])
    ia.set_xlim(-1, 1); ia.set_ylim(-1, 1); ia.set_aspect("equal"); ia.axis("off")
    return ia


def _ic_camera(ia, path):
    from PIL import Image
    ia.imshow(Image.open(path).convert("RGB"), extent=[-1, 1, -1, 1], aspect="auto", zorder=2)


def _ic_chip(ia):
    ia.add_patch(FancyBboxPatch((-0.66, -0.66), 1.32, 1.32, boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc="#CBDCEC", ec=INK, lw=1.6, zorder=3))
    for f in (-0.33, 0, 0.33):
        ia.plot([-0.66, 0.66], [f, f], color=INK, lw=0.7, zorder=4)
        ia.plot([f, f], [-0.66, 0.66], color=INK, lw=0.7, zorder=4)
    for f in (-0.4, 0, 0.4):     # pins
        ia.plot([-0.88, -0.66], [f, f], color=INK, lw=1.4, zorder=3)
        ia.plot([0.66, 0.88], [f, f], color=INK, lw=1.4, zorder=3)
        ia.plot([f, f], [0.66, 0.88], color=INK, lw=1.4, zorder=3)
        ia.plot([f, f], [-0.88, -0.66], color=INK, lw=1.4, zorder=3)


def _ic_stack(ia):
    for k, d in enumerate((0.26, 0.0, -0.26)):
        ia.add_patch(FancyBboxPatch((-0.5 + d, -0.5 - d), 0.95, 0.95, boxstyle="round,pad=0.01,rounding_size=0.08",
                     fc=["#EAD9A0", "#E3C97A", GOLD][k], ec=INK, lw=1.4, zorder=3 - k))


def _ic_gate(ia):
    ia.add_patch(FancyBboxPatch((-0.32, -0.9), 0.64, 1.8, boxstyle="round,pad=0.02,rounding_size=0.18",
                 fc="#3B3B3B", ec=INK, lw=1.4, zorder=3))
    for cy, c in ((0.5, "#5FB65F"), (0.0, GOLD), (-0.5, "#C0403E")):
        ia.add_patch(Circle((0, cy), 0.2, fc=c, ec="white", lw=1.0, zorder=4))


def _ic_horizon(ia):
    ia.plot([-0.8, -0.8, 0.85], [0.85, -0.7, -0.7], color=INK, lw=1.4, zorder=3)
    xs = np.linspace(-0.75, 0.8, 24); ys = -0.45 + 1.0 / (1 + np.exp(-6 * xs))
    ia.plot(xs, ys, color=CARD, lw=2.6, zorder=4)
    ia.plot([xs[-1]], [ys[-1]], marker="o", ms=6, color=CARD, mec="white", mew=1.2, zorder=5)


def _ic_film(ia):
    ia.add_patch(FancyBboxPatch((-0.92, -0.62), 1.84, 1.24, boxstyle="round,pad=0.01,rounding_size=0.05",
                 fc="#2E2E2E", ec=INK, lw=1.4, zorder=2))
    for x in (-0.5, 0.0, 0.5):
        ia.add_patch(Rectangle((x - 0.22, -0.34), 0.44, 0.68,
                     fc=("#8FB06B" if x != 0.0 else "#B79A6A"), ec="white", lw=0.8, zorder=3))
    for x in np.linspace(-0.78, 0.78, 6):
        ia.add_patch(Rectangle((x - 0.05, 0.46), 0.1, 0.12, fc="#DDDDDD", ec="none", zorder=3))
        ia.add_patch(Rectangle((x - 0.05, -0.58), 0.1, 0.12, fc="#DDDDDD", ec="none", zorder=3))


def _ic_swatches(ia):
    for x, c in zip(np.linspace(-0.72, 0.72, 5), TCOL):
        ia.add_patch(FancyBboxPatch((x - 0.13, -0.28), 0.26, 0.56, boxstyle="round,pad=0.004,rounding_size=0.04",
                     fc=c, ec=INK, lw=1.0, zorder=3))


def _sample_crop():
    """Pick a real ground-ahead crop for the input-box thumbnail (fallback: None)."""
    import glob
    for pat in ("seqcrops/creek_*_v0.jpg", "seqcrops/*_v0.jpg", "patches/*.jpg", "patches/*.png"):
        fs = sorted(glob.glob(os.path.join(ROOT, pat)))
        if fs:
            return fs[len(fs) // 2]
    return None


def _band(ax, x, w, yb, label, color):
    ax.add_patch(Rectangle((x, yb), w, 0.036, fc=color, ec="none", zorder=1))
    ax.text(x + w / 2, yb + 0.018, label, va="center", ha="center",
            fontsize=13.5, fontweight="bold", color="white", zorder=2)


def fig_methodology():
    """Two-stage SAGE-style methodology: (A) data curation + perception training,
    (B) uncertainty-gated embodied deployment, plus an illustrative per-frame trace."""
    fig = plt.figure(figsize=(17, FIGH))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    crop = _sample_crop()

    # ================= ROW A — CURATION & TRAINING (built once) =================
    _band(ax, 0.008, 0.984, 0.905, "STAGE A · DATA CURATION & PERCEPTION TRAINING   (built once, offline)", BLUE)
    yA, hA = 0.70, 0.175
    A = [
        (0.010, 0.150, DATAT, "RUGD off-road runs", "park · trail · creek\nframe sequences", "data",  "film"),
        (0.205, 0.160, DATAT, "Ground-ahead crops", "160 px patch, ≥85%\none terrain group",  "curate","camera"),
        (0.410, 0.150, DATAT, "5 terrain groups",   "firm · dense · rough\nsoft · canopy",     "label", "swatch"),
        (0.605, 0.160, PROBE, "CLIP ViT-B/32",      "frozen encoder\n512-d embeddings",        "frozen","chip"),
        (0.810, 0.180, PROBE, "Bootstrap ensemble", "5 linear probes\n+ calibrate gate τ",     "train", "stack"),
    ]
    iyA = yA + hA * 0.71
    for bx, bw, tint, title, sub, badge, ic in A:
        _box(ax, bx, yA, bw, hA, tint, title, sub, badge)
        cx = bx + bw / 2
        if ic == "film":   _ic_film(_icon_ax(ax, cx, iyA, 0.085))
        elif ic == "camera" and crop: _ic_camera(_icon_ax(ax, cx, iyA, 0.085), crop)
        elif ic == "swatch": _ic_swatches(_icon_ax(ax, cx, iyA, 0.085))
        elif ic == "chip":  _ic_chip(_icon_ax(ax, cx, iyA, 0.078))
        elif ic == "stack": _ic_stack(_icon_ax(ax, cx, iyA, 0.078))
    for a, b in [(0.160, 0.205), (0.365, 0.410), (0.560, 0.605), (0.765, 0.810)]:
        _arrow(ax, a, yA + hA / 2, b, yA + hA / 2)

    # ---- transition: frozen perception is reused per frame ----
    ax.add_patch(FancyArrowPatch((0.71, yA - 0.005), (0.30, 0.545),
                 arrowstyle="-|>", mutation_scale=16, lw=1.8, color=INK,
                 connectionstyle="arc3,rad=0.18", ls=(0, (5, 3)), zorder=1))
    ax.text(0.508, 0.60, "frozen CLIP + probes reused at deployment", ha="center",
            fontsize=10.5, color=INK, style="italic")

    # ================= ROW B — EMBODIED DEPLOYMENT (per frame) =================
    _band(ax, 0.008, 0.984, 0.475, "STAGE B · EMBODIED DEPLOYMENT   (per camera frame, uncertainty-gated)", CARD)
    yB, hB = 0.275, 0.175
    B = [
        (0.010, 0.135, PERC,  "Camera crop",    "ground-ahead\n160 px patch",       "input", "camera"),
        (0.175, 0.150, PROBE, "CLIP ViT-B/32",  "frozen encoder\n512-d embedding",  "frozen","chip"),
        (0.355, 0.165, PROBE, "Ensemble probe", "5 bootstrap heads\n(BALD)",        "5×",    "stack"),
        (0.545, 0.140, GATEC, "Epistemic gate", "go / look-again /\nhold  (thr. τ)","Eq. 7", "gate"),
        (0.705, 0.150, GATEC, "Utility horizon","H-step projection\nmargin vs switch","Eq. 9","horizon"),
        (0.875, 0.118, EMB,   "Morph body",     "reconfigure to\nmatched form",     "output","wheel"),
    ]
    iyB = yB + hB * 0.71
    for bx, bw, tint, title, sub, badge, ic in B:
        _box(ax, bx, yB, bw, hB, tint, title, sub, badge)
        cx = bx + bw / 2
        if ic == "camera" and crop: _ic_camera(_icon_ax(ax, cx, iyB, 0.082), crop)
        elif ic == "chip":  _ic_chip(_icon_ax(ax, cx, iyB, 0.075))
        elif ic == "stack": _ic_stack(_icon_ax(ax, cx, iyB, 0.075))
        elif ic == "gate":  _ic_gate(_icon_ax(ax, cx, iyB, 0.085))
        elif ic == "horizon": _ic_horizon(_icon_ax(ax, cx, iyB, 0.075))
        elif ic == "wheel": pf._wheel(_icon_ax(ax, cx, iyB, 0.075), 0, 0, s=0.72)
    for a, b in [(0.145, 0.175), (0.325, 0.355), (0.520, 0.545), (0.685, 0.705), (0.855, 0.875)]:
        _arrow(ax, a, yB + hB / 2, b, yB + hB / 2)
    for cx, lab in ((0.250, "512-d embedding"), (0.4375, "terrain posterior"), (0.615, "+ disagreement")):
        ax.text(cx, yB + hB + 0.016, lab, ha="center", va="bottom", fontsize=9.5, color=MUTED, style="italic")

    # look-again feedback loop (gate -> re-observe crop)
    ax.add_patch(FancyArrowPatch((0.615, yB), (0.077, yB),
                 arrowstyle="-|>", mutation_scale=15, lw=1.7, color=GOLD,
                 connectionstyle="arc3,rad=-0.32", ls=(0, (4, 3)), zorder=1))
    ax.text(0.345, yB - 0.052, "look-again: re-observe on high disagreement", ha="center",
            fontsize=10.5, color=GOLD, style="italic", fontweight="bold")

    # ================= illustrative per-frame trace =================
    ax.text(0.012, 0.175, "Illustrative per-frame trace  (terrain ahead → chosen body; gate fires when ambiguous):",
            ha="left", fontsize=10.5, color=INK, style="italic")
    tr_terr = [0, 0, 3, 3, 2, 2, 4, 4, 1, 1]
    tax = ax.inset_axes([0.055, 0.045, 0.90, 0.115]); tax.set_xlim(0, 10); tax.set_ylim(0, 2.4); tax.axis("off")
    tax.text(-0.15, 0.35, "terrain", ha="right", va="center", fontsize=9, style="italic", color=MUTED)
    tax.text(-0.15, 1.6, "body", ha="right", va="center", fontsize=9, style="italic", color=MUTED)
    for t, g in enumerate(tr_terr):
        tax.add_patch(Rectangle((t + 0.06, 0.05), 0.88, 0.6, fc=TCOL[g], ec="white", lw=1, zorder=2))
        GLYPHS[g](tax, t + 0.5, 1.55, s=0.34)
    for tt in (2, 4, 6, 8):   # gate fires at each terrain transition
        tax.axvline(tt, ymin=0.02, ymax=0.62, color=GOLD, lw=2.0, ls=(0, (2, 2)), zorder=4)
    tax.text(2, -0.18, "gate: look-again at transitions", ha="left", va="top", fontsize=8.5,
             color=GOLD, style="italic", fontweight="bold")

    fig.text(0.5, 0.012, "Perception is real (camera → frozen CLIP → bootstrap ensemble); the per-body utility table is an assumed "
             "design table. Ensemble disagreement (BALD mutual information) drives both the gate and the switch margin.",
             ha="center", fontsize=11.5, color=MUTED, style="italic")
    pf.save(fig, "r0_methodology_flow")


def _cell(ax, x, y, w, h, text, fc, tc=INK, bold=False, fs=14, ha="center"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="white", lw=1.4, zorder=1))
    tx = x + w / 2 if ha == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=2)


def fig_results_table():
    """Decision-level results as a clean SAGE-style table (highlight AgriMorph)."""
    import pandas as pd
    D = pd.read_csv(f"{ROOT}/results_decision.csv")
    R = json.load(open(f"{ROOT}/results_perception.json"))
    pols = ["Fixed (wheel)", "Reactive", "Predictive (no gate)", "AgriMorph", "Oracle"]

    def get(split, pol, col):
        m = D[(D.split == split) & (D.policy == pol)]
        return float(m[col].values[0]) if len(m) else np.nan

    splits = ["seen type", "unseen (creek)"]
    fig, ax = plt.subplots(figsize=(13, 6.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # geometry
    x0, xp = 0.02, 0.24                 # policy column
    cw = (0.98 - xp) / 4                # 4 metric columns (2 splits x {util, fail})
    rows_top = 0.72; rh = 0.108
    header_h = 0.10

    # ---- two-level header ----
    ax.add_patch(Rectangle((x0, rows_top + header_h), xp - x0, header_h, fc=CARD, ec="white", lw=1.4))
    ax.text(x0 + 0.01, rows_top + header_h * 1.5, "Policy", va="center", ha="left",
            fontsize=15, color="white", fontweight="bold")
    for si, s in enumerate(splits):
        gx = xp + si * 2 * cw
        ax.add_patch(Rectangle((gx, rows_top + header_h), 2 * cw, header_h, fc=CARD, ec="white", lw=1.4))
        ax.text(gx + cw, rows_top + header_h * 1.5, s, va="center", ha="center",
                fontsize=14.5, color="white", fontweight="bold")
    subhdr = ["net util ↑", "fail % ↓", "net util ↑", "fail % ↓"]
    for ci, lab in enumerate(subhdr):
        gx = xp + ci * cw
        ax.add_patch(Rectangle((gx, rows_top), cw, header_h, fc="#8E3230", ec="white", lw=1.4))
        ax.text(gx + cw / 2, rows_top + header_h / 2, lab, va="center", ha="center",
                fontsize=12.5, color="white")

    # ---- body rows ----
    for ri, pol in enumerate(pols):
        y = rows_top - (ri + 1) * rh
        hl = pol == "AgriMorph"
        rowfc = "#F3DEDD" if hl else (SOFT if ri % 2 == 0 else "white")
        _cell(ax, x0, y, xp - x0, rh, pol, rowfc, tc=CARD if hl else INK, bold=hl, fs=13.5, ha="left")
        vals = []
        for s in splits:
            vals.append((get(s, pol, "util"), get(s, pol, "failure_rate")))
        for si, (u, f) in enumerate(vals):
            _cell(ax, xp + (2 * si) * cw, y, cw, rh, f"{u:+.2f}", rowfc, bold=hl, fs=13.5)
            _cell(ax, xp + (2 * si + 1) * cw, y, cw, rh, f"{100 * f:.0f}", rowfc, bold=hl, fs=13.5)

    # ---- title + perception caption ----
    ax.text(0.5, 0.955, "Body choice on real RUGD runs — net utility per frame (↑ better) "
            "and terrain-failure rate (↓ safer)", ha="center", fontsize=15.5, fontweight="bold", color=INK)
    cap = (f"Perception (balanced acc over 5 terrain groups):  CLIP zero-shot "
           f"{R['id']['zeroshot_bacc']:.2f} / {R['ood']['zeroshot_bacc']:.2f}   →   "
           f"ensemble probe {R['id']['probe_bacc']:.2f} / {R['ood']['probe_bacc']:.2f}   "
           f"(seen / unseen creek).   Disagreement flags errors: AUROC "
           f"{R['id']['auroc_mi']:.2f} / {R['ood']['auroc_mi']:.2f}.")
    ax.text(0.5, 0.05, cap, ha="center", fontsize=12, color=MUTED, style="italic", wrap=True)
    pf.save(fig, "r2b_results_table")


if __name__ == "__main__":
    fig_methodology()
    fig_results_table()
    print("poster_extra: done", flush=True)
