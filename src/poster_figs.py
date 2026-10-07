"""Shared poster styling for the AgriMorph RUGD figures.

Provides the names figs_rugd.py imports:
  OUT                      output directory for saved figures
  CARD, GOLD, INK, MUTED, LINE, SOFT, BLUE   palette
  GLYPHS                   5 morphology glyphs (Wheel, Snake, Quadruped, Crawler, Extended)
  save(fig, name)          write <name>.png (600 dpi) + <name>.pdf (vector)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle, Wedge

ROOT = os.environ.get("AGRI_ROOT", os.getcwd())
OUT = os.environ.get("AGRI_FIGS", os.path.join(ROOT, "figures"))
os.makedirs(OUT, exist_ok=True)

# ---- palette (warm earthy poster theme) -------------------------------------
INK   = "#2B2A28"   # primary text / outlines
MUTED = "#7C776F"   # captions, secondary
LINE  = "#D7D2C8"   # light rules / bars
SOFT  = "#F4EFE8"   # soft fill
CARD  = "#B0413E"   # brand accent (AgriMorph highlight)
GOLD  = "#D9A441"   # secondary accent
BLUE  = "#3E6D9C"   # tertiary accent

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 1.1,
})


def save(fig, name):
    """Save a figure as high-res PNG + vector PDF into OUT."""
    for ext, dpi in (("png", 600), ("pdf", None)):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=dpi,
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("saved", name, flush=True)


# ---- morphology glyphs ------------------------------------------------------
# Each glyph(ax, x, y, s) draws an icon centred near (x, y) at scale s in axis
# coords.  Kept deliberately simple so they read at small poster sizes.
def _wheel(ax, x, y, s=0.5):
    ax.add_patch(Circle((x, y), s, fill=False, ec=INK, lw=3, zorder=3))
    ax.add_patch(Circle((x, y), s * 0.28, fc=GOLD, ec=INK, lw=2, zorder=4))
    for a in range(0, 360, 45):
        import math
        ax.plot([x, x + s * math.cos(math.radians(a))],
                [y, y + s * math.sin(math.radians(a))], color=INK, lw=1.6, zorder=2)


def _snake(ax, x, y, s=0.5):
    import numpy as np
    t = np.linspace(-1, 1, 120)
    ax.plot(x + t * s * 1.15, y + s * 0.42 * np.sin(t * 3.1),
            color=INK, lw=6, solid_capstyle="round", zorder=3)
    ax.add_patch(Circle((x + s * 1.15, y + s * 0.42 * np.sin(3.1)), s * 0.16,
                        fc=GOLD, ec=INK, lw=1.8, zorder=4))


def _quadruped(ax, x, y, s=0.5):
    ax.add_patch(Rectangle((x - s * 0.75, y + s * 0.05), s * 1.5, s * 0.5,
                           fc=SOFT, ec=INK, lw=2.5, zorder=3))
    for dx in (-0.6, -0.25, 0.25, 0.6):
        ax.plot([x + dx * s, x + dx * s], [y + s * 0.05, y - s * 0.7],
                color=INK, lw=3, solid_capstyle="round", zorder=2)
    ax.add_patch(Circle((x + s * 0.9, y + s * 0.32), s * 0.16, fc=GOLD, ec=INK, lw=1.8, zorder=4))


def _crawler(ax, x, y, s=0.5):
    ax.add_patch(Rectangle((x - s * 0.85, y - s * 0.1), s * 1.7, s * 0.55,
                           fc=SOFT, ec=INK, lw=2.5, zorder=3))
    ax.add_patch(Rectangle((x - s * 0.95, y - s * 0.6), s * 1.9, s * 0.42,
                           fc=INK, ec=INK, lw=1, zorder=2))
    for dx in (-0.7, -0.3, 0.1, 0.5, 0.85):
        ax.add_patch(Circle((x + dx * s, y - s * 0.39), s * 0.1, fc=GOLD, ec=SOFT, lw=1, zorder=4))


def _extended(ax, x, y, s=0.5):
    ax.add_patch(Rectangle((x - s * 0.45, y - s * 0.6), s * 0.9, s * 0.5,
                           fc=SOFT, ec=INK, lw=2.5, zorder=3))
    ax.plot([x, x - s * 0.35], [y - s * 0.1, y + s * 0.35], color=INK, lw=4, solid_capstyle="round", zorder=2)
    ax.plot([x - s * 0.35, x + s * 0.25], [y + s * 0.35, y + s * 0.9], color=INK, lw=4, solid_capstyle="round", zorder=2)
    ax.add_patch(Circle((x + s * 0.25, y + s * 0.9), s * 0.16, fc=GOLD, ec=INK, lw=1.8, zorder=4))


GLYPHS = [_wheel, _snake, _quadruped, _crawler, _extended]
