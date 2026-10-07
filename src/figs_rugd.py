"""Poster figures from the RUGD experiments (600 dpi PNG + vector PDF)."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/Desktop/Research/IROS2026_poster_figures/src"))
sys.path.insert(0, os.path.dirname(__file__))
import poster_figs as pf
import build_patches as bp
from exp import G, MORPH, U

ROOT = bp.ROOT
OUT = pf.OUT
CARD, GOLD, INK, MUTED, LINE, SOFT, BLUE = pf.CARD, pf.GOLD, pf.INK, pf.MUTED, pf.LINE, pf.SOFT, pf.BLUE
GCOL = {"firm": "#C9C4A6", "dense": "#7FB069", "rough": "#6B6B6B", "soft": "#8A5A2B", "canopy": "#1F5F3A"}
GLAB = {"firm": "firm ground", "dense": "dense brush", "rough": "rock / log", "soft": "soft ground", "canopy": "tree canopy"}
GSRC = {"firm": "asphalt · concrete · gravel · grass", "dense": "bush", "rough": "rock · rock-bed · log",
        "soft": "dirt · sand · mulch · water", "canopy": "tree"}
PCOL = {"Fixed (wheel)": MUTED, "Reactive": BLUE, "Predictive (no gate)": GOLD, "AgriMorph": CARD, "Oracle": "#9C9C9C"}


def frame(seq, fn):
    return Image.open(f"{ROOT}/RUGD_frames-with-annotations/{seq}/{fn}").convert("RGB")


def overlay(seq, fn):
    gm = bp.group_map(f"{ROOT}/RUGD_annotations/{seq}/{fn}")
    img = np.asarray(frame(seq, fn)).astype(float) / 255
    col = np.zeros_like(img)
    for i, g in enumerate(G):
        col[gm == i] = matplotlib.colors.to_rgb(GCOL[g])
    a = np.where(gm[..., None] >= 0, 0.6, 0.0)
    return img * (1 - a) + col * a


def fig_raw_frames():
    picks = [("park-1", "park-1_00801.png"), ("trail-7", None), ("creek", "creek_01761.png"), ("creek", "creek_01821.png")]
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.6), gridspec_kw={"hspace": 0.06, "wspace": 0.04})
    for j, (seq, i) in enumerate(picks):
        fs = sorted(f for f in os.listdir(f"{ROOT}/RUGD_annotations/{seq}") if f.endswith(".png"))
        fn = i if i in fs else fs[120]
        axs[0, j].imshow(frame(seq, fn)); axs[1, j].imshow(overlay(seq, fn))
        axs[1, j].add_patch(Rectangle((264, 300), 160, 160, fill=False, ec="white", lw=2.5, ls="--"))
        axs[0, j].set_title(seq.replace("-", " "), fontsize=18)
        for a in axs[:, j]:
            a.axis("off")
    axs[0, 0].text(-20, 275, "RGB", rotation=90, ha="right", va="center", fontsize=18)
    axs[1, 0].text(-20, 275, "terrain → body", rotation=90, ha="right", va="center", fontsize=18)
    hs = [Rectangle((0, 0), 1, 1, fc=GCOL[g]) for g in G]
    fig.legend(hs, [f"{GLAB[g]} → {m}" for g, m in zip(G, MORPH)], ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=15)
    fig.text(0.5, -0.08, "RUGD off-road robot frames (Wigness et al., IROS 2019); dashed box = ground-ahead crop the robot decides from",
             ha="center", fontsize=13.5, color=MUTED, style="italic")
    pf.save(fig, "r0_raw_frames_terrain_to_body")


def fig_morph_samples(n=4):
    df = pd.read_csv(f"{ROOT}/patches.csv"); rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(n + 1, 5, height_ratios=[1.25] + [1] * n, hspace=0.08, wspace=0.08)
    for j, (g, glyph) in enumerate(zip(G, pf.GLYPHS)):
        ax = fig.add_subplot(gs[0, j]); ax.set_xlim(-1, 1); ax.set_ylim(-.9, 1.1); ax.axis("off")
        glyph(ax, 0, 0.45, s=.55)
        ax.text(0, -0.25, f"{MORPH[j]} $m_{j+1}$", ha="center", fontsize=19, fontweight="bold", color=CARD)
        ax.text(0, -0.7, GSRC[g], ha="center", fontsize=12.5, color=MUTED, style="italic")
        sub = df[df.group == g]
        sub = sub.groupby("seq", group_keys=False).apply(lambda d: d.sample(min(len(d), 2), random_state=1))
        pick = sub.sample(min(n, len(sub)), random_state=5)
        for i, f in enumerate(pick.file):
            a = fig.add_subplot(gs[i + 1, j]); a.imshow(Image.open(f"{ROOT}/patches/{f}")); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_edgecolor(GCOL[g]); sp.set_linewidth(4)
    pf.save(fig, "r1_morphology_samples")


def fig_quant():
    R = json.load(open(f"{ROOT}/results_perception.json"))
    D = pd.read_csv(f"{ROOT}/results_decision.csv")
    O = np.load(f"{ROOT}/perception_out.npz")
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"wspace": 0.36, "width_ratios": [0.9, 1, 1.25]})
    # (a) recognition
    ax = axs[0]; w = .36
    for k, (s, lab) in enumerate((("id", "seen type"), ("ood", "unseen (creek)"))):
        vals = [R[s]["zeroshot_bacc"], R[s]["probe_bacc"]]
        b = ax.bar(np.arange(2) + (k - .5) * w, vals, w, color=[LINE, CARD][k] if False else (MUTED if k == 0 else CARD),
                   edgecolor=INK, lw=1, label=lab)
        for x, v in zip(np.arange(2) + (k - .5) * w, vals):
            ax.text(x, v + .015, f"{v:.2f}", ha="center", fontsize=14)
    ax.axhline(0.2, color=INK, ls=":", lw=1.2); ax.text(1.55, 0.21, "chance", fontsize=12, ha="right")
    ax.set_xticks([0, 1], ["CLIP\nzero-shot", "ensemble\nprobe"], fontsize=15)
    ax.set_ylim(0, 1.05); ax.set_ylabel("balanced accuracy (5 terrain groups)", fontsize=15)
    ax.set_title("(a) Terrain → body recognition", fontsize=18, fontweight="bold", pad=10)
    ax.legend(fontsize=13, frameon=False, loc="upper left")
    # (b) error vs epistemic uncertainty
    ax = axs[1]
    for s, c, lab in (("id", MUTED, "seen type"), ("ood", CARD, "unseen (creek)")):
        y, pr, mi = O[f"{s}_y"], O[f"{s}_pr"], O[f"{s}_mi"]
        err = (y != pr).astype(float)
        q = np.quantile(mi, np.linspace(0, 1, 6))
        b = np.clip(np.searchsorted(q, mi, side="right") - 1, 0, 4)
        ax.plot(np.arange(5) + 1, [err[b == i].mean() for i in range(5)], marker="o", ms=10, lw=2.8, color=c,
                label=f"{lab}  (AUROC {R[s]['auroc_mi']:.2f})", mec=INK)
    ax.set_xticks(range(1, 6), ["low", "", "", "", "high"], fontsize=14)
    ax.set_xlabel("ensemble disagreement (quintile)", fontsize=15); ax.set_ylabel("error rate", fontsize=15)
    ax.set_title("(b) Disagreement flags mistakes", fontsize=18, fontweight="bold", pad=10)
    ax.legend(fontsize=13, frameon=False, loc="upper left"); ax.set_ylim(0, None)
    # (c) decisions on real sequences
    ax = axs[2]
    pols = ["Fixed (wheel)", "Reactive", "Predictive (no gate)", "AgriMorph", "Oracle"]
    splits = ["seen type", "unseen (creek)"]
    w = .16
    for i, p in enumerate(pols):
        v = [D[(D.split == s) & (D.policy == p)].util.values[0] for s in splits]
        xs = np.arange(2) + (i - 2) * w
        ax.bar(xs, v, w, color=PCOL[p], edgecolor=INK, lw=1, label=p, hatch="//" if p == "Oracle" else None)
    ax.set_xticks([0, 1], splits, fontsize=15)
    ax.set_ylabel("net utility per frame  ↑", fontsize=15)
    ax.set_title("(c) Body choice on real RUGD runs", fontsize=18, fontweight="bold", pad=10)
    ax.legend(fontsize=12, frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for a in axs:
        a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=13)
    fig.text(0.5, -0.06, "Frozen CLIP ViT-B/32 + 5 bootstrap linear heads; perception real, per-body utility table assumed; "
             "trained on park/trail sequences, gate τ set on a held-out trail", ha="center", fontsize=13, color=MUTED, style="italic")
    pf.save(fig, "r2_quantitative_results")


def fig_confusion():
    R = json.load(open(f"{ROOT}/results_perception.json"))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"wspace": 0.45})
    for ax, (s, t) in zip(axs, (("id", "seen type"), ("ood", "unseen (creek)"))):
        M = np.array(R[s]["cm"])
        ax.imshow(M, cmap="Reds", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=13, color="white" if M[i, j] > .55 else INK)
        ax.set_xticks(range(5), MORPH, rotation=30, fontsize=13); ax.set_yticks(range(5), [GLAB[g] for g in G], fontsize=13)
        ax.set_xlabel("body chosen", fontsize=14); ax.set_title(t, fontsize=16, fontweight="bold")
    pf.save(fig, "r4_confusion")


def fig_filmstrip(seq="creek", n=6):
    tr = np.load(f"{ROOT}/traces.npy", allow_pickle=True).item()[seq]
    gt, pred, mi, gate, ag, pr = tr["gt"], tr["pred"], tr["mi"], tr["gate"], tr["AgriMorph"], tr["Predictive (no gate)"]
    # pick a window with a gate event and a switch
    score = np.convolve((gate != "go").astype(float) + (np.diff(ag, prepend=ag[0]) != 0), np.ones(n * 3), "same")
    c = int(np.argmax(score)); idx = np.linspace(max(c - n * 3 // 2, 0), min(c + n * 3 // 2, len(gt) - 1), n).astype(int)
    fig, axs = plt.subplots(1, n, figsize=(18, 5.2), gridspec_kw={"wspace": 0.05})
    for ax, t in zip(axs, idx):
        img = frame(seq, tr["frames"][t]); ax.imshow(img); ax.axis("off")
        col = CARD if gate[t] != "go" else "white"
        ax.add_patch(Rectangle((264, 300), 160, 160, fill=False, ec=col, lw=3))
        ok = pred[t] == gt[t]
        ax.set_title(f"t = {t}", fontsize=14, color=MUTED)
        txt = (f"true: {GLAB[G[gt[t]]]}\npred: {GLAB[G[pred[t]]]} {'✓' if ok else '✗'}\n"
               f"disagree: {mi[t]:.2f}\ngate: {dict(go='commit', look='look again', hold='HOLD')[gate[t]]}\n"
               f"body: {MORPH[ag[t]]}")
        ax.text(0.5, -0.04, txt, transform=ax.transAxes, ha="center", va="top", fontsize=13,
                color=INK, linespacing=1.35, family="DejaVu Sans")
    fig.text(0.5, 1.0, f"AgriMorph on the unseen creek run (RUGD): red box = gate fired on high ensemble disagreement",
             ha="center", fontsize=15, color=MUTED, style="italic")
    pf.save(fig, "r3_qualitative_filmstrip")


def fig_timeline(seq="creek"):
    tr = np.load(f"{ROOT}/traces.npy", allow_pickle=True).item()[seq]
    rows = ["Reactive", "Predictive (no gate)", "AgriMorph", "Oracle"]
    mcol = ["#8C8C8C", "#7FB069", BLUE, "#8A5A2B", GOLD]
    n = len(tr["gt"])
    fig, ax = plt.subplots(figsize=(17, 4.4))
    for t, g in enumerate(tr["gt"]):
        ax.add_patch(Rectangle((t, 4.2), 1, .6, fc=GCOL[G[g]], ec="none"))
    ax.text(-3, 4.5, "terrain", ha="right", va="center", fontsize=15, style="italic")
    for r, p in enumerate(rows):
        y = 3.2 - r * 0.95; seq_ = tr[p]
        for t, m in enumerate(seq_):
            ax.add_patch(Rectangle((t, y), 1, .6, fc=mcol[m], ec="none"))
        bad = np.where(U[tr["gt"], seq_] < 0)[0]
        ax.scatter(bad + .5, np.full(len(bad), y + .3), marker="x", s=22, color=CARD, lw=1.4, zorder=4)
        ax.text(-3, y + .3, p, ha="right", va="center", fontsize=15, fontweight="bold" if p == "AgriMorph" else "normal",
                color=CARD if p == "AgriMorph" else INK)
    hold = np.where(tr["gate"] != "go")[0]
    ax.scatter(hold + .5, np.full(len(hold), 1.3 + .75), marker="|", s=40, color=GOLD, lw=1.2)
    ax.set_xlim(0, n); ax.set_ylim(0.2, 4.9); ax.axis("off")
    hs = [Rectangle((0, 0), 1, 1, fc=c) for c in mcol] + [plt.Line2D([], [], marker="x", color=CARD, ls="", ms=9, mew=2)]
    ax.legend(hs, MORPH + ["body fails terrain"], ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.04), frameon=False, fontsize=14)
    pf.save(fig, "r5_creek_timeline")



TODO_KW = dict(ha="center", va="center", fontsize=16, color=CARD, fontweight="bold")


def todo_axes(ax, title, xlabel, ylabel, note):
    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=15); ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linestyle((0, (5, 4))); sp.set_color(MUTED)
    ax.set_facecolor("#FBF6F6")
    ax.text(0.5, 0.56, "TODO", transform=ax.transAxes, **TODO_KW, alpha=.9)
    ax.text(0.5, 0.40, note, transform=ax.transAxes, ha="center", va="center", fontsize=13, color=MUTED, style="italic")


def fig_quant_todo():
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"wspace": 0.28})
    todo_axes(axs[0], "(a) Terrain → body recognition", "", "balanced accuracy",
              "zero-shot vs ensemble probe\nseen trails vs unseen creek")
    todo_axes(axs[1], "(b) Disagreement flags mistakes", "ensemble disagreement", "error rate",
              "error rate per disagreement\nquintile + AUROC")
    todo_axes(axs[2], "(c) Body choice on real runs", "", "net utility / frame",
              "Fixed · Reactive · Predictive\n(no gate) · AgriMorph · Oracle")
    fig.text(0.5, -0.04, "Placeholder: results pending the GPU run on RUGD", ha="center", fontsize=13, color=MUTED, style="italic")
    pf.save(fig, "r2_quantitative_TODO")


def ground_group(seq, fn):
    gm = bp.group_map(f"{ROOT}/RUGD_annotations/{seq}/{fn}")
    w = gm[300:460, 264:424]; v = w[w >= 0]
    return int(np.bincount(v, minlength=5).argmax()) if v.size > .5 * w.size else None


def fig_film_todo(seq="creek", n=6):
    fs = sorted(f for f in os.listdir(f"{ROOT}/RUGD_annotations/{seq}") if f.endswith(".png"))
    # find a stretch where the ground ahead changes group
    gg = [ground_group(seq, f) for f in fs[::4]]
    best, bi = -1, 0
    for i in range(len(gg) - n):
        w = [g for g in gg[i:i + n] if g is not None]
        k = len(set(w))
        if k > best:
            best, bi = k, i
    pick = [fs[(bi + j) * 4] for j in range(n)]
    fig, axs = plt.subplots(1, n, figsize=(18, 5.3), gridspec_kw={"wspace": 0.05})
    for ax, fn in zip(axs, pick):
        g = ground_group(seq, fn)
        ax.imshow(frame(seq, fn)); ax.axis("off")
        ax.add_patch(Rectangle((264, 300), 160, 160, fill=False, ec="white", lw=3, ls="--"))
        ax.set_title(f"frame {int(fn.split('_')[-1][:-4])}", fontsize=14, color=MUTED)
        gl = GLAB[G[g]] if g is not None else "mixed"
        need = MORPH[g] if g is not None else "—"
        ax.text(0.5, -0.04, f"ground ahead: {gl}\nbest body: {need}", transform=ax.transAxes,
                ha="center", va="top", fontsize=14, color=INK, linespacing=1.35)
        ax.text(0.5, -0.30, "pred · gate · body: TODO", transform=ax.transAxes, ha="center", va="top",
                fontsize=13.5, color=CARD, fontweight="bold")
    fig.text(0.5, axs[0].get_position().y1 + 0.1, "Unseen creek run (RUGD): labels from annotation; AgriMorph decisions to be filled after the GPU run",
             ha="center", fontsize=15, color=MUTED, style="italic")
    pf.save(fig, "r3_qualitative_filmstrip_TODO")


def fig_timeline_todo(seq="creek"):
    fs = sorted(f for f in os.listdir(f"{ROOT}/RUGD_annotations/{seq}") if f.endswith(".png"))[::2]
    gt = [ground_group(seq, f) for f in fs]
    fig, ax = plt.subplots(figsize=(17, 3.6))
    for t, g in enumerate(gt):
        ax.add_patch(Rectangle((t, 3.2), 1, .7, fc=GCOL[G[g]] if g is not None else "white", ec="none"))
    ax.text(-3, 3.55, "terrain ahead", ha="right", va="center", fontsize=15, style="italic")
    for r, p in enumerate(["Reactive", "Predictive (no gate)", "AgriMorph"]):
        y = 2.2 - r * .95
        ax.add_patch(Rectangle((0, y), len(gt), .6, fc="#FBF6F6", ec=MUTED, ls=(0, (5, 4)), lw=1.2))
        ax.text(len(gt) / 2, y + .3, "TODO", ha="center", va="center", fontsize=13, color=CARD, fontweight="bold")
        ax.text(-3, y + .3, p, ha="right", va="center", fontsize=15, fontweight="bold" if p == "AgriMorph" else "normal",
                color=CARD if p == "AgriMorph" else INK)
    ax.set_xlim(0, len(gt)); ax.set_ylim(0.2, 4.0); ax.axis("off")
    hs = [Rectangle((0, 0), 1, 1, fc=GCOL[g]) for g in G]
    ax.legend(hs, [GLAB[g] for g in G], ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.06), frameon=False, fontsize=14)
    pf.save(fig, "r5_creek_timeline_TODO")


def fig_data_stats():
    df = pd.read_csv(f"{ROOT}/patches.csv")
    try:
        df = pd.concat([df, pd.read_csv(f"{ROOT}/patches_rare.csv")])
    except FileNotFoundError:
        pass
    from exp import TRAIN, CAL, ID_TEST, OOD_TEST
    sp = np.select([df.seq.isin(TRAIN), df.seq.isin(CAL), df.seq.isin(ID_TEST), df.seq.isin(OOD_TEST)],
                   ["train", "gate calibration", "test: seen type", "test: unseen creek"], "unused")
    tab = pd.crosstab(sp, df.group).reindex(columns=G).reindex(["train", "gate calibration", "test: seen type", "test: unseen creek"])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    left = np.zeros(len(tab))
    for g in G:
        ax.barh(tab.index[::-1], tab[g].values[::-1], left=left[::-1], color=GCOL[g], edgecolor="white", lw=1, label=f"{GLAB[g]} → {MORPH[G.index(g)]}")
        left = left + tab[g].values
    for y, v in enumerate(left[::-1]):
        ax.text(v + 60, y, f"{int(v):,}", va="center", fontsize=14)
    ax.set_xlabel("single-terrain image patches (160 px, ≥85% one group)", fontsize=15)
    ax.tick_params(labelsize=14); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, left.max() * 1.15)
    ax.legend(fontsize=12.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.22))
    ax.set_title("RUGD patch split — train on parks/trails, test on held-out trails and an unseen creek", fontsize=15, pad=10)
    pf.save(fig, "r6_dataset_split")


if __name__ == "__main__":
    for f in (fig_raw_frames, fig_morph_samples, fig_data_stats, fig_quant_todo, fig_film_todo, fig_timeline_todo):
        f(); print("ok", f.__name__, flush=True)
