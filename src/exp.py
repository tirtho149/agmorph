"""AgriMorph perception -> embodiment experiments on RUGD.

1. Terrain-group recognition from single camera patches (frozen CLIP ViT-B/32):
   zero-shot text prompts vs a 5-member bootstrap ensemble of linear probes.
2. Does ensemble disagreement flag the mistakes?  (AUROC, risk-coverage)
3. Decision level: replay real RUGD sequences frame by frame and let each policy choose a body.
   Perception is real (camera -> classifier); the per-body utility table is an assumed design table.
"""
import os, json
import numpy as np
import pandas as pd
# sklearn imported lazily inside fit_ensemble()/main() so the figure code can
# `from exp import G, MORPH, U` under a spack env that has no scikit-learn.

ROOT = os.environ.get("AGRI_ROOT", os.path.expanduser("~/Desktop/Research/IROS2026_data"))
G = ["firm", "dense", "rough", "soft", "canopy"]
MORPH = ["Wheel", "Snake", "Quadruped", "Crawler", "Extended"]
TRAIN = ["park-1", "park-2", "trail", "trail-3", "trail-4", "trail-6", "trail-9", "trail-10", "trail-11", "trail-12"]
CAL = ["trail-14"]          # held out from fitting; sets the gate threshold tau
ID_TEST = ["park-8", "trail-5", "trail-7", "trail-13", "trail-15"]
OOD_TEST = ["creek"]
E = 5

# assumed per-frame utility U[terrain group, body]; diagonal = matched body
U = np.full((5, 5), 0.25)
np.fill_diagonal(U, 1.0)
U[G.index("soft"), MORPH.index("Wheel")] = -0.6      # sinks
U[G.index("rough"), MORPH.index("Wheel")] = -0.6     # high-centres on rocks
U[G.index("dense"), MORPH.index("Crawler")] = -0.3   # flattens stems
U[G.index("dense"), MORPH.index("Wheel")] = -0.3
U[G.index("firm"), MORPH.index("Crawler")] = 0.6     # works, just slow
U[G.index("soft"), MORPH.index("Quadruped")] = 0.5
K_SWITCH = 1.5           # reconfiguration cost (utility-frames)
REACT_DELAY = 2          # frames of slip before a reactive robot notices
H = 8                    # projection horizon (frames) for the predictive policies


def fit_ensemble(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    ms = []
    for e in range(E):
        idx = rng.choice(len(X), len(X), replace=True)
        m = LogisticRegression(C=[0.5, 1, 2, 4, 8][e], max_iter=3000, class_weight="balanced")
        m.fit(X[idx], y[idx]); ms.append(m)
    return ms


def ens_predict(ms, X):
    P = np.stack([m.predict_proba(X) for m in ms])          # E x N x C
    pm = P.mean(0)
    ent = lambda p: -(p * np.log(p + 1e-12)).sum(-1)
    mi = ent(pm) - ent(P).mean(0)                           # epistemic part (BALD)
    vote = (P.argmax(-1) == pm.argmax(-1)[None]).mean(0)
    return pm, mi, 1 - vote, P


def risk_coverage(correct, unc):
    o = np.argsort(unc)
    c = correct[o]
    cov = np.arange(1, len(c) + 1) / len(c)
    return cov, np.cumsum(c) / np.arange(1, len(c) + 1)


def main():
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score
    df = pd.read_csv(f"{ROOT}/patches_all.csv")
    X = np.load(f"{ROOT}/clip_feats.npy"); T = np.load(f"{ROOT}/clip_text.npy")
    y = df.group.map({g: i for i, g in enumerate(G)}).values
    sp = np.where(df.seq.isin(TRAIN), "train", np.where(df.seq.isin(ID_TEST), "id", np.where(df.seq.isin(OOD_TEST), "ood", "other")))
    df["split"] = sp
    tr = sp == "train"
    ms = fit_ensemble(X[tr], y[tr])
    res, out = {}, {}
    for s in ("id", "ood"):
        k = sp == s
        pm, mi, dis, _ = ens_predict(ms, X[k])
        zs = (X[k] @ T.T).argmax(1)
        pr = pm.argmax(1); cor = (pr == y[k])
        res[s] = dict(n=int(k.sum()),
                      zeroshot_bacc=balanced_accuracy_score(y[k], zs), zeroshot_f1=f1_score(y[k], zs, average="macro"),
                      probe_bacc=balanced_accuracy_score(y[k], pr), probe_f1=f1_score(y[k], pr, average="macro"),
                      probe_acc=cor.mean(), auroc_mi=roc_auc_score(~cor, mi), mean_mi=float(mi.mean()),
                      per_class_n=np.bincount(y[k], minlength=5).tolist())
        out[s] = dict(y=y[k], pr=pr, zs=zs, mi=mi, pm=pm, idx=np.where(k)[0])
        res[s]["cm"] = confusion_matrix(y[k], pr, labels=range(5), normalize="true").round(3).tolist()
    print(json.dumps({s: {a: b for a, b in r.items() if a != "cm"} for s, r in res.items()}, indent=1))
    pmc, mic, _, _ = ens_predict(ms, X[df.seq.isin(CAL).values])
    res["tau"] = float(np.quantile(mic, 0.85))
    json.dump(res, open(f"{ROOT}/results_perception.json", "w"), indent=1)
    np.savez(f"{ROOT}/perception_out.npz", **{f"{s}_{a}": v for s, d in out.items() for a, v in d.items()})
    return ms


# ------------------------------------------------------------------ decision level
def policies(gt, pm_single, pm_ens, mi, pm_look, mi_look, tau, kappa=1.0):
    """gt: true group per frame. Returns dict policy -> list of chosen bodies."""
    n = len(gt)
    out = {}
    out["Fixed (wheel)"] = np.zeros(n, int)
    # oracle: knows terrain, still pays switching cost -> DP
    V = np.zeros((n + 1, 5)); choice = np.zeros((n, 5), int)
    for t in range(n - 1, -1, -1):
        for m in range(5):
            vals = [U[gt[t], b] - (K_SWITCH if b != m else 0) + V[t + 1, b] for b in range(5)]
            choice[t, m] = int(np.argmax(vals)); V[t, m] = max(vals)
    m, seq = 0, []
    for t in range(n):
        m = choice[t, m]; seq.append(m)
    out["Oracle"] = np.array(seq)
    # reactive: moves to the right body only after REACT_DELAY frames of mismatch
    m, bad, seq = 0, 0, []
    for t in range(n):
        best = int(np.argmax(U[gt[t]]))
        bad = bad + 1 if U[gt[t], m] < U[gt[t], best] else 0
        if bad > REACT_DELAY:
            m, bad = best, 0
        seq.append(m)
    out["Reactive"] = np.array(seq)
    # predictive, single model: horizon-H projected gain must beat the reconfiguration cost (Eq. 6)
    m, seq, win = 0, [], []
    for t in range(n):
        win = (win + [pm_single[t]])[-H:]
        J = np.mean(win, 0) @ U
        c = int(np.argmax(J))
        if H * (J[c] - J[m]) > K_SWITCH:
            m = c
        seq.append(m)
    out["Predictive (no gate)"] = np.array(seq)
    # AgriMorph: ensemble + epistemic gate with look-again (Eq. 7) + spread-scaled margin (Eq. 9)
    m, seq, gate, win, uw = 0, [], [], [], []
    for t in range(n):
        p, u, state = pm_ens[t], mi[t], "go"
        if u > tau:
            p, u, state = pm_look[t], mi_look[t], "look"
            if u > tau:
                seq.append(m); gate.append("hold"); continue
        win, uw = (win + [p])[-H:], (uw + [u])[-H:]
        J = np.mean(win, 0) @ U
        sigma = np.sqrt(max(np.mean(uw), 0.0))                  # ensemble spread over the horizon
        c = int(np.argmax(J))
        if H * (J[c] - J[m]) > K_SWITCH + H * kappa * sigma:
            m = c
        seq.append(m); gate.append(state)
    out["AgriMorph"] = np.array(seq)
    return out, gate


def score(gt, seq):
    u = U[gt, seq].sum() - K_SWITCH * ((np.diff(seq) != 0).sum() + int(seq[0] != 0))
    sw = np.where(np.diff(seq) != 0)[0] + 1
    unsafe = sum(U[gt[t], seq[t]] < U[gt[t], seq[t - 1]] for t in sw)
    fail = (U[gt, seq] < 0).mean()
    return dict(util=u / len(gt), switches=len(sw), unsafe=int(unsafe), failure_rate=float(fail),
                matched=float((seq == np.array([np.argmax(U[g]) for g in gt])).mean()))


def decisions(ms, tau):
    fr = pd.read_csv(f"{ROOT}/seq_frames.csv"); F = np.load(f"{ROOT}/seq_feats.npy")   # N x V x D
    rows, traces = [], {}
    for seq, d in fr.groupby("seq", sort=False):
        k = d.index.values; gt = d["gt"].values
        P0 = np.stack([m.predict_proba(F[k, 0]) for m in ms])                 # E x n x C
        PV = np.stack([np.stack([m.predict_proba(F[k, v]) for v in range(F.shape[1])]).mean(0) for m in ms])
        ent = lambda p: -(p * np.log(p + 1e-12)).sum(-1)
        pm, pl = P0.mean(0), PV.mean(0)
        mi, mil = ent(pm) - ent(P0).mean(0), ent(pl) - ent(PV).mean(0)
        pols, gate = policies(gt, P0[0], pm, mi, pl, mil, tau)
        traces[seq] = dict(gt=gt, frames=d.frame.values, pred=pm.argmax(1), mi=mi, gate=np.array(gate), **{p: v for p, v in pols.items()})
        for p, sq in pols.items():
            rows.append(dict(seq=seq, split="unseen (creek)" if seq in OOD_TEST else "seen type", policy=p, n=len(gt), **score(gt, sq)))
    R = pd.DataFrame(rows)
    agg = R.groupby(["split", "policy"]).apply(lambda g: pd.Series({c: np.average(g[c], weights=g.n) if c in ("util", "failure_rate", "matched") else g[c].sum()
                                                                  for c in ("util", "failure_rate", "matched", "switches", "unsafe")}))
    print(agg.round(3))
    R.to_csv(f"{ROOT}/results_decision_per_seq.csv", index=False); agg.to_csv(f"{ROOT}/results_decision.csv")
    np.save(f"{ROOT}/traces.npy", traces, allow_pickle=True)


if __name__ == "__main__":
    ms = main()
    decisions(ms, json.load(open(f"{ROOT}/results_perception.json"))["tau"])
