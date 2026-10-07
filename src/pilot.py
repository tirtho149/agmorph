"""Fast pilot: embed a stratified SUBSAMPLE of patches, fit the ensemble, and
report the headline perception numbers (zero-shot vs probe balanced accuracy,
uncertainty AUROC). Validates the pipeline without the full 17.5k-image run."""
import os, json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "src")
ROOT = os.environ["AGRI_ROOT"]

t0 = time.time()
import features as ft
from exp import G, TRAIN, ID_TEST, OOD_TEST, fit_ensemble, ens_predict
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
print(f"[{time.time()-t0:.0f}s] model + imports ready", flush=True)


def split_of(seq):
    return "train" if seq in TRAIN else "id" if seq in ID_TEST else "ood" if seq in OOD_TEST else "other"


df = pd.read_csv(f"{ROOT}/patches.csv")
df["split"] = df.seq.map(split_of)
df = df[df.split.isin(["train", "id", "ood"])]
parts = []
for (_, sp), d in df.groupby(["group", "split"]):
    cap = 400 if sp == "train" else 150
    parts.append(d.sample(min(len(d), cap), random_state=0))
sub = pd.concat(parts).reset_index(drop=True)
print(f"[{time.time()-t0:.0f}s] pilot subsample: {len(sub)} patches "
      f"({dict(sub.split.value_counts())})", flush=True)

X = ft.embed_images([f"{ROOT}/patches/{f}" for f in sub.file])
T = ft.embed_text()
print(f"[{time.time()-t0:.0f}s] embedded {X.shape}", flush=True)

y = sub.group.map({g: i for i, g in enumerate(G)}).values
sp = sub.split.values
ms = fit_ensemble(X[sp == "train"], y[sp == "train"])

res = {"pilot_subsample": {k: int(v) for k, v in sub.split.value_counts().items()}}
for s in ("id", "ood"):
    k = sp == s
    if k.sum() == 0:
        continue
    pm, mi, _, _ = ens_predict(ms, X[k])
    zs = (X[k] @ T.T).argmax(1); pr = pm.argmax(1); cor = (pr == y[k])
    res[s] = dict(
        n=int(k.sum()),
        zeroshot_bacc=round(float(balanced_accuracy_score(y[k], zs)), 3),
        probe_bacc=round(float(balanced_accuracy_score(y[k], pr)), 3),
        probe_f1=round(float(f1_score(y[k], pr, average="macro")), 3),
        probe_acc=round(float(cor.mean()), 3),
        auroc_mi=(round(float(roc_auc_score((~cor).astype(int), mi)), 3) if len(set(cor.tolist())) > 1 else None),
    )
print("PILOT_RESULTS " + json.dumps(res), flush=True)
json.dump(res, open(f"{ROOT}/PILOT_RESULTS.json", "w"), indent=2)
print(f"[{time.time()-t0:.0f}s] wrote PILOT_RESULTS.json", flush=True)
