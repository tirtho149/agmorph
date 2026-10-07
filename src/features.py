"""Frozen CLIP ViT-B/32 image embeddings for every patch (CPU)."""
import os
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image

ROOT = os.environ.get("AGRI_ROOT", os.path.expanduser("~/Desktop/Research/IROS2026_data"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    # some nodes' cuDNN can't pick a conv engine ("GET was unable to find an engine");
    # CLIP ViT-B/32 has a single patch-embed conv, so disabling cuDNN is robust & cheap.
    torch.backends.cudnn.enabled = False
torch.set_num_threads(min(int(os.environ.get("OMP_NUM_THREADS", "8")), len(os.sched_getaffinity(0))))
print(f"CLIP device={DEV}, threads={torch.get_num_threads()}", flush=True)
# Use a locally pre-fetched checkpoint if provided (offline compute nodes),
# otherwise fall back to the open_clip pretrained tag (downloads from HF).
_CKPT = os.environ.get("AGRI_CLIP_CKPT", "")
_PRETRAINED = _CKPT if (_CKPT and os.path.exists(_CKPT)) else "laion2b_s34b_b79k"
model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained=_PRETRAINED)
model.eval().to(DEV)
tok = open_clip.get_tokenizer("ViT-B-32")


@torch.no_grad()
def embed_images(paths, bs=128, workers=64):
    # /work is latency-bound on small-file reads; load each batch with a big thread
    # pool so dozens of image opens overlap (turns ~0.6 img/s serial into ~tens/s).
    from concurrent.futures import ThreadPoolExecutor
    _load = lambda p: prep(Image.open(p).convert("RGB"))
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(paths), bs):
            x = torch.stack(list(ex.map(_load, paths[i:i + bs])))
            f = model.encode_image(x.to(DEV)).float().cpu()
            out.append(torch.nn.functional.normalize(f, dim=-1).numpy().astype(np.float32))
            if i % (bs * 4) == 0:
                print(i, len(paths), flush=True)
    return np.concatenate(out)


PROMPTS = {
    "firm": ["a photo of a paved path", "a photo of a gravel road", "a photo of short grass lawn", "a photo of concrete ground"],
    "dense": ["a photo of dense bushes", "a photo of thick shrubs", "a photo of dense undergrowth"],
    "rough": ["a photo of rocks", "a photo of a rocky creek bed", "a photo of fallen logs"],
    "soft": ["a photo of a dirt trail", "a photo of sand", "a photo of mulch", "a photo of muddy water"],
    "canopy": ["a photo of tree trunks", "a photo of tree foliage", "a photo of a forest canopy"],
}


@torch.no_grad()
def embed_text():
    W = []
    for g, ps in PROMPTS.items():
        t = model.encode_text(tok(ps).to(DEV)).float().cpu()
        t = torch.nn.functional.normalize(t, dim=-1).mean(0)
        W.append(torch.nn.functional.normalize(t, dim=0).numpy())
    return np.stack(W)


if __name__ == "__main__":
    df = pd.read_csv(f"{ROOT}/patches.csv")
    if os.path.exists(f"{ROOT}/patches_rare.csv"):
        df = pd.concat([df, pd.read_csv(f"{ROOT}/patches_rare.csv")], ignore_index=True)
        df.to_csv(f"{ROOT}/patches_all.csv", index=False)
    X = embed_images([f"{ROOT}/patches/{f}" for f in df.file])
    np.save(f"{ROOT}/clip_feats.npy", X)
    np.save(f"{ROOT}/clip_text.npy", embed_text())
    print("done", X.shape)
