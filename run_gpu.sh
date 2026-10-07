#!/usr/bin/env bash
# AgriMorph RUGD experiments on a GPU box.  Usage:  bash run_gpu.sh
set -euo pipefail
cd "$(dirname "$0")"
export AGRI_ROOT="$PWD"
python -m pip install -q open_clip_torch scikit-learn pandas pillow
python - <<'P'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
P
python src/features.py            # CLIP ViT-B/32 embeddings for all patches
python src/seq_feats.py --embed   # embeddings for the test-sequence crops (4 views/frame)
python src/exp.py                 # perception metrics + decision replay
mkdir -p results && cp results_*.json results_*.csv perception_out.npz traces.npy clip_text.npy results/
tar czf results.tgz results
echo "DONE -> results.tgz (send this file back)"
