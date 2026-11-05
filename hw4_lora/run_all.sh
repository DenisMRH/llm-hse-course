#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
python train.py prepare
for name in base full lora rank32 alpha64 dropout0 all_linear; do
  python train.py "$name"
done
