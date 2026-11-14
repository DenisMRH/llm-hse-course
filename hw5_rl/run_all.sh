#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
GPU_INDEX=${GPU_INDEX:-0}
TOOLACE_REVISION=6bda777c88d21e5a204703c1ee45597a8fa4f734
TOOLACE_SHA256=ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

if [[ ! -f toolace_data.json ]]; then
  curl -fL --retry 3 \
    "https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/${TOOLACE_REVISION}/data.json" \
    -o toolace_data.json
fi
echo "${TOOLACE_SHA256}  toolace_data.json" | sha256sum --check

"${DOCKER[@]}" build -t rlvr-homework .
mkdir -p hf_cache tmp cache
container() {
  "${DOCKER[@]}" run --rm --gpus "device=${GPU_INDEX}" --ipc=host \
    -e TMPDIR=/workspace/tmp -e XDG_CACHE_HOME=/workspace/cache \
    -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    -v "$PWD:/workspace" -v "$PWD/hf_cache:/cache" \
    rlvr-homework "$@"
}

container python3 prepare_data.py
retry() {
  local attempt
  for attempt in 1 2 3 4 5 6; do
    if container "$@"; then
      return 0
    fi
    echo "Attempt ${attempt} failed; retrying in 15 seconds" >&2
    sleep 15
  done
  return 1
}

retry python3 evaluate.py Qwen/Qwen3.5-2B --output results/base
for name in rl1_grpo rl2_early_stop rl3_balanced rl4_balanced_concise; do
  retry python3 train.py --config "configs/${name}.json" --output "runs/${name}"
  retry python3 evaluate.py "runs/${name}/merged" --output "results/${name}"
done
