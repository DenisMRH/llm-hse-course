#!/usr/bin/env bash
set -euo pipefail

# Включаем API очистки кеша
export VLLM_SERVER_DEV_MODE=1

config=${CONFIG:-base}
args=()
case ${config%_ngram} in
  base) ;;
  base_v1) export VLLM_USE_V2_MODEL_RUNNER=0 ;;
  prefill512) args+=(--max-num-batched-tokens 512) ;;
  prefill2048) args+=(--max-num-batched-tokens 2048) ;;
  prefill16384) args+=(--max-num-batched-tokens 16384) ;;
  interactive) args+=(--performance-mode interactivity) ;;
  *) echo "Unknown configuration: $config" >&2; exit 2 ;;
esac
if [[ $config == *_ngram ]]; then
  args+=(--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}')
fi

exec vllm serve Qwen/Qwen3-8B "${args[@]}" "$@"
