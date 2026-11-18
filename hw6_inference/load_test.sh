#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Пример: bash load_test.sh 240 results/load-240
rpm="${1:-120}"
url="${URL:-http://localhost:8000}"
output="${2:-results/load-${rpm}rpm-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$(dirname "$output")"
mkdir "$output"

# Очищаем кеш
trap 'curl -fsS -X POST "$url/reset_prefix_cache" >/dev/null' EXIT
curl -fsS -X POST "$url/reset_prefix_cache" >/dev/null

rps="$(awk -v rpm="$rpm" 'BEGIN { printf "%.9f", rpm / 60 }')"
tokenizer=()
if [[ -n ${TOKENIZER_PATH:-} ]]; then tokenizer=(--tokenizer "$TOKENIZER_PATH"); fi

aiperf profile --url "$url" --model Qwen/Qwen3-8B \
  --endpoint-type chat --custom-dataset-type raw_payload \
  --input-file data/requests.jsonl --dataset-sampling-strategy sequential \
  --warmup-request-count 40 --warmup-concurrency 8 \
  --benchmark-duration 300 --benchmark-grace-period inf \
  --request-rate "$rps" --arrival-pattern constant --random-seed 42 "${tokenizer[@]}" \
  --streaming --request-timeout-seconds 120 \
  --export-outputs-json --output-artifact-dir "$output/profile"
