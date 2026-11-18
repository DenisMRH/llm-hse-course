#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

url="${URL:-http://localhost:8000}"
data="data/requests.jsonl"
output="${1:-results/$(date +%Y%m%d-%H%M%S)}"
rpm="${RPM:-120}"
rps="$(awk -v rpm="$rpm" 'BEGIN { printf "%.9f", rpm / 60 }')"
tokenizer=()
if [[ -n ${TOKENIZER_PATH:-} ]]; then tokenizer=(--tokenizer "$TOKENIZER_PATH"); fi
mkdir -p "$(dirname "$output")"
mkdir "$output"

# Очищаем кеш
trap 'curl -fsS -X POST "$url/reset_prefix_cache" >/dev/null' EXIT
curl -fsS -X POST "$url/reset_prefix_cache" >/dev/null

aiperf profile --url "$url" --model Qwen/Qwen3-8B \
  --endpoint-type chat --custom-dataset-type raw_payload \
  --input-file "$data" --dataset-sampling-strategy sequential \
  --warmup-request-count 40 --warmup-concurrency 8 \
  --request-count "${REQUEST_COUNT:-$(wc -l < "$data")}" \
  --request-rate "$rps" --arrival-pattern constant --random-seed 42 "${tokenizer[@]}" \
  --streaming --request-timeout-seconds 120 \
  --export-outputs-json --output-artifact-dir "$output/profile"
