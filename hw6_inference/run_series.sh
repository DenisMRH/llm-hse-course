#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

gpu="${1:?GPU index}"
config="${2:?serve configuration}"
phase="${3:?tuning, spec, spec_check or load}"
shift 3
[[ $# -gt 0 ]] || { echo "Provide one or more RPM values." >&2; exit 2; }
[[ $phase =~ ^(tuning|spec|spec_check|load)$ ]] || exit 2

image="${IMAGE:-hw6-inference}"
client_image="${CLIENT_IMAGE:-$image}"
scratch="${SCRATCH:-$PWD/.scratch}"
revision=b968826d9c46dd6066d109eabc6255188de91218
port="${PORT:-$((18650 + gpu))}"
url="http://127.0.0.1:$port"
label="${RESULT_CONFIG:-$config}"
name="hw6-$USER-$gpu-$$"
server_dir="${RESULTS_DIR:-results}/$phase/$label"
mkdir -p "$server_dir" "$scratch"/{hf,tmp}
scratch="$(cd "$scratch" && pwd)"
printf '%s\n' '67c48379e5ae2608f7ca08650fd346facdc01f766a3e8eecea6459d7cec00ebc  data/requests.jsonl' | sha256sum -c

docker=(docker)
if ! docker info >/dev/null 2>&1; then docker=(sudo -n docker); fi
affinity=()
if [[ -n ${CPUSET:-} ]]; then affinity=(--cpuset-cpus "$CPUSET"); fi
common=(--rm --log-driver=none --network host --ipc host --tmpfs /tmp:size=2g
  --user "$(id -u):$(id -g)" "${affinity[@]}"
  -e "USER=$USER" -e "LOGNAME=$USER" -e HOME=/scratch -e HF_HOME=/scratch/hf -e TMPDIR=/scratch/tmp
  -v "$scratch:/scratch" -v "$PWD:/homework" -w /homework --entrypoint bash)

server=( "${docker[@]}" run "${common[@]}" --name "$name" --gpus "device=$gpu" -e "CONFIG=$config"
  -e HF_HUB_DISABLE_XET=1
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
  "$image" serve.sh --revision "$revision" --host 127.0.0.1 --port "$port" )

server_log="$server_dir/server-$port.log"
printf '%q ' "${server[@]}" > "$server_dir/serve-$port.command"
printf '\n' >> "$server_dir/serve-$port.command"
"${server[@]}" > "$server_log" 2>&1 &
server_pid=$!
trap '"${docker[@]}" stop --time 10 "$name" >/dev/null 2>&1 || true; wait "$server_pid" || true' EXIT
trap 'exit 130' INT TERM
for ((attempt=0; attempt<180; attempt++)); do
  if curl -fsS "$url/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then tail -n 50 "$server_log"; exit 1; fi
  sleep 5
done
curl -fsS "$url/health" >/dev/null

client=( "${docker[@]}" run "${common[@]}" -e "URL=$url" -e HF_HUB_DISABLE_XET=1 )
if [[ -x $scratch/benchmark/bin/aiperf ]]; then
  client+=(-e PATH=/scratch/benchmark/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
fi
tokenizer="/scratch/hf/hub/models--Qwen--Qwen3-8B/snapshots/$revision"
client+=(-e "TOKENIZER_PATH=$tokenizer")
for rpm; do
  output="$server_dir/$rpm"
  echo "$(date -Is) $phase $label $rpm RPM"
  curl -fsS "$url/metrics" > "$server_dir/metrics-$rpm-before.txt"
  if [[ $phase == load ]]; then
    "${client[@]}" "$client_image" load_test.sh "$rpm" "$output" > "$server_dir/client-$rpm.log" 2>&1
  else
    "${client[@]}" -e "RPM=$rpm" -e "REQUEST_COUNT=${REQUEST_COUNT:-}" "$client_image" benchmark.sh "$output" > "$server_dir/client-$rpm.log" 2>&1
  fi
  curl -fsS "$url/metrics" > "$server_dir/metrics-$rpm-after.txt"
done
