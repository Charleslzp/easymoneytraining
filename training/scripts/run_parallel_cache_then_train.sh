#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  :
elif [[ -f "${HOME}/freqenv/bin/activate" ]]; then
  source "${HOME}/freqenv/bin/activate"
fi

CONFIG_PATH="${1:-training/configs/e2e_ppo_run19_sl15_valrandom.json}"
mkdir -p logs

LOG0="logs/kron_cache_parallel_gpu0.log"
LOG1="logs/kron_cache_parallel_gpu1.log"
LOGM="logs/kron_cache_parallel_meta.log"
LOGO="logs/kron_cache_parallel_orchestrator.log"
LOGT="logs/train_run19.log"
rm -f "${LOG0}" "${LOG1}" "${LOGM}" "${LOGO}" "${LOGT}"

echo "$(date +%H:%M:%S) orchestrator start config=${CONFIG_PATH}" | tee -a "${LOGO}"

python training/train_e2e_ppo.py --config "${CONFIG_PATH}" --cache-only --cache-device cuda:0 --num-shards 2 --shard-index 0 > "${LOG0}" 2>&1 &
pid0=$!
echo "$(date +%H:%M:%S) gpu0 pid=${pid0}" | tee -a "${LOGO}"

python training/train_e2e_ppo.py --config "${CONFIG_PATH}" --cache-only --cache-device cuda:1 --num-shards 2 --shard-index 1 > "${LOG1}" 2>&1 &
pid1=$!
echo "$(date +%H:%M:%S) gpu1 pid=${pid1}" | tee -a "${LOGO}"

set +e
wait "${pid0}"
status0=$?
wait "${pid1}"
status1=$?
set -e

echo "$(date +%H:%M:%S) builder exit status gpu0=${status0} gpu1=${status1}" | tee -a "${LOGO}"
if [[ ${status0} -ne 0 || ${status1} -ne 0 ]]; then
  exit 1
fi

python training/train_e2e_ppo.py --config "${CONFIG_PATH}" --cache-meta-only > "${LOGM}" 2>&1
echo "$(date +%H:%M:%S) meta refreshed" | tee -a "${LOGO}"

python training/train_e2e_ppo.py --config "${CONFIG_PATH}" > "${LOGT}" 2>&1
echo "$(date +%H:%M:%S) training exited" | tee -a "${LOGO}"
