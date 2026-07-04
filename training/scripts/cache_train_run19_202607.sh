#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ENV_FILE="${ENV_FILE:-.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.example to .env and adjust paths." >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  :
elif [[ -f "${HOME}/freqenv/bin/activate" ]]; then
  source "${HOME}/freqenv/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}}"

CACHE_LOG0="${RUN19_CACHE_LOG0:-logs/kron_cache_run19_202607_gpu0.log}"
CACHE_LOG1="${RUN19_CACHE_LOG1:-logs/kron_cache_run19_202607_gpu1.log}"
CACHE_LOGM="${RUN19_CACHE_LOG_META:-logs/kron_cache_run19_202607_meta.log}"
TRAIN_LOG="${RUN19_LOG_FILE:-logs/train_run19_seq300_noasset_teacher_202607_u200.log}"

mkdir -p logs "${RUN19_REPR_CACHE_DIR}" "${RUN19_STRENGTH_CACHE_DIR}" "$(dirname "${RUN19_RESOLVED_CONFIG}")" "$(dirname "${TRAIN_LOG}")"

RESOLVED_CONFIG="$("${PYTHON_BIN}" training/scripts/render_run19_finetune_config.py --env-file "${ENV_FILE}")"
echo "$(date +%H:%M:%S) resolved config: ${RESOLVED_CONFIG}"

echo "$(date +%H:%M:%S) cache teacher shard 0/2 -> ${CACHE_LOG0}"
"${PYTHON_BIN}" training/train_e2e_ppo.py --config "${RESOLVED_CONFIG}" --cache-only --cache-device cuda:0 --num-shards 2 --shard-index 0 > "${CACHE_LOG0}" 2>&1 &
pid0=$!

echo "$(date +%H:%M:%S) cache teacher shard 1/2 -> ${CACHE_LOG1}"
"${PYTHON_BIN}" training/train_e2e_ppo.py --config "${RESOLVED_CONFIG}" --cache-only --cache-device cuda:1 --num-shards 2 --shard-index 1 > "${CACHE_LOG1}" 2>&1 &
pid1=$!

set +e
wait "${pid0}"
status0=$?
wait "${pid1}"
status1=$?
set -e

echo "$(date +%H:%M:%S) cache exits: gpu0=${status0} gpu1=${status1}"
if [[ ${status0} -ne 0 || ${status1} -ne 0 ]]; then
  exit 1
fi

"${PYTHON_BIN}" training/train_e2e_ppo.py --config "${RESOLVED_CONFIG}" --cache-meta-only > "${CACHE_LOGM}" 2>&1
echo "$(date +%H:%M:%S) cache meta refreshed -> ${CACHE_LOGM}"

echo "$(date +%H:%M:%S) training start -> ${TRAIN_LOG}"
"${PYTHON_BIN}" training/train_e2e_ppo.py --config "${RESOLVED_CONFIG}" > "${TRAIN_LOG}" 2>&1
echo "$(date +%H:%M:%S) training finished"
