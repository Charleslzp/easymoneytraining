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
RUN19_LOG_FILE="${RUN19_LOG_FILE:-logs/train_run19_finetune_from_best.log}"
mkdir -p "$(dirname "${RUN19_LOG_FILE}")"

RESOLVED_CONFIG="$("${PYTHON_BIN}" training/scripts/render_run19_finetune_config.py --env-file "${ENV_FILE}")"

echo "Run19 finetune config: ${RESOLVED_CONFIG}"
echo "Run19 finetune log: ${RUN19_LOG_FILE}"

if [[ "${RUN19_BACKGROUND:-1}" == "0" ]]; then
  "${PYTHON_BIN}" training/train_e2e_ppo.py --config "${RESOLVED_CONFIG}"
else
  nohup "${PYTHON_BIN}" training/train_e2e_ppo.py --config "${RESOLVED_CONFIG}" > "${RUN19_LOG_FILE}" 2>&1 &
  echo "Run19 finetune started"
  echo "PID: $!"
  echo "Monitor with: tail -f ${RUN19_LOG_FILE}"
fi
