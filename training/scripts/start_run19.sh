#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  :
elif [[ -f "${HOME}/freqenv/bin/activate" ]]; then
  source "${HOME}/freqenv/bin/activate"
fi

mkdir -p logs
nohup python training/train_e2e_ppo.py   --config training/configs/e2e_ppo_run19_sl15_valrandom.json   > logs/train_run19.log 2>&1 &

echo "Run19 training started"
echo "PID: $!"
echo "Log: logs/train_run19.log"
echo "Monitor with: tail -f logs/train_run19.log"
