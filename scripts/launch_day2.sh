#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/minimind-v-master
bash scripts/run_grpo_fix.sh
bash scripts/run_official_evals.sh
echo "DAY2_PIPELINE_COMPLETE"
