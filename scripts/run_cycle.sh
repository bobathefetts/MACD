#!/usr/bin/env bash
# Sanity run on the toy QA task with the mock backend.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m macd.main cycle --task-config configs/tasks/qa.yaml --model-config configs/models/local.yaml --cycles 2 "$@"
