
#!/usr/bin/env bash
set -euo pipefail
python -m macd.main cycle --task-config configs/tasks/qa.yaml --model-config configs/models/local.yaml --cycles 2
