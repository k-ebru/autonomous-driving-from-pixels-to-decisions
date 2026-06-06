#!/usr/bin/env bash
# Run the three sensitivity configurations sequentially.
# Each run is capped to one hour by training.time_limit_secs in its config.
set -euo pipefail

for cfg in configs/model_A.yaml configs/model_B.yaml configs/model_C.yaml; do
  echo ""
  echo "=========================================="
  echo "Training with $cfg"
  echo "=========================================="
  python -m src.train --config "$cfg"
done

echo ""
echo "All three runs complete. Outputs under runs/."
