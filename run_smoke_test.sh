#!/usr/bin/env bash
set -euo pipefail

# A technical smoke test. This is deliberately NOT a paper reproduction run.
# It checks installation, model loading, autograd, saving, fitting, and plotting.

PYTHON="${PYTHON:-python3}"
mkdir -p runs logs

$PYTHON minimal_green_profiles.py \
  --model EleutherAI/pythia-14m-deduped \
  --dataset-source wikitext2 \
  --seq-len 1024 \
  --n-samples 32 \
  --target-j-min 768 \
  --target-j-max 896 \
  --target-stride 16 \
  --r-values 0,1,2,3,4,5,6,8,10,12,16,24,32,48,64,96,128,192,256,384,512 \
  --source-min-index 256 \
  --plot-r-min 1 \
  --r-min-fit 1 \
  --outdir runs/smoke \
  --plot-prefix smoke \
  2>&1 | tee logs/smoke.log

echo
echo "Smoke-test plots:"
echo "  runs/smoke/smoke_pred_gradnorm_diag_aggregate_fit_loglog.png"
echo "  runs/smoke/smoke_pred_gradnorm_diag_individual_loglog.png"
