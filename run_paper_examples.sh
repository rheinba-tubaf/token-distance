#!/usr/bin/env bash
set -euo pipefail

# Reproduce the two paper examples with the minimal implementation:
#   1) EleutherAI/pythia-14m-deduped, seq_len=2048
#   2) Qwen/Qwen2.5-0.5B, seq_len=4096
#
# Required input:
#   green_assets/gutenberg_longdocs_1500.jsonl
#
# Create it with:
#   python prepare_green_assets.py --max-docs 1500 --outdir green_assets
#
# Environment switches:
#   RUN_PYTHIA14M=0 ./run_paper_examples.sh   # skip Pythia-14M
#   RUN_QWEN490M=0  ./run_paper_examples.sh   # skip Qwen 0.5B
#   PYTHON=python3 ./run_paper_examples.sh

PYTHON="${PYTHON:-python3}"
DOCS="${DOCS:-green_assets/gutenberg_longdocs_1500.jsonl}"
mkdir -p runs logs

if [[ ! -f "$DOCS" ]]; then
  echo "Missing local documents file: $DOCS" >&2
  echo "Create it first with:" >&2
  echo "  $PYTHON prepare_green_assets.py --max-docs 1500 --outdir green_assets" >&2
  exit 1
fi

if [[ "${RUN_PYTHIA14M:-1}" != "0" ]]; then
  echo "=== Paper example 1/2: Pythia-14M ==="
  $PYTHON minimal_green_profiles.py \
    --device auto \
    --model EleutherAI/pythia-14m-deduped \
    --dataset-source local-jsonl \
    --local-docs "$DOCS" \
    --seq-len 2048 \
    --n-samples 1024 \
    --target-j-min 1920 \
    --target-j-max 2048 \
    --target-stride 8 \
    --r-values 0,1,2,3,4,5,6,8,10,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1280,1536 \
    --source-min-index 384 \
    --plot-r-min 1 \
    --r-min-fit 1 \
    --max-docs 1500 \
    --outdir runs/pythia14m \
    --plot-prefix pythia14m \
    2>&1 | tee logs/pythia14m_paper.log
fi

if [[ "${RUN_QWEN490M:-1}" != "0" ]]; then
  echo "=== Paper example 2/2: Qwen2.5-0.5B / 490M ==="
  $PYTHON minimal_green_profiles.py \
    --device auto \
    --model Qwen/Qwen2.5-0.5B \
    --dataset-source local-jsonl \
    --local-docs "$DOCS" \
    --seq-len 4096 \
    --n-samples 1024 \
    --target-j-min 3584 \
    --target-j-max 4096 \
    --target-stride 32 \
    --r-values 0,1,2,3,4,5,6,8,10,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048 \
    --source-min-index 1536 \
    --plot-r-min 1 \
    --r-min-fit 1 \
    --max-docs 1500 \
    --outdir runs/qwen490m \
    --plot-prefix qwen490m \
    2>&1 | tee logs/qwen490m_paper.log
fi

echo
echo "Done. Main paper plots:"
echo "  runs/pythia14m/pythia14m_pred_gradnorm_diag_aggregate_fit_loglog.png"
echo "  runs/pythia14m/pythia14m_pred_gradnorm_diag_individual_loglog.png"
echo "  runs/qwen490m/qwen490m_pred_gradnorm_diag_aggregate_fit_loglog.png"
echo "  runs/qwen490m/qwen490m_pred_gradnorm_diag_individual_loglog.png"
