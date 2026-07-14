#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible convenience wrapper for only the Qwen paper example.
RUN_PYTHIA14M=0 RUN_QWEN490M=1 ./run_paper_examples.sh
