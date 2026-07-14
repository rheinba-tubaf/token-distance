# Minimal empirical Green profiles for language models
# Companion code to https://arxiv.org/abs/2606.29139

This repository contains a compact reproduction pipeline for the empirical Green-profile measurements used in the paper.

It measures how a selected next-token logit changes under infinitesimal perturbations of earlier input embeddings. For a target position `j` and causal distance `r`, the core quantity is

```text
G(j,r) = || d z[j, selected_token] / d e[j-r] ||_2.
```

The script computes two selected-token variants:

- `gradnorm_diag`: the actually observed next token;
- `pred_gradnorm_diag`: the model-predicted token.

Both are normalized row-wise by their value at `r=0`. The script also saves the corresponding raw gradient norms. The code intentionally omits the logit-standard-deviation variants, top-k diagnostics, and extra helper plots from the internal experiment pipeline.

## What should be reproduced?

For the GitHub repository accompanying the paper, the main reproduction commands should match the two paper examples:

1. `EleutherAI/pythia-14m-deduped`, sequence length 2048;
2. `Qwen/Qwen2.5-0.5B`, sequence length 4096, called `qwen490m` in the output filenames.

The smoke test is separate. It is only a technical check that installation, model loading, autograd, fitting, and plotting work. It is not meant to produce a publication-quality curve and should not be interpreted as an additional experiment.

## Files

```text
minimal_green_profiles.py   # measure profiles, fit decay models, write plots
prepare_green_assets.py     # create local Gutenberg-style JSONL documents
run_paper_examples.sh       # reproduce the two paper examples: 14M and 490M/0.5B
run_smoke_test.sh           # short technical check, not a paper experiment
run_qwen490m_example.sh     # convenience wrapper for the Qwen paper example only
requirements.txt            # Python dependencies
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU runs, install a PyTorch build matching your CUDA version if the default `pip` build is not appropriate on your machine.

## Prepare the local long-document file

The paper runs use a local JSONL file with long Gutenberg-style documents:

```bash
python prepare_green_assets.py \
  --max-docs 1500 \
  --min-doc-chars 2000 \
  --outdir green_assets
```

This writes, for example,

```text
green_assets/gutenberg_longdocs_1500.jsonl
```

Each line is a JSON object with a `text` field. The exact numerical results depend on the actual documents, package versions, model revision, dtype, and hardware. To reproduce bitwise-identical paper figures, keep the same local JSONL file and software environment. The scripts below reproduce the paper protocol and output filenames.

## Reproduce the two paper examples

Run both paper examples:

```bash
./run_paper_examples.sh
```

The main output plots are:

```text
runs/pythia14m/pythia14m_pred_gradnorm_diag_aggregate_fit_loglog.png
runs/pythia14m/pythia14m_pred_gradnorm_diag_individual_loglog.png

runs/qwen490m/qwen490m_pred_gradnorm_diag_aggregate_fit_loglog.png
runs/qwen490m/qwen490m_pred_gradnorm_diag_individual_loglog.png
```

The Qwen aggregate plot above is the requested paper-style file:

```text
qwen490m_pred_gradnorm_diag_aggregate_fit_loglog.png
```

To run only one of the two examples:

```bash
RUN_QWEN490M=0 ./run_paper_examples.sh   # only Pythia-14M
RUN_PYTHIA14M=0 ./run_paper_examples.sh  # only Qwen/Qwen2.5-0.5B
```

## Paper parameters

The paper examples are encoded in `run_paper_examples.sh`.

### Pythia-14M

```bash
python minimal_green_profiles.py \
  --model EleutherAI/pythia-14m-deduped \
  --dataset-source local-jsonl \
  --local-docs green_assets/gutenberg_longdocs_1500.jsonl \
  --seq-len 2048 \
  --n-samples 1024 \
  --target-j-min 1920 \
  --target-j-max 2048 \
  --target-stride 8 \
  --r-values 0,1,2,3,4,5,6,8,10,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1280,1536 \
  --source-min-index 384 \
  --plot-r-min 1 \
  --r-min-fit 1 \
  --outdir runs/pythia14m \
  --plot-prefix pythia14m
```

Here the largest distance is `r=1536` and the smallest target is `j=1920`, so the most distant source position is `j-r=384`. This keeps the source away from the left boundary.

### Qwen/Qwen2.5-0.5B, called qwen490m in filenames

```bash
python minimal_green_profiles.py \
  --model Qwen/Qwen2.5-0.5B \
  --dataset-source local-jsonl \
  --local-docs green_assets/gutenberg_longdocs_1500.jsonl \
  --seq-len 4096 \
  --n-samples 1024 \
  --target-j-min 3584 \
  --target-j-max 4096 \
  --target-stride 32 \
  --r-values 0,1,2,3,4,5,6,8,10,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048 \
  --source-min-index 1536 \
  --plot-r-min 1 \
  --r-min-fit 1 \
  --outdir runs/qwen490m \
  --plot-prefix qwen490m
```

Here the largest distance is `r=2048` and the smallest target is `j=3584`, so the most distant source position is `j-r=1536`. This avoids the left-boundary hook while preserving the measured distance range.

## Smoke test

Run:

```bash
./run_smoke_test.sh
```

This downloads Pythia-14M and WikiText-2 and writes quick diagnostic plots under `runs/smoke`. The smoke test deliberately uses fewer samples and a shorter context than the paper examples. It should only be used to check that the code runs.

For an even faster syntax-only check, reduce `--n-samples`, but the resulting plots should not be used as visual evidence.

## Avoiding the hook artefact

There are two easy ways to get a misleading hook in short diagnostic plots:

1. plotting the artificial normalization point `r=0`, where every diagonal-normalized row is exactly one;
2. allowing the largest distances to sample source positions `i=j-r` too close to the left boundary of the context window.

The script therefore keeps `r=0` in the saved arrays, because it is needed for normalization, but excludes it from log-log plots by default via

```text
--plot-r-min 1
```

It also enforces a left-context buffer via

```text
--source-min-index ...
```

This means that every measured source position satisfies `j-r >= source_min_index`. In the paper examples this does not shorten the original distance list; it only makes the intended boundary margin explicit.

## Outputs

For each run, the script writes:

```text
r_values.npy
metadata.json
row_metadata.csv
summary.csv
fit_results.csv

autograd_green_gradnorm.npy
autograd_green_gradnorm_diag.npy
autograd_green_pred_gradnorm.npy
autograd_green_pred_gradnorm_diag.npy

<plot-prefix>_gradnorm_aggregate_fit_loglog.png
<plot-prefix>_gradnorm_individual_loglog.png
<plot-prefix>_gradnorm_diag_aggregate_fit_loglog.png
<plot-prefix>_gradnorm_diag_individual_loglog.png
<plot-prefix>_pred_gradnorm_aggregate_fit_loglog.png
<plot-prefix>_pred_gradnorm_individual_loglog.png
<plot-prefix>_pred_gradnorm_diag_aggregate_fit_loglog.png
<plot-prefix>_pred_gradnorm_diag_individual_loglog.png
```

The fits are performed in log-space for three simple decay models:

```text
A (r+1)^(-p)
c + A (r+1)^(-p)
c + A exp(-r/xi)
```

The rows in `fit_results.csv` are ranked by log-space RMSE, matching the model-selection criterion used in the paper. The AIC value is retained only as an additional diagnostic. By default, `r=0` is used for normalization but excluded from both decay fits and log-log plots.

The default option `--dtype auto` does not impose a hardware-dependent dtype. It preserves the dtype stored in the selected checkpoint. For the reported checkpoints this gives FP16 parameters for the Pythia models and BF16 parameters for Qwen2.5-0.5B. An explicit override can be requested with `--dtype float32`, `--dtype float16`, or `--dtype bfloat16`.

## Notes on interpretation

The measured object is not a Green's function of language itself. It is a local Jacobian response of a trained nonlinear sequence operator, evaluated at sampled token windows. The row-wise diagonal normalization is important because the absolute derivative scale depends on the model, embeddings, and logit scale.
# token-distance
