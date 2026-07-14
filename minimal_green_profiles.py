#!/usr/bin/env python3
"""
Minimal empirical Green-profile experiment for autoregressive language models.

The script measures two row-wise diagonal-normalized Jacobian profiles:

    gradnorm_diag(j,r)      = || d z[j,true_next] / d e[j-r] ||_2 / value at r=0
    pred_gradnorm_diag(j,r) = || d z[j,predicted] / d e[j-r] ||_2 / value at r=0

It also saves the raw gradient norms.  It deliberately omits the larger internal
pipeline's logit-std variants, top-k diagnostics, and extra plots.

Use the shell scripts for the intended entry points:

    ./run_smoke_test.sh        # technical check only
    ./run_paper_examples.sh    # paper reproduction: Pythia-14M and Qwen/Qwen2.5-0.5B

The Qwen paper command writes, among other files,

    qwen490m_pred_gradnorm_diag_aggregate_fit_loglog.png
    qwen490m_pred_gradnorm_diag_individual_loglog.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_R_VALUES = [
    0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64,
    96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def parse_int_list(text: str) -> List[int]:
    if text.strip().lower() == "auto":
        return DEFAULT_R_VALUES.copy()
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def slugify(text: str) -> str:
    text = text.strip().lower().replace("/", "_")
    text = re.sub(r"[^a-z0-9_+.-]+", "_", text)
    return text.strip("_") or "green"


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def choose_dtype(requested: str, device: str):
    """Return an explicit torch dtype, or None to preserve the checkpoint dtype."""
    del device  # kept in the signature for backward compatibility with older calls
    requested = requested.lower()
    if requested == "auto":
        return None
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown --dtype={requested!r}")


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------


def get_backbone(causal_lm: torch.nn.Module) -> torch.nn.Module:
    """Return the transformer backbone module, avoiding the LM head."""
    for name in ("gpt_neox", "transformer", "model"):
        if hasattr(causal_lm, name):
            return getattr(causal_lm, name)
    if hasattr(causal_lm, "base_model"):
        return causal_lm.base_model
    raise RuntimeError("Could not identify the causal-LM backbone.")


def run_backbone(backbone: torch.nn.Module, embeds: torch.Tensor) -> torch.Tensor:
    """Run the backbone on input embeddings and return last_hidden_state."""
    try:
        out = backbone(inputs_embeds=embeds, use_cache=False)
    except TypeError:
        out = backbone(inputs_embeds=embeds)

    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, tuple):
        return out[0]
    raise RuntimeError("Backbone output has no last_hidden_state.")


def selected_logits(lm_head: torch.nn.Module, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Compute z[j, token_id] without materializing logits for all sequence positions."""
    if isinstance(lm_head, torch.nn.Linear):
        weight = lm_head.weight[token_ids, :]
        out = (hidden * weight).sum(dim=-1)
        if lm_head.bias is not None:
            out = out + lm_head.bias[token_ids]
        return out
    logits = lm_head(hidden)
    return logits.gather(-1, token_ids.view(-1, 1)).squeeze(-1)


# -----------------------------------------------------------------------------
# Text loading and chunking
# -----------------------------------------------------------------------------


def is_wikitext_heading(line: str) -> bool:
    s = line.strip()
    return len(s) >= 5 and s.startswith("=") and s.endswith("=")


def split_wikitext_documents(lines: Sequence[str], min_chars: int) -> List[str]:
    docs: List[str] = []
    cur: List[str] = []
    for line in lines:
        if not isinstance(line, str):
            continue
        if is_wikitext_heading(line) and cur:
            text = "\n".join(cur).strip()
            if len(text) >= min_chars:
                docs.append(text)
            cur = [line]
        elif line.strip():
            cur.append(line)
    if cur:
        text = "\n".join(cur).strip()
        if len(text) >= min_chars:
            docs.append(text)
    return docs


def load_documents(args: argparse.Namespace) -> List[str]:
    source = args.dataset_source.lower()

    if source == "local-jsonl":
        path = Path(args.local_docs)
        if not path.exists():
            raise FileNotFoundError(f"Missing --local-docs file: {path}")
        docs: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if path.suffix.lower() == ".jsonl":
                    obj = json.loads(line)
                    text = str(obj.get(args.text_column, obj.get("text", "")))
                else:
                    text = line
                if len(text) >= args.min_doc_chars:
                    docs.append(text)
                if len(docs) >= args.max_docs:
                    break
        if not docs:
            raise RuntimeError(f"No usable documents found in {path}")
        return docs

    if source in {"wikitext2", "wikitext103"}:
        config = "wikitext-2-raw-v1" if source == "wikitext2" else "wikitext-103-raw-v1"
        ds = load_dataset("Salesforce/wikitext", config, split=args.split)
        docs = split_wikitext_documents(ds["text"], min_chars=args.min_doc_chars)
        if not docs:
            text = "\n\n".join(x for x in ds["text"] if isinstance(x, str) and x.strip())
            docs = [text]
        return docs[: args.max_docs]

    if source == "hf":
        if not args.hf_dataset_name:
            raise ValueError("--dataset-source hf requires --hf-dataset-name")
        if args.hf_dataset_config:
            ds = load_dataset(args.hf_dataset_name, args.hf_dataset_config, split=args.split)
        else:
            ds = load_dataset(args.hf_dataset_name, split=args.split)
        if args.text_column not in ds.column_names:
            raise ValueError(f"Text column {args.text_column!r} not found. Columns: {ds.column_names}")
        return [str(x) for x in ds[args.text_column] if isinstance(x, str) and len(x) >= args.min_doc_chars][: args.max_docs]

    raise ValueError("Use --dataset-source local-jsonl, wikitext2, wikitext103, or hf")


def build_token_chunks(
    tokenizer,
    docs: List[str],
    seq_len: int,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
    """Sample contiguous token chunks of length seq_len+1 without crossing documents."""
    needed = seq_len + 1
    token_docs: List[np.ndarray] = []
    token_meta: List[Dict[str, int]] = []

    order = np.arange(len(docs))
    rng.shuffle(order)
    for doc_id in tqdm(order, desc="tokenizing documents"):
        ids = tokenizer(docs[int(doc_id)], add_special_tokens=False).input_ids
        if len(ids) >= needed:
            token_docs.append(np.asarray(ids, dtype=np.int64))
            token_meta.append({"doc_id": int(doc_id), "doc_tokens": int(len(ids))})
        if len(token_docs) >= max(8, min(n_samples, len(docs))):
            break

    if not token_docs:
        raise RuntimeError(f"No document has at least {needed} tokens. Reduce --seq-len or use longer docs.")

    chunks: List[np.ndarray] = []
    metas: List[Dict[str, int]] = []
    for sample in range(n_samples):
        k = int(rng.integers(0, len(token_docs)))
        doc = token_docs[k]
        start = int(rng.integers(0, len(doc) - needed + 1))
        chunks.append(doc[start : start + needed].copy())
        meta = dict(token_meta[k])
        meta.update({"sample": int(sample), "chunk_start": int(start), "chunk_len": int(needed)})
        metas.append(meta)
    return chunks, metas


def build_targets_and_distances(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if args.target_j_min < 0:
        args.target_j_min = 3 * args.seq_len // 4
    if args.target_j_max < 0:
        args.target_j_max = min(args.seq_len, args.target_j_min + args.seq_len // 8)

    targets = np.arange(max(0, args.target_j_min), min(args.seq_len, args.target_j_max), args.target_stride, dtype=int)
    if len(targets) == 0:
        raise ValueError("Empty target range. Check --target-j-min, --target-j-max, and --target-stride.")

    # Keep all source positions j-r away from the left boundary.  Without this
    # guard, the largest distances in short smoke tests can pick up a visible
    # boundary artefact (the "hook" at the tail of the log-log profile).
    max_r_allowed = int(targets.min()) - int(args.source_min_index)
    if max_r_allowed < 0:
        raise ValueError(
            "Empty valid distance range: target_j_min is smaller than --source-min-index. "
            "Move targets to the right, reduce --source-min-index, or use a longer --seq-len."
        )
    r_values = sorted(set(r for r in parse_int_list(args.r_values) if 0 <= r < args.seq_len and r <= max_r_allowed))
    if 0 not in r_values:
        r_values = [0] + r_values
    return targets, np.asarray(r_values, dtype=int)


# -----------------------------------------------------------------------------
# Measurement
# -----------------------------------------------------------------------------


def extract_by_distance(norms: np.ndarray, r_values: np.ndarray, target_j: int) -> np.ndarray:
    values = np.full(len(r_values), np.nan, dtype=np.float64)
    for k, r in enumerate(r_values):
        i = int(target_j - r)
        if 0 <= i < len(norms):
            values[k] = float(norms[i])
    return values


def normalize_by_r0(values: np.ndarray, r_values: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    idx = np.where(r_values == 0)[0]
    if len(idx) == 0:
        return out
    denom = values[:, int(idx[0])]
    good = np.isfinite(denom) & (np.abs(denom) > eps)
    out[good, :] = values[good, :] / denom[good, None]
    return out


def compute_profiles(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    dtype_label = "checkpoint default" if dtype is None else str(dtype)
    print(f"Device: {device}; requested model dtype: {dtype_label}")

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if dtype is not None:
        # ``torch_dtype`` remains compatible with the supported Transformers
        # versions.  It is deliberately omitted for --dtype auto so that the
        # checkpoint's native dtype is preserved.
        load_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(device)
    parameter_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    print(f"Loaded parameter dtype(s): {parameter_dtypes}")
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos is not None and args.seq_len > int(max_pos):
        raise ValueError(f"--seq-len={args.seq_len} exceeds model max_position_embeddings={max_pos}")

    backbone = get_backbone(model)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model has no LM head / output embeddings.")

    targets, r_values = build_targets_and_distances(args)
    print(f"Target positions: {targets[0]}..{targets[-1]} ({len(targets)} positions)")
    print(f"Left source-position buffer: i=j-r >= {args.source_min_index}")
    print(f"Distances r ({len(r_values)}): {list(map(int, r_values))}")

    print("Loading documents and sampling chunks...")
    docs = load_documents(args)
    chunks, chunk_meta = build_token_chunks(tokenizer, docs, args.seq_len, args.n_samples, rng)
    print(f"Built {len(chunks)} token chunks of length {args.seq_len}+1")

    grad_rows: List[np.ndarray] = []
    pred_grad_rows: List[np.ndarray] = []
    row_meta: List[Dict[str, object]] = []

    for sample_idx, ids_np in enumerate(tqdm(chunks, desc="chunks")):
        ids = torch.tensor(ids_np, dtype=torch.long, device=device)
        input_ids = ids[: args.seq_len].unsqueeze(0)
        next_ids_all = ids[1 : args.seq_len + 1]

        base_embeds = model.get_input_embeddings()(input_ids).detach()
        embeds = base_embeds.clone().requires_grad_(True)
        hidden = run_backbone(backbone, embeds)              # [1,L,d]

        target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        hidden_targets = hidden[0, target_tensor, :]         # [T,d]
        true_ids = next_ids_all[target_tensor]               # [T]
        true_logits = selected_logits(lm_head, hidden_targets, true_ids)

        with torch.no_grad():
            target_logits_full = lm_head(hidden_targets.detach()).float()
            pred_ids = target_logits_full.argmax(dim=-1)

        true_ids_cpu = true_ids.detach().cpu().numpy()
        pred_ids_cpu = pred_ids.detach().cpu().numpy()
        pred_logits = selected_logits(lm_head, hidden_targets, pred_ids)

        for t_idx, j in enumerate(tqdm(targets, desc="target gradients", leave=False)):
            same_token = bool(int(true_ids_cpu[t_idx]) == int(pred_ids_cpu[t_idx]))

            # If the last target needs a second backward pass for the predicted token,
            # keep the graph after the true-token gradient. Otherwise it can be freed.
            true_retain = (t_idx < len(targets) - 1) or (not same_token)
            true_grad = torch.autograd.grad(true_logits[t_idx], embeds, retain_graph=true_retain)[0]
            true_norms = true_grad.detach().norm(dim=-1).squeeze(0).float().cpu().numpy()
            grad_rows.append(extract_by_distance(true_norms, r_values, int(j)))

            if same_token:
                pred_grad_rows.append(grad_rows[-1].copy())
            else:
                pred_retain = t_idx < len(targets) - 1
                pred_grad = torch.autograd.grad(pred_logits[t_idx], embeds, retain_graph=pred_retain)[0]
                pred_norms = pred_grad.detach().norm(dim=-1).squeeze(0).float().cpu().numpy()
                pred_grad_rows.append(extract_by_distance(pred_norms, r_values, int(j)))

            meta = dict(chunk_meta[sample_idx])
            meta.update({
                "row": len(grad_rows) - 1,
                "sample": sample_idx,
                "target_j": int(j),
                "true_token_id": int(true_ids_cpu[t_idx]),
                "pred_token_id": int(pred_ids_cpu[t_idx]),
                "top1_correct": int(same_token),
            })
            row_meta.append(meta)

        del embeds, hidden, hidden_targets, true_logits, pred_logits
        if device == "cuda":
            torch.cuda.empty_cache()

    gradnorm = np.asarray(grad_rows, dtype=np.float64)
    pred_gradnorm = np.asarray(pred_grad_rows, dtype=np.float64)
    arrays = {
        "r_values": r_values,
        "gradnorm": gradnorm,
        "gradnorm_diag": normalize_by_r0(gradnorm, r_values),
        "pred_gradnorm": pred_gradnorm,
        "pred_gradnorm_diag": normalize_by_r0(pred_gradnorm, r_values),
    }
    arrays["row_metadata"] = np.asarray(row_meta, dtype=object)
    return arrays


# -----------------------------------------------------------------------------
# Statistics, fitting, and plotting
# -----------------------------------------------------------------------------


def trim_mean_1d(x: np.ndarray, trim: float = 0.10) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan
    x = np.sort(x)
    k = int(math.floor(trim * x.size))
    if 2 * k >= x.size:
        return float(np.mean(x))
    return float(np.mean(x[k : x.size - k]))


def profile_stats(values: np.ndarray) -> Dict[str, np.ndarray]:
    keys = ["mean", "median", "trim10", "q10", "q25", "q75", "q90", "count"]
    out: Dict[str, np.ndarray] = {k: np.full(values.shape[1], np.nan, dtype=np.float64) for k in keys}
    out["count"] = np.zeros(values.shape[1], dtype=np.int64)
    for k in range(values.shape[1]):
        x = values[:, k]
        x = x[np.isfinite(x) & (x > 0)]
        out["count"][k] = x.size
        if x.size == 0:
            continue
        out["mean"][k] = np.mean(x)
        out["median"][k] = np.median(x)
        out["trim10"][k] = trim_mean_1d(x, 0.10)
        out["q10"][k] = np.quantile(x, 0.10)
        out["q25"][k] = np.quantile(x, 0.25)
        out["q75"][k] = np.quantile(x, 0.75)
        out["q90"][k] = np.quantile(x, 0.90)
    return out


def model_pure_power(r: np.ndarray, logA: float, p: float) -> np.ndarray:
    return logA - p * np.log(r + 1.0)


def model_power_floor(r: np.ndarray, logA: float, p: float, logc: float) -> np.ndarray:
    return np.logaddexp(logc, logA - p * np.log(r + 1.0))


def model_exp_floor(r: np.ndarray, logA: float, logxi: float, logc: float) -> np.ndarray:
    return np.logaddexp(logc, logA - r / np.exp(logxi))


def fit_models(r_values: np.ndarray, y_values: np.ndarray, r_min: float, r_max: float | None) -> List[Dict[str, object]]:
    mask = np.isfinite(y_values) & (y_values > 0) & (r_values >= r_min)
    if r_max is not None:
        mask &= r_values <= r_max
    r = r_values[mask].astype(np.float64)
    y = y_values[mask].astype(np.float64)
    if len(r) < 4:
        return []

    ylog = np.log(y)
    ymin = max(float(np.min(y)), 1e-300)
    ymax = max(float(np.max(y)), ymin * 1.01)
    c0 = max(0.5 * ymin, 1e-300)
    a0 = max(ymax - c0, 1e-300)
    xi0 = max(float(np.median(r)), 1.0)

    specs = [
        ("pure_power", model_pure_power, [math.log(ymax), 1.0], ([-np.inf, 0.0], [np.inf, 10.0])),
        ("power_floor", model_power_floor, [math.log(a0), 1.0, math.log(c0)], ([-np.inf, 0.0, -np.inf], [np.inf, 10.0, np.inf])),
        ("exp_floor", model_exp_floor, [math.log(a0), math.log(xi0), math.log(c0)], ([-np.inf, math.log(1e-12), -np.inf], [np.inf, np.inf, np.inf])),
    ]

    results: List[Dict[str, object]] = []
    for name, fun, p0, bounds in specs:
        try:
            popt, _ = curve_fit(fun, r, ylog, p0=p0, bounds=bounds, maxfev=200000)
            pred_log = fun(r, *popt)
            resid = ylog - pred_log
            rss = float(np.sum(resid ** 2))
            n = int(len(r))
            k = len(popt)
            aic = float(n * np.log(rss / n) + 2 * k) if rss > 0 else -np.inf
            params = unpack_params(name, popt)
            results.append({
                "model": name,
                "params": params,
                "n": n,
                "log_rmse": float(np.sqrt(rss / n)),
                "aic_log": aic,
                "failed": False,
            })
        except Exception as err:  # keep analysis robust on tiny smoke tests
            results.append({"model": name, "params": {}, "n": int(len(r)), "log_rmse": np.nan, "aic_log": np.nan, "failed": True, "error": str(err)})

    # The paper selects the best model by log-space RMSE.  AIC is retained
    # only as an additional diagnostic column in fit_results.csv.
    return sorted(
        results,
        key=lambda x: x["log_rmse"] if not x.get("failed", False) else np.inf,
    )


def unpack_params(name: str, popt: Sequence[float]) -> Dict[str, float]:
    if name == "pure_power":
        return {"A": float(np.exp(popt[0])), "p": float(popt[1])}
    if name == "power_floor":
        return {"A": float(np.exp(popt[0])), "p": float(popt[1]), "c": float(np.exp(popt[2]))}
    if name == "exp_floor":
        return {"A": float(np.exp(popt[0])), "xi": float(np.exp(popt[1])), "c": float(np.exp(popt[2]))}
    raise ValueError(name)


def eval_model(name: str, params: Dict[str, float], r: np.ndarray) -> np.ndarray:
    if name == "pure_power":
        return params["A"] * (r + 1.0) ** (-params["p"])
    if name == "power_floor":
        return params["c"] + params["A"] * (r + 1.0) ** (-params["p"])
    if name == "exp_floor":
        return params["c"] + params["A"] * np.exp(-r / params["xi"])
    raise ValueError(name)


def positive_limits(*arrays: np.ndarray) -> Tuple[float, float]:
    x = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return 1e-6, 1.0
    lo = max(float(np.quantile(x, 0.005)) * 0.7, 1e-300)
    hi = max(float(np.quantile(x, 0.995)) * 1.3, lo * 10.0)
    return lo, hi


def plot_individual(
    outpath: Path,
    r_values: np.ndarray,
    values: np.ndarray,
    quantity: str,
    max_rows: int,
    seed: int,
    plot_r_min: float,
) -> None:
    rng = np.random.default_rng(seed)
    rows = np.arange(values.shape[0])
    if len(rows) > max_rows:
        rows = rng.choice(rows, size=max_rows, replace=False)
    stats = profile_stats(values)
    keep_r = r_values.astype(float) >= float(plot_r_min)
    if not np.any(keep_r):
        return
    x = r_values[keep_r].astype(float) + 1.0

    plt.figure(figsize=(8.0, 5.0))
    for row in rows:
        y = values[row, keep_r]
        ok = np.isfinite(y) & (y > 0)
        if np.any(ok):
            plt.plot(x[ok], y[ok], linewidth=0.75, alpha=0.28)
    for key, linewidth in [("median", 2.8), ("trim10", 2.0), ("mean", 1.6)]:
        y = stats[key][keep_r]
        ok = np.isfinite(y) & (y > 0)
        if np.any(ok):
            plt.plot(x[ok], y[ok], linewidth=linewidth, label=key)

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(*positive_limits(values[np.ix_(rows, keep_r)], stats["median"][keep_r]))
    plt.xlabel("causal token distance r + 1")
    plt.ylabel(quantity)
    plt.title(f"Individual profiles: {quantity}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_aggregate_fit(
    outpath: Path,
    r_values: np.ndarray,
    values: np.ndarray,
    quantity: str,
    stat: str,
    fit_results: List[Dict[str, object]],
    r_min_fit: float,
    r_max_fit: float | None,
    plot_r_min: float,
) -> None:
    stats = profile_stats(values)
    keep_r = r_values.astype(float) >= float(plot_r_min)
    if r_max_fit is not None:
        keep_r &= r_values.astype(float) <= float(r_max_fit)
    if not np.any(keep_r):
        return
    x = r_values[keep_r].astype(float) + 1.0
    y = stats[stat][keep_r]

    plt.figure(figsize=(8.0, 5.0))
    q25 = stats["q25"][keep_r]
    q75 = stats["q75"][keep_r]
    band = np.isfinite(q25) & np.isfinite(q75) & (q25 > 0) & (q75 > 0)
    if np.any(band):
        plt.fill_between(x[band], q25[band], q75[band], alpha=0.20, label="25--75% quantile")

    for key, linewidth in [("median", 2.8), ("trim10", 2.0), ("mean", 1.6)]:
        yy = stats[key][keep_r]
        ok = np.isfinite(yy) & (yy > 0)
        if np.any(ok):
            plt.plot(x[ok], yy[ok], marker="o", linewidth=linewidth, label=key)

    if fit_results:
        fit_start = max(float(r_min_fit), float(plot_r_min))
        fit_candidates = r_values.astype(float)
        fit_candidates = fit_candidates[fit_candidates >= fit_start]
        if r_max_fit is not None:
            fit_candidates = fit_candidates[fit_candidates <= float(r_max_fit)]
        if fit_candidates.size > 0:
            fit_r_max = float(np.max(fit_candidates))
            r_dense = np.geomspace(fit_start + 1.0, fit_r_max + 1.0, 400) - 1.0
            for res in fit_results:
                if res.get("failed", False):
                    continue
                pred = eval_model(str(res["model"]), res["params"], r_dense)
                label = f"fit: {res['model']}"
                if "p" in res["params"]:
                    label += f" (p={res['params']['p']:.3g})"
                plt.plot(r_dense + 1.0, pred, linestyle="--", linewidth=2.0, label=label)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(*positive_limits(q25, q75, y))
    plt.xlabel("causal token distance r + 1")
    plt.ylabel(quantity)
    plt.title(f"{quantity}: aggregate profiles and fits to {stat}, r >= {r_min_fit:g}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def save_row_metadata(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(path: Path, r_values: np.ndarray, arrays: Dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "r", "count", "mean", "trim10", "median", "q10", "q25", "q75", "q90"])
        for name, values in arrays.items():
            stats = profile_stats(values)
            for k, r in enumerate(r_values):
                writer.writerow([
                    name, int(r), int(stats["count"][k]),
                    float(stats["mean"][k]), float(stats["trim10"][k]), float(stats["median"][k]),
                    float(stats["q10"][k]), float(stats["q25"][k]), float(stats["q75"][k]), float(stats["q90"][k]),
                ])


def save_fits(path: Path, all_fits: Dict[str, List[Dict[str, object]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "rank", "model", "log_rmse", "aic_log", "n", "params", "failed", "error"])
        for quantity, fits in all_fits.items():
            for rank, res in enumerate(fits, start=1):
                writer.writerow([
                    quantity, rank, res.get("model", ""), res.get("log_rmse", ""), res.get("aic_log", ""),
                    res.get("n", ""), json.dumps(res.get("params", {}), sort_keys=True),
                    bool(res.get("failed", False)), res.get("error", ""),
                ])


def write_outputs(args: argparse.Namespace, arrays: Dict[str, np.ndarray]) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.plot_prefix or slugify(args.model)

    r_values = arrays["r_values"]
    quantities = {
        "gradnorm": arrays["gradnorm"],
        "gradnorm_diag": arrays["gradnorm_diag"],
        "pred_gradnorm": arrays["pred_gradnorm"],
        "pred_gradnorm_diag": arrays["pred_gradnorm_diag"],
    }

    np.save(outdir / "r_values.npy", r_values)
    for name, values in quantities.items():
        np.save(outdir / f"autograd_green_{name}.npy", values)

    row_metadata = list(arrays["row_metadata"])
    save_row_metadata(outdir / "row_metadata.csv", row_metadata)
    save_summary(outdir / "summary.csv", r_values, quantities)

    all_fits: Dict[str, List[Dict[str, object]]] = {}
    for name, values in quantities.items():
        stats = profile_stats(values)
        fits = fit_models(r_values.astype(float), stats[args.fit_stat], args.r_min_fit, args.r_max_fit)
        all_fits[name] = fits

        # Individual profiles are most useful for the normalized quantities,
        # but writing all four is still cheap and keeps the naming uniform.
        plot_individual(
            outdir / f"{prefix}_{name}_individual_loglog.png",
            r_values,
            values,
            name,
            args.max_individual_rows,
            args.seed,
            args.plot_r_min,
        )
        plot_aggregate_fit(
            outdir / f"{prefix}_{name}_aggregate_fit_loglog.png",
            r_values,
            values,
            name,
            args.fit_stat,
            fits,
            args.r_min_fit,
            args.r_max_fit,
            args.plot_r_min,
        )

    save_fits(outdir / "fit_results.csv", all_fits)

    metadata = {
        "model": args.model,
        "dataset_source": args.dataset_source,
        "local_docs": args.local_docs,
        "seq_len": args.seq_len,
        "n_samples": args.n_samples,
        "target_j_min": args.target_j_min,
        "target_j_max": args.target_j_max,
        "target_stride": args.target_stride,
        "source_min_index": args.source_min_index,
        "plot_r_min": args.plot_r_min,
        "r_values": list(map(int, r_values)),
        "fit_stat": args.fit_stat,
        "r_min_fit": args.r_min_fit,
        "r_max_fit": args.r_max_fit,
        "rows": int(quantities["gradnorm"].shape[0]),
        "seed": args.seed,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nWrote output files to:", outdir)
    print("Key plot:", outdir / f"{prefix}_pred_gradnorm_diag_aggregate_fit_loglog.png")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal empirical Green-profile experiment for language models.")
    parser.add_argument("--model", default="EleutherAI/pythia-14m-deduped")
    parser.add_argument("--dataset-source", default="wikitext2", choices=["local-jsonl", "wikitext2", "wikitext103", "hf"])
    parser.add_argument("--local-docs", default="", help="JSONL with a text field, used with --dataset-source local-jsonl")
    parser.add_argument("--hf-dataset-name", default="")
    parser.add_argument("--hf-dataset-config", default="")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-doc-chars", type=int, default=2000)
    parser.add_argument("--max-docs", type=int, default=200)

    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--target-j-min", type=int, default=-1)
    parser.add_argument("--target-j-max", type=int, default=-1)
    parser.add_argument("--target-stride", type=int, default=32)
    parser.add_argument("--r-values", default="auto")
    parser.add_argument(
        "--source-min-index",
        type=int,
        default=128,
        help="Require every measured source position i=j-r to satisfy i >= this value. This avoids left-boundary hook artefacts.",
    )

    parser.add_argument("--fit-stat", default="median", choices=["median", "mean", "trim10"])
    parser.add_argument("--r-min-fit", type=float, default=1.0, help="Exclude r=0 from decay fits by default.")
    parser.add_argument("--r-max-fit", type=float, default=None)
    parser.add_argument(
        "--plot-r-min",
        type=float,
        default=1.0,
        help="Smallest distance shown in log-log figures. The default hides the artificial r=0 normalization point.",
    )

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model loading dtype. 'auto' preserves the dtype stored in the checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--outdir", default="green_minimal_run")
    parser.add_argument("--plot-prefix", default="", help="Prefix for PNG files, e.g. qwen490m")
    parser.add_argument("--max-individual-rows", type=int, default=80)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    arrays = compute_profiles(args)
    write_outputs(args, arrays)


if __name__ == "__main__":
    main()
