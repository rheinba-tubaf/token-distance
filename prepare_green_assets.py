#!/usr/bin/env python3
"""Prepare a local JSONL file of long Gutenberg-style documents.

This is intentionally small. It writes records of the form

    {"text": "...", "source_dataset": "...", ...}

The measurement script can then be run with

    --dataset-source local-jsonl --local-docs green_assets/gutenberg_longdocs_1500.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import load_dataset


GUTENBERG_CANDIDATES = [
    "sedthh/gutenberg_english",
    "adhyanshaa/project-gutenberg-en",
    "incredible45/Gutenberg-BookCorpus-Cleaned-Data-English",
    "manu/project_gutenberg",
    "deepmind/pg19",
    "emozilla/pg19",
    "Tanushreeeeee/pg19",
]


def unique(xs: Iterable[str]) -> List[str]:
    out: List[str] = []
    for x in xs:
        if x and x not in out:
            out.append(x)
    return out


def candidate_splits(split: str) -> List[str]:
    return unique([split, "train", "en"])


def choose_text_column(columns: List[str], requested: str) -> str:
    preferred = [
        requested,
        "text", "TEXT", "book_text", "content", "cleaned_text",
        "body", "book", "full_text", "Text",
    ]
    for name in preferred:
        if name in columns:
            return name
    for name in columns:
        if "text" in name.lower():
            return name
    raise ValueError(f"Could not infer text column from columns: {columns}")


def load_streaming_dataset(name: str, split: str, token: Optional[str]):
    try:
        return load_dataset(name, split=split, streaming=True, token=token)
    except TypeError:
        if token:
            return load_dataset(name, split=split, streaming=True, use_auth_token=token)
        return load_dataset(name, split=split, streaming=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare long local documents for the Green-profile paper examples.")
    parser.add_argument("--outdir", default="green_assets")
    parser.add_argument("--outfile", default="", help="Override output JSONL path.")
    parser.add_argument("--hf-dataset-name", default="", help="Try this dataset first.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--min-doc-chars", type=int, default=2000)
    parser.add_argument("--max-docs", type=int, default=1500)
    parser.add_argument("--max-scan-factor", type=int, default=20)
    parser.add_argument("--min-scan", type=int, default=2000)
    parser.add_argument("--hf-token", default="")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = Path(args.outfile) if args.outfile else outdir / f"gutenberg_longdocs_{args.max_docs}.jsonl"
    metafile = outfile.with_name(outfile.stem + "_metadata.json")

    names = unique(([args.hf_dataset_name] if args.hf_dataset_name else []) + GUTENBERG_CANDIDATES)
    last_error: Exception | None = None
    token = args.hf_token or None

    for dataset_name in names:
        for split in candidate_splits(args.split):
            try:
                print(f"Trying dataset={dataset_name!r}, split={split!r}")
                ds = load_streaming_dataset(dataset_name, split, token)
                scanned = 0
                written = 0
                text_column = None
                max_scan = max(args.max_docs * args.max_scan_factor, args.min_scan)

                with outfile.open("w", encoding="utf-8") as f:
                    for ex in ds:
                        scanned += 1
                        if text_column is None:
                            text_column = choose_text_column(list(ex.keys()), args.text_column)
                            print(f"  using text column {text_column!r}")
                        text = str(ex.get(text_column, ""))
                        if len(text) >= args.min_doc_chars:
                            record = {
                                "text": text,
                                "source_dataset": dataset_name,
                                "split": split,
                                "source_index": scanned - 1,
                                "n_chars": len(text),
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            written += 1
                            if written % 50 == 0:
                                print(f"  written {written}; scanned {scanned}")
                            if written >= args.max_docs:
                                break
                        if scanned >= max_scan and written > 0:
                            break

                if written > 0:
                    metadata = {
                        "source_dataset": dataset_name,
                        "split": split,
                        "text_column": text_column,
                        "docs_written": written,
                        "examples_scanned": scanned,
                        "min_doc_chars": args.min_doc_chars,
                        "outfile": str(outfile),
                    }
                    metafile.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    print(f"Wrote {written} documents to {outfile}")
                    print(f"Metadata: {metafile}")
                    return

                last_error = RuntimeError(f"No long documents found in {dataset_name!r}, split={split!r}")
                print(f"  failed: {last_error}")
            except Exception as err:
                last_error = err
                print(f"  failed: {err}")

    raise RuntimeError(f"Could not prepare long documents. Last error: {last_error}")


if __name__ == "__main__":
    main()
