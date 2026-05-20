#!/usr/bin/env python3
"""
Create a No-EPSS ablation CSV for ThreatLens by removing EPSS evidence
from the model-facing text column only.

This script is intentionally a post-processing utility:
- It does NOT rebuild snapshots.
- It does NOT change labels, cutoffs, splits, CVE IDs, or enrichment columns.
- It only removes the TEMPORAL_EPSS_SIGNAL block from the text column.

Expected input:
  Data/snapshots_weekly_kev_forecast_epss_ghsa_exploits.csv

Example:
  python Data/ablations/make_no_epss_ablation.py \
    --input Data/snapshots_weekly_kev_forecast_epss_ghsa_exploits.csv \
    --output Data/ablations/snapshots_no_epss.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


EPSS_BLOCK_RE = re.compile(
    # Actual block emitted by Data/enrich_snapshots_with_epss.py:
    # TEMPORAL_EPSS_SIGNAL:
    # ...
    # Stop before next TEMPORAL_* block or end of text.
    r"\n{0,2}TEMPORAL_EPSS_SIGNAL:\n.*?(?=\n{1,2}TEMPORAL_[A-Z0-9_]+:|\Z)",
    re.DOTALL,
)


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove TEMPORAL_EPSS_SIGNAL blocks from ThreatLens model-facing text."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input enriched snapshot CSV, usually Data/snapshots_weekly_kev_forecast_epss_ghsa_exploits.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output No-EPSS ablated CSV path, e.g. Data/ablations/snapshots_no_epss.csv",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Model-facing text column name. Default: text",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON manifest path. Default: <output>.manifest.json",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification checks after writing the ablated file.",
    )
    return parser.parse_args()


def count_token(df: pd.DataFrame, text_col: str, token: str) -> int:
    return int(df[text_col].fillna("").astype(str).str.contains(token, regex=False).sum())


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(output_path.suffix + ".manifest.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path)
    if args.text_col not in df.columns:
        raise ValueError(
            f"Input CSV does not contain text column {args.text_col!r}. "
            f"Available columns: {list(df.columns)}"
        )

    original_text = df[args.text_col].fillna("").astype(str)

    rows_changed = 0
    epss_blocks_removed = 0
    chars_before = 0
    chars_after = 0
    new_texts: list[str] = []

    for text in original_text:
        chars_before += len(text)
        ablated_text, removed_count = EPSS_BLOCK_RE.subn("", text)
        ablated_text = clean_text(ablated_text)

        epss_blocks_removed += removed_count
        rows_changed += int(ablated_text != text)
        chars_after += len(ablated_text)
        new_texts.append(ablated_text)

    df[args.text_col] = new_texts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    manifest: dict[str, Any] = {
        "ablation": "no_epss",
        "input": str(input_path),
        "output": str(output_path),
        "text_col": args.text_col,
        "rows": int(len(df)),
        "rows_changed": int(rows_changed),
        "epss_blocks_removed": int(epss_blocks_removed),
        "chars_before": int(chars_before),
        "chars_after": int(chars_after),
        "removed_block": "TEMPORAL_EPSS_SIGNAL",
        "note": (
            "Removed TEMPORAL_EPSS_SIGNAL blocks from model-facing text only. "
            "Other columns, labels, cutoffs, and splits are unchanged."
        ),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote ablated CSV: {output_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Rows: {len(df)}")
    print(f"Rows changed: {rows_changed}")
    print(f"EPSS blocks removed: {epss_blocks_removed}")
    print(f"Characters before: {chars_before}")
    print(f"Characters after: {chars_after}")

    if args.verify:
        ablated = pd.read_csv(output_path)
        print("\nVerification")
        print("------------")
        print(f"Same row count: {len(df) == len(ablated)}")
        for token in [
            "TEMPORAL_EPSS_SIGNAL:",
            "TEMPORAL_GHSA_OSV_ADVISORY:",
            "TEMPORAL_PUBLIC_EXPLOIT_EVIDENCE:",
        ]:
            before_count = count_token(pd.read_csv(input_path), args.text_col, token)
            after_count = count_token(ablated, args.text_col, token)
            print(f"{token} before={before_count} after={after_count}")

        remaining_epss = count_token(ablated, args.text_col, "TEMPORAL_EPSS_SIGNAL:")
        if remaining_epss != 0:
            raise RuntimeError(
                f"No-EPSS ablation failed: {remaining_epss} rows still contain TEMPORAL_EPSS_SIGNAL."
            )


if __name__ == "__main__":
    main()
