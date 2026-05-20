#!/usr/bin/env python3
"""
Ablate NVD-derived fields from ThreatLens model-facing snapshot text.

This is a post-processing utility: it does not rebuild weekly snapshots and does
not change labels, cutoffs, or enrichment columns. It edits only the model-facing
text column.

For the NVD metadata ablation, use:
  --drop-cvss --drop-cwe --drop-cpe

When --drop-cvss is enabled, this script removes both the structured NVD CVSS
line and CVSS score/vector fragments embedded inside NVD DESCRIPTION lines.
It does not remove advisory CVSS fields from GHSA/OSV blocks.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


DOWNSTREAM_SECTION_RE = re.compile(
    r"^(?:TEMPORAL_[A-Z0-9_]+|EPSS|GHSA|OSV|PUBLIC_EXPLOIT|METASPLOIT|EXPLOITDB)\b",
    re.IGNORECASE,
)


@dataclass
class AblationStats:
    rows: int = 0
    rows_changed: int = 0
    cvss_lines_removed: int = 0
    embedded_cvss_description_lines_scrubbed: int = 0
    embedded_cvss_mentions_removed: int = 0
    cwe_lines_removed: int = 0
    cpe_blocks_removed: int = 0
    cpe_lines_removed: int = 0
    description_lines_removed: int = 0
    age_lines_removed: int = 0
    nvd_header_removed: int = 0
    chars_before: int = 0
    chars_after: int = 0

    def add(self, other: "AblationStats") -> None:
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)


def normalize_blank_lines(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_line_if(text: str, predicate: Callable[[str], bool]) -> tuple[str, int]:
    lines = text.splitlines()
    kept: list[str] = []
    removed = 0
    for line in lines:
        if predicate(line):
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept), removed


def scrub_cvss_from_description_lines(text: str) -> tuple[str, int, int]:
    """Remove CVSS score/vector fragments embedded in NVD DESCRIPTION lines.

    We only edit lines beginning with DESCRIPTION:. This preserves advisory CVSS
    fields in downstream GHSA/OSV blocks, which are not part of the NVD metadata
    ablation.
    """
    lines = text.splitlines()
    changed_lines = 0
    mentions_removed = 0
    new_lines: list[str] = []

    # Remove common Oracle/VMware/NVD description fragments that encode CVSS.
    patterns = [
        # " CVSS v3.0 Base Score 4.3 (Integrity impacts)."
        r"\s*CVSS\s*v?\d(?:\.\d)?\s+Base\s+Score\s+[0-9.]+(?:\s*\([^)]*\))?\.?",
        # " CVSS 3.1 Base Score 9.8 (...)."
        r"\s*CVSS\s+\d(?:\.\d)?\s+Base\s+Score\s+[0-9.]+(?:\s*\([^)]*\))?\.?",
        # " CVSS Base Score 9.8 (...)."
        r"\s*CVSS\s+Base\s+Score\s+[0-9.]+(?:\s*\([^)]*\))?\.?",
        # " CVSSv3 base score of 9.8."
        r"\s*CVSSv?\d(?:\.\d)?\s+base\s+score\s+of\s+[0-9.]+\.?",
        # " CVSS Vector: (CVSS:3.1/AV:N/...)."
        r"\s*CVSS\s+Vector:\s*\([^)]*\)\.?",
        # " CVSS Vector: CVSS:3.1/AV:N/..."
        r"\s*CVSS\s+Vector:\s*\S+\.?",
        # bare vector fragments
        r"\s*\(CVSS:\d(?:\.\d)?/[^)]*\)\.?",
        r"\s*CVSS:\d(?:\.\d)?/[^\s.)]+\.?",
    ]

    for line in lines:
        if not line.startswith("DESCRIPTION:") or "CVSS" not in line:
            new_lines.append(line)
            continue

        before = line
        after = line
        removed_here = 0
        for pat in patterns:
            after, n = re.subn(pat, "", after, flags=re.IGNORECASE)
            removed_here += n

        # If anything with CVSS remains in the DESCRIPTION line, remove sentence
        # fragments containing CVSS while keeping the rest of the description.
        if "CVSS" in after:
            prefix = "DESCRIPTION:"
            body = after[len(prefix):].strip()
            parts = re.split(r"(?<=[.!?])\s+", body)
            kept_parts = [p for p in parts if "CVSS" not in p]
            removed_here += len(parts) - len(kept_parts)
            after = prefix + (" " + " ".join(kept_parts).strip() if kept_parts else "")

        after = re.sub(r"\s{2,}", " ", after).strip()

        if after != before:
            changed_lines += 1
            mentions_removed += max(removed_here, 1)

        new_lines.append(after)

    return "\n".join(new_lines), changed_lines, mentions_removed


def drop_cpe_block(text: str) -> tuple[str, int, int]:
    lines = text.splitlines()
    kept: list[str] = []
    i = 0
    blocks_removed = 0
    lines_removed = 0

    while i < len(lines):
        line = lines[i]
        if line.strip() == "CPES:":
            blocks_removed += 1
            lines_removed += 1
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                lines_removed += 1
                i += 1
            continue
        kept.append(line)
        i += 1

    return "\n".join(kept), blocks_removed, lines_removed


def drop_empty_nvd_header_if_requested(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    if not lines:
        return text, 0

    try:
        idx = next(i for i, line in enumerate(lines) if line.strip() == "NVD_RECORD:")
    except StopIteration:
        return text, 0

    has_nvd_content = False
    for line in lines[idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if DOWNSTREAM_SECTION_RE.match(stripped):
            break
        has_nvd_content = True
        break

    if has_nvd_content:
        return text, 0

    return "\n".join(lines[:idx] + lines[idx + 1 :]), 1


def ablate_text(
    text: str,
    *,
    drop_cvss: bool = False,
    drop_cwe: bool = False,
    drop_cpe: bool = False,
    drop_description: bool = False,
    drop_age_context: bool = False,
    drop_empty_nvd_header: bool = False,
) -> tuple[str, AblationStats]:
    text = "" if pd.isna(text) else str(text)
    original = text
    stats = AblationStats(rows=1, chars_before=len(original))

    if drop_cvss:
        text, removed = remove_line_if(text, lambda line: line.lstrip().startswith("CVSS:"))
        stats.cvss_lines_removed += removed
        text, changed_lines, mentions_removed = scrub_cvss_from_description_lines(text)
        stats.embedded_cvss_description_lines_scrubbed += changed_lines
        stats.embedded_cvss_mentions_removed += mentions_removed

    if drop_cwe:
        text, removed = remove_line_if(text, lambda line: line.lstrip().startswith("CWES:"))
        stats.cwe_lines_removed += removed

    if drop_cpe:
        text, blocks, removed_lines = drop_cpe_block(text)
        stats.cpe_blocks_removed += blocks
        stats.cpe_lines_removed += removed_lines

    if drop_description:
        text, removed = remove_line_if(text, lambda line: line.lstrip().startswith("DESCRIPTION:"))
        stats.description_lines_removed += removed

    if drop_age_context:
        text, removed = remove_line_if(
            text,
            lambda line: line.lstrip().startswith("DAYS_SINCE_PUBLICATION:")
            or line.lstrip().startswith("AGE_BUCKET:"),
        )
        stats.age_lines_removed += removed

    if drop_empty_nvd_header:
        text, removed = drop_empty_nvd_header_if_requested(text)
        stats.nvd_header_removed += removed

    text = normalize_blank_lines(text)
    stats.chars_after = len(text)
    stats.rows_changed = int(text != original)
    return text, stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create NVD-field ablation versions of a ThreatLens snapshot CSV by editing only model-facing text."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-col", default="text")

    ap.add_argument("--drop-cvss", action="store_true")
    ap.add_argument("--drop-cwe", action="store_true")
    ap.add_argument("--drop-cpe", action="store_true")
    ap.add_argument("--drop-description", action="store_true")
    ap.add_argument("--drop-age-context", action="store_true")
    ap.add_argument("--drop-empty-nvd-header", action="store_true")

    ap.add_argument(
        "--nvd-minimal",
        action="store_true",
        help="Equivalent to --drop-cvss --drop-cwe --drop-cpe. Keeps DESCRIPTION and age context.",
    )
    ap.add_argument(
        "--nvd-description-only",
        action="store_true",
        help="Keep only DESCRIPTION from the NVD block; remove CVSS, CWE, CPE, and age context.",
    )
    ap.add_argument("--manifest", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.nvd_minimal:
        args.drop_cvss = True
        args.drop_cwe = True
        args.drop_cpe = True

    if args.nvd_description_only:
        args.drop_cvss = True
        args.drop_cwe = True
        args.drop_cpe = True
        args.drop_age_context = True

    if not any(
        [
            args.drop_cvss,
            args.drop_cwe,
            args.drop_cpe,
            args.drop_description,
            args.drop_age_context,
            args.drop_empty_nvd_header,
        ]
    ):
        raise SystemExit("No ablation was selected.")

    in_path = Path(args.input)
    out_path = Path(args.output)

    df = pd.read_csv(in_path)
    if args.text_col not in df.columns:
        raise ValueError(f"Input CSV does not contain text column {args.text_col!r}. Available columns: {list(df.columns)}")

    total = AblationStats()
    new_texts: list[str] = []
    for value in df[args.text_col].tolist():
        new_text, row_stats = ablate_text(
            value,
            drop_cvss=args.drop_cvss,
            drop_cwe=args.drop_cwe,
            drop_cpe=args.drop_cpe,
            drop_description=args.drop_description,
            drop_age_context=args.drop_age_context,
            drop_empty_nvd_header=args.drop_empty_nvd_header,
        )
        new_texts.append(new_text)
        total.add(row_stats)

    df[args.text_col] = new_texts
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    manifest = {
        "script": "Data/ablations/ablate_nvd_text.py",
        "input": str(in_path),
        "output": str(out_path),
        "text_col": args.text_col,
        "settings": {
            "drop_cvss": args.drop_cvss,
            "drop_cwe": args.drop_cwe,
            "drop_cpe": args.drop_cpe,
            "drop_description": args.drop_description,
            "drop_age_context": args.drop_age_context,
            "drop_empty_nvd_header": args.drop_empty_nvd_header,
            "nvd_minimal": args.nvd_minimal,
            "nvd_description_only": args.nvd_description_only,
        },
        "stats": asdict(total),
    }

    manifest_path = Path(args.manifest) if args.manifest else out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote ablated CSV: {out_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Rows: {total.rows}")
    print(f"Rows changed: {total.rows_changed}")
    print(f"CVSS lines removed: {total.cvss_lines_removed}")
    print(f"Embedded CVSS description lines scrubbed: {total.embedded_cvss_description_lines_scrubbed}")
    print(f"Embedded CVSS mentions removed: {total.embedded_cvss_mentions_removed}")
    print(f"CWE lines removed: {total.cwe_lines_removed}")
    print(f"CPE blocks removed: {total.cpe_blocks_removed}")
    print(f"CPE lines removed: {total.cpe_lines_removed}")
    print(f"Description lines removed: {total.description_lines_removed}")
    print(f"Age-context lines removed: {total.age_lines_removed}")
    print(f"Characters before: {total.chars_before}")
    print(f"Characters after: {total.chars_after}")


if __name__ == "__main__":
    main()
