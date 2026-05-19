# build_events_kev.py
"""
Build a standardized KEV events CSV.

Output columns:
- cve
- event_type
- event_date
- source
- raw_date_field

Supported input formats:
1) JSON list/dict from CISA KEV-like exports
2) CSV exports containing at least CVE + date-added style field

Usage examples
--------------
python build_events_kev.py --input kev.json --output events_kev.csv
python build_events_kev.py --input known_exploited_vulnerabilities.csv --output events_kev.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
import re

import pandas as pd


CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

CVE_CANDIDATE_COLUMNS = [
    "cve", "cveid", "cve_id", "cveID", "vulnerabilityid", "vulnerability_id"
]

DATE_CANDIDATE_COLUMNS = [
    "dateadded", "date_added", "dateAdded", "added", "added_date", "known_ransomware_campaign_use"
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def extract_first_cve(value: object) -> str | None:
    if value is None:
        return None
    m = CVE_RE.search(str(value))
    return m.group(0).upper() if m else None


def load_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return normalize_columns(pd.read_csv(path))
    if suffix in {".json", ".js"}:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # common KEV shape: {"vulnerabilities": [...]}
            if "vulnerabilities" in data and isinstance(data["vulnerabilities"], list):
                data = data["vulnerabilities"]
            else:
                data = [data]
        return normalize_columns(pd.DataFrame(data))
    raise ValueError(f"Unsupported KEV input format: {path.suffix}")


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    cve_col = find_column(df.columns, CVE_CANDIDATE_COLUMNS)
    if cve_col is None:
        # try extracting CVE from any column
        extracted = None
        for col in df.columns:
            tmp = df[col].map(extract_first_cve)
            if tmp.notna().any():
                extracted = tmp
                break
        if extracted is None:
            raise ValueError("Could not find a KEV CVE column.")
        df = df.copy()
        df["_cve"] = extracted
        cve_col = "_cve"

    date_col = find_column(df.columns, DATE_CANDIDATE_COLUMNS)
    if date_col is None:
        # try any date-looking column name
        for col in df.columns:
            if "date" in col.lower():
                date_col = col
                break
    if date_col is None:
        raise ValueError("Could not find a KEV date-added column.")

    out = pd.DataFrame({
        "cve": df[cve_col].map(extract_first_cve),
        "event_type": "KEV",
        "event_date": pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.date.astype("string"),
        "source": "KEV",
        "raw_date_field": date_col,
    })

    out = out.dropna(subset=["cve", "event_date"]).drop_duplicates(subset=["cve", "event_type", "event_date"])
    out = out.sort_values(["event_date", "cve"]).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build standardized KEV events CSV.")
    ap.add_argument("--input", required=True, help="Path to KEV JSON or CSV")
    ap.add_argument("--output", required=True, help="Path to output CSV")
    args = ap.parse_args()

    df = load_any(Path(args.input))
    out = build_events(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} KEV rows to {args.output}")


if __name__ == "__main__":
    main()
