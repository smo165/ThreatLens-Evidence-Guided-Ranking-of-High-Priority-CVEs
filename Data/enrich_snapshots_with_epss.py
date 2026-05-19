#!/usr/bin/env python3
"""
Append cutoff-safe EPSS evidence to an existing ThreatLens snapshot CSV.

Input snapshot CSV is expected to contain at least:
  cve, cutoff_date, text

Output CSV preserves all columns and adds:
  epss_found, epss_asof_date, epss_score, epss_percentile

It also appends a TEMPORAL_EPSS_SIGNAL text block to the existing text field.

The script is intentionally an enrichment step, not a replacement for the snapshot
builder. This keeps the NVD-only baseline reproducible.

Data source:
  FIRST EPSS API: https://api.first.org/data/v1/epss?cve=...&date=YYYY-MM-DD

Temporal safety:
  For each row, the query date is the row's cutoff_date. If --fallback_days > 0,
  the script may use the most recent available EPSS date BEFORE the cutoff, never
  after the cutoff.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

EPSS_API = "https://api.first.org/data/v1/epss"
MIN_EPSS_DATE = date(2021, 4, 14)


def parse_date(x: object) -> Optional[date]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "nat", "none"}:
        return None
    try:
        return pd.to_datetime(s, utc=False).date()
    except Exception:
        return None


def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def cache_path(cache_dir: Path, query_date: date, cves: List[str]) -> Path:
    # Chunk cache identity is based on sorted CVEs. Keep filenames short enough.
    import hashlib

    key = "\n".join(sorted(cves)).encode("utf-8")
    h = hashlib.sha1(key).hexdigest()[:16]
    return cache_dir / f"epss_{query_date.isoformat()}_{len(cves)}_{h}.json"


def fetch_epss_batch(
    cves: List[str],
    query_date: date,
    cache_dir: Path,
    timeout: int = 60,
    sleep_seconds: float = 0.25,
    retries: int = 3,
) -> Dict[str, Tuple[str, float, float]]:
    """Return {CVE: (asof_date, epss, percentile)} for a date/CVE batch."""
    cves = sorted(set(cves))
    if not cves:
        return {}

    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_path(cache_dir, query_date, cves)
    if cp.exists():
        try:
            raw = json.loads(cp.read_text())
            return {
                k: (str(v[0]), float(v[1]), float(v[2]))
                for k, v in raw.items()
                if v is not None
            }
        except Exception:
            pass

    params = {
        "cve": ",".join(cves),
        "date": query_date.isoformat(),
    }
    url = EPSS_API + "?" + urlencode(params)

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "ThreatLens-EPSS-Enricher/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            out: Dict[str, Tuple[str, float, float]] = {}
            for item in payload.get("data", []):
                cve = item.get("cve")
                epss = item.get("epss")
                percentile = item.get("percentile")
                asof = item.get("date", query_date.isoformat())
                if cve is None or epss is None or percentile is None:
                    continue
                out[str(cve)] = (str(asof), float(epss), float(percentile))
            cp.write_text(json.dumps(out, sort_keys=True))
            time.sleep(sleep_seconds)
            return out
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(sleep_seconds * attempt * 2)

    print(f"WARNING: EPSS fetch failed for {query_date} batch size={len(cves)}: {last_err}", file=sys.stderr)
    return {}


def get_epss_for_cutoff(
    cves: List[str],
    cutoff: date,
    cache_dir: Path,
    batch_size: int,
    fallback_days: int,
    timeout: int,
    sleep_seconds: float,
) -> Dict[str, Tuple[str, float, float]]:
    """Fetch EPSS for CVEs as of cutoff, optionally falling back backward."""
    remaining = sorted(set(cves))
    results: Dict[str, Tuple[str, float, float]] = {}

    latest_allowed = cutoff
    if latest_allowed < MIN_EPSS_DATE:
        return results

    for delta in range(0, fallback_days + 1):
        query_date = cutoff - timedelta(days=delta)
        if query_date < MIN_EPSS_DATE:
            break
        if not remaining:
            break
        for batch in chunked(remaining, batch_size):
            got = fetch_epss_batch(
                batch,
                query_date,
                cache_dir=cache_dir,
                timeout=timeout,
                sleep_seconds=sleep_seconds,
            )
            results.update(got)
        remaining = [c for c in remaining if c not in results]

    return results


def format_epss_block(found: bool, asof: object, score: object, percentile: object) -> str:
    if not found or pd.isna(score) or pd.isna(percentile):
        return "\n\nTEMPORAL_EPSS_SIGNAL:\nEPSS_AVAILABLE_BY_CUTOFF: no"
    return (
        "\n\nTEMPORAL_EPSS_SIGNAL:\n"
        "EPSS_AVAILABLE_BY_CUTOFF: yes\n"
        #f"EPSS_AS_OF_DATE: {asof}\n"
        f"EPSS_SCORE: {float(score):.6f}\n"
        f"EPSS_PERCENTILE: {float(percentile):.6f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", required=True, help="Input snapshot CSV with cve, cutoff_date, text")
    ap.add_argument("--output", required=True, help="Output enriched snapshot CSV")
    ap.add_argument("--cache_dir", default="Data/epss_cache", help="Cache directory for EPSS API responses")
    ap.add_argument("--batch_size", type=int, default=100, help="CVEs per EPSS API request")
    ap.add_argument("--fallback_days", type=int, default=3, help="Use latest previous EPSS date within this many days if exact date is missing")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--sleep_seconds", type=float, default=0.25)
    ap.add_argument("--no_append_text", action="store_true", help="Add EPSS columns only; do not append EPSS block to text")
    args = ap.parse_args()

    df = pd.read_csv(args.snapshots)
    required = {"cve", "cutoff_date", "text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df["cutoff_date"] = df["cutoff_date"].astype(str)
    cutoff_dates = df["cutoff_date"].map(parse_date)
    if cutoff_dates.isna().any():
        bad = df.loc[cutoff_dates.isna(), "cutoff_date"].head().tolist()
        raise SystemExit(f"Could not parse some cutoff_date values, e.g. {bad}")

    cache_dir = Path(args.cache_dir)
    epss_found = [False] * len(df)
    epss_asof = [None] * len(df)
    epss_score = [float("nan")] * len(df)
    epss_percentile = [float("nan")] * len(df)

    grouped = df.groupby("cutoff_date", sort=True).indices
    print(f"Rows: {len(df)}")
    print(f"Distinct cutoff dates: {len(grouped)}")
    print(f"Cache dir: {cache_dir}")

    for i, (cutoff_s, idxs) in enumerate(grouped.items(), start=1):
        cutoff = parse_date(cutoff_s)
        assert cutoff is not None
        cves = df.loc[idxs, "cve"].astype(str).str.upper().tolist()
        got = get_epss_for_cutoff(
            cves,
            cutoff,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            fallback_days=args.fallback_days,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
        )
        found_count = 0
        for idx in idxs:
            cve = str(df.at[idx, "cve"]).upper()
            if cve in got:
                asof, score, pct = got[cve]
                epss_found[idx] = True
                epss_asof[idx] = asof
                epss_score[idx] = score
                epss_percentile[idx] = pct
                found_count += 1
        if i == 1 or i % 25 == 0 or i == len(grouped):
            print(f"[{i}/{len(grouped)}] {cutoff_s}: found {found_count}/{len(idxs)}")

    df["epss_found"] = epss_found
    df["epss_asof_date"] = epss_asof
    df["epss_score"] = epss_score
    df["epss_percentile"] = epss_percentile

    if not args.no_append_text:
        blocks = [format_epss_block(f, a, s, p) for f, a, s, p in zip(epss_found, epss_asof, epss_score, epss_percentile)]
        df["text"] = df["text"].fillna("").astype(str) + blocks

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nWrote EPSS-enriched snapshots: {out}")
    print(f"Rows: {len(df)}")
    print(f"EPSS found: {int(pd.Series(epss_found).sum())}")
    print(f"EPSS missing: {int((~pd.Series(epss_found)).sum())}")
    if epss_found:
        print("EPSS score summary:")
        print(df.loc[df["epss_found"], "epss_score"].describe())


if __name__ == "__main__":
    main()
