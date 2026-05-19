#!/usr/bin/env python3
"""
Enrich ThreatLens snapshot CSVs with cutoff-safe GHSA/OSV advisory text.

This is a modular post-processing step. It works with any ThreatLens snapshot
builder output as long as the CSV has:
    cve, cutoff_date, text

Example:
    python Data/enrich_snapshots_with_ghsa_osv.py \
      --snapshots Data/snapshots_text_nn_aligned_epss.csv \
      --output Data/snapshots_text_nn_aligned_epss_ghsa.csv \
      --advisory_dirs Data/external/github-advisory-database/advisories

Temporal safety:
    Strict mode is the default:
      include advisory only if published <= cutoff AND modified <= cutoff.

    Less strict exploratory mode:
      --allow_modified_after_cutoff
      include advisory if published <= cutoff, even if modified > cutoff.
      This may introduce post-cutoff text changes, so do not use it as the
      main leakage-safe result unless you explicitly discuss the limitation.

Model-text shortcut policy:
    The model text intentionally excludes exact advisory IDs and exact
    published/modified dates. Those are kept as CSV audit columns only.

    Text includes relative advisory age, severity, affected packages,
    ecosystem/package names, summary, and details.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
GHSA_RE = re.compile(r"\bGHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}\b", re.IGNORECASE)


def parse_dt(value: Any) -> Optional[pd.Timestamp]:
    """Parse OSV RFC3339-ish timestamps to naive normalized dates."""
    if value is None or value == "":
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.tz_convert(None).normalize()
    except Exception:
        return None


def clean_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if limit > 0 and len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text

def scrub_model_text_ids(text: str) -> str:
    """Replace raw CVE/GHSA identifiers in model text while keeping audit columns unchanged."""
    text = CVE_RE.sub("CVE_TOKEN", text)
    text = GHSA_RE.sub("GHSA_TOKEN", text)
    return text


def date_to_str(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return ts.date().isoformat()


@dataclass(frozen=True)
class AdvisoryRecord:
    advisory_id: str
    published: Optional[pd.Timestamp]
    modified: Optional[pd.Timestamp]
    cves: List[str]
    summary: str
    details: str
    severity: List[str]
    affected_packages: List[str]
    source_file: str


def iter_json_files(paths: List[Path]) -> Iterable[Path]:
    for root in paths:
        if not root.exists():
            print(f"WARNING: advisory path does not exist: {root}")
            continue
        if root.is_file() and root.suffix.lower() == ".json":
            yield root
        elif root.is_dir():
            yield from root.rglob("*.json")


def extract_cves(obj: Dict[str, Any], advisory_id: str) -> List[str]:
    candidates: List[str] = []
    for key in ("aliases", "related", "upstream"):
        vals = obj.get(key, [])
        if isinstance(vals, list):
            candidates.extend(str(x) for x in vals if isinstance(x, str))
    candidates.append(advisory_id)

    cves = sorted({m.group(0).upper() for s in candidates for m in CVE_RE.finditer(s)})
    if cves:
        return cves

    # Fallback for imperfect records that mention the CVE in text but do not
    # list it in aliases. This is only used for indexing, not for labeling.
    scan = f"{obj.get('summary', '')} {obj.get('details', '')}"
    return sorted({m.group(0).upper() for m in CVE_RE.finditer(scan)})


def extract_severity(obj: Dict[str, Any], max_items: int = 6) -> List[str]:
    parts: List[str] = []

    sev = obj.get("severity", [])
    if isinstance(sev, list):
        for item in sev:
            if not isinstance(item, dict):
                continue
            typ = item.get("type")
            score = item.get("score")
            if typ and score:
                parts.append(f"{typ}={score}")
            elif score:
                parts.append(str(score))

    db = obj.get("database_specific", {})
    if isinstance(db, dict) and db.get("severity"):
        parts.append(f"github_severity={db.get('severity')}")

    out = []
    seen = set()
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out[:max_items]


def extract_packages(obj: Dict[str, Any], max_packages: int) -> List[str]:
    packages: List[str] = []
    affected = obj.get("affected", [])
    if not isinstance(affected, list):
        return []

    for aff in affected:
        if not isinstance(aff, dict):
            continue
        pkg = aff.get("package", {})
        if not isinstance(pkg, dict):
            continue
        ecosystem = pkg.get("ecosystem")
        name = pkg.get("name")
        purl = pkg.get("purl")
        if ecosystem and name:
            packages.append(f"{ecosystem}:{name}")
        elif purl:
            packages.append(str(purl))

    unique = []
    seen = set()
    for p in packages:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique[:max_packages]


def load_advisory(path: Path, max_summary_chars: int, max_details_chars: int, max_packages: int) -> Optional[AdvisoryRecord]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        print(f"WARNING: failed to read {path}: {exc}")
        return None

    if not isinstance(obj, dict):
        return None

    advisory_id = str(obj.get("id") or path.stem)
    cves = extract_cves(obj, advisory_id)
    if not cves:
        return None

    return AdvisoryRecord(
        advisory_id=advisory_id,
        published=parse_dt(obj.get("published")),
        modified=parse_dt(obj.get("modified")),
        cves=cves,
        summary=clean_text(obj.get("summary", ""), max_summary_chars),
        details=clean_text(obj.get("details", ""), max_details_chars),
        severity=extract_severity(obj),
        affected_packages=extract_packages(obj, max_packages=max_packages),
        source_file=str(path),
    )


def build_index(advisory_dirs: List[Path], max_summary_chars: int, max_details_chars: int, max_packages: int) -> Dict[str, List[AdvisoryRecord]]:
    index: Dict[str, List[AdvisoryRecord]] = {}
    scanned = 0
    with_cve = 0

    for path in iter_json_files(advisory_dirs):
        scanned += 1
        rec = load_advisory(
            path,
            max_summary_chars=max_summary_chars,
            max_details_chars=max_details_chars,
            max_packages=max_packages,
        )
        if rec is None:
            continue
        with_cve += 1
        for cve in rec.cves:
            index.setdefault(cve.upper(), []).append(rec)

    for cve, recs in index.items():
        recs.sort(key=lambda r: (r.published or pd.Timestamp.max, r.advisory_id))

    print(f"Scanned JSON files: {scanned}")
    print(f"Advisory records with CVE aliases/text: {with_cve}")
    print(f"Indexed CVEs: {len(index)}")
    return index


def age_bucket(days: Optional[int]) -> str:
    if days is None:
        return "unknown"
    if days < 0:
        return "future"
    if days <= 7:
        return "0-7d"
    if days <= 30:
        return "8-30d"
    if days <= 90:
        return "31-90d"
    if days <= 180:
        return "91-180d"
    if days <= 365:
        return "181-365d"
    if days <= 730:
        return "1-2y"
    return "2y+"


def render_advisory_block(rec: AdvisoryRecord, cutoff: pd.Timestamp) -> str:
    # No advisory ID, published date, or modified date in model text.
    # These remain available in audit columns.
    lines = ["TEMPORAL_GHSA_OSV_ADVISORY:"]

    if rec.published is not None:
        days = int((cutoff - rec.published).days)
        lines.append(f"ADVISORY_DAYS_SINCE_PUBLICATION: {days}")
        lines.append(f"ADVISORY_AGE_BUCKET: {age_bucket(days)}")

    if rec.severity:
        lines.append("ADVISORY_SEVERITY: " + "; ".join(rec.severity))

    if rec.affected_packages:
        lines.append("AFFECTED_PACKAGES:")
        for pkg in rec.affected_packages:
            lines.append(f"- {scrub_model_text_ids(pkg)}")

    if rec.summary:
        lines.append("ADVISORY_SUMMARY: " + scrub_model_text_ids(rec.summary))

    if rec.details:
        lines.append("ADVISORY_DETAILS: " + scrub_model_text_ids(rec.details))

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich ThreatLens snapshots with cutoff-safe GHSA/OSV advisory text.")
    ap.add_argument("--snapshots", required=True, help="Input snapshots CSV with cve, cutoff_date, text columns.")
    ap.add_argument("--output", required=True, help="Output enriched snapshots CSV.")
    ap.add_argument(
        "--advisory_dirs",
        nargs="+",
        required=True,
        help="One or more directories/files containing OSV-format JSON advisories, e.g. GitHub Advisory Database advisories/.",
    )
    ap.add_argument(
        "--allow_modified_after_cutoff",
        action="store_true",
        help="Exploratory mode: include advisories published by cutoff even if modified after cutoff. Default strict mode requires modified <= cutoff too.",
    )
    ap.add_argument("--max_records_per_cve", type=int, default=2)
    ap.add_argument("--max_summary_chars", type=int, default=600)
    ap.add_argument("--max_details_chars", type=int, default=1800)
    ap.add_argument("--max_packages", type=int, default=20)
    args = ap.parse_args()

    snapshots_path = Path(args.snapshots)
    out_path = Path(args.output)
    advisory_dirs = [Path(p) for p in args.advisory_dirs]

    df = pd.read_csv(snapshots_path)
    required = {"cve", "cutoff_date", "text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in snapshots CSV: {sorted(missing)}")

    df["cve"] = df["cve"].astype(str).str.upper()
    cutoff_dt = pd.to_datetime(df["cutoff_date"], errors="coerce").dt.normalize()
    if cutoff_dt.isna().any():
        bad = int(cutoff_dt.isna().sum())
        raise SystemExit(f"Could not parse {bad} cutoff_date values.")

    index = build_index(
        advisory_dirs,
        max_summary_chars=args.max_summary_chars,
        max_details_chars=args.max_details_chars,
        max_packages=args.max_packages,
    )

    enriched_texts: List[str] = []
    ids_col: List[str] = []
    published_dates_col: List[str] = []
    modified_dates_col: List[str] = []
    count_col: List[int] = []
    available_col: List[bool] = []

    rows_with_indexed_advisory = 0
    rows_enriched = 0
    skipped_unpublished = 0
    skipped_modified_after = 0
    strict_candidate_count = 0

    for cve, cutoff, base_text in zip(df["cve"], cutoff_dt, df["text"]):
        recs = index.get(cve, [])
        if recs:
            rows_with_indexed_advisory += 1

        eligible: List[AdvisoryRecord] = []
        for rec in recs:
            if rec.published is None:
                continue
            if rec.published > cutoff:
                skipped_unpublished += 1
                continue
            if not args.allow_modified_after_cutoff and rec.modified is not None and rec.modified > cutoff:
                skipped_modified_after += 1
                continue
            strict_candidate_count += 1
            eligible.append(rec)

        eligible = eligible[: args.max_records_per_cve]

        if eligible:
            rows_enriched += 1
            blocks = [render_advisory_block(rec, cutoff) for rec in eligible]
            enriched_texts.append(str(base_text) + "\n\n" + "\n\n".join(blocks))
            ids_col.append(";".join(rec.advisory_id for rec in eligible))
            published_dates_col.append(";".join(date_to_str(rec.published) for rec in eligible))
            modified_dates_col.append(";".join(date_to_str(rec.modified) for rec in eligible))
            count_col.append(len(eligible))
            available_col.append(True)
        else:
            enriched_texts.append(str(base_text))
            ids_col.append("")
            published_dates_col.append("")
            modified_dates_col.append("")
            count_col.append(0)
            available_col.append(False)

    out = df.copy()
    out["text"] = enriched_texts
    out["ghsa_osv_available_by_cutoff"] = available_col
    out["ghsa_osv_advisory_count"] = count_col
    out["ghsa_osv_advisory_ids"] = ids_col
    out["ghsa_osv_published_dates"] = published_dates_col
    out["ghsa_osv_modified_dates"] = modified_dates_col

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote GHSA/OSV-enriched snapshots: {out_path}")
    print(f"Rows: {len(out)}")
    print(f"Rows with at least one indexed GHSA/OSV advisory for CVE: {rows_with_indexed_advisory}")
    print(f"Rows enriched under cutoff rules: {rows_enriched}")
    print(f"Strict/eligible advisory-row matches considered: {strict_candidate_count}")
    print(f"Skipped advisories published after cutoff: {skipped_unpublished}")
    if not args.allow_modified_after_cutoff:
        print(f"Skipped advisories modified after cutoff: {skipped_modified_after}")
    print("Default mode is strict: published <= cutoff and modified <= cutoff.")


if __name__ == "__main__":
    main()
