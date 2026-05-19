from __future__ import annotations

"""
build_snapshots_weekly_kev_forecast.py

Academic real-time deployment simulator snapshot builder for ThreatLens.

This builder creates weekly (CVE, cutoff_date) rows that mimic a weekly
prioritization run:

    At cutoff date t, rank CVEs that were already published and not already in
    CISA KEV. A row is positive if the CVE enters KEV within the next
    horizon_days; otherwise it is a temporal weak negative.

This is different from the conservative event-anchored builder:

    conservative: positive rows are anchored at KEV date - lead_days; negatives
                  are never-KEV CVEs sampled at the same cutoff.

    weekly simulator: cutoffs are regular weekly dates; positives/negatives are
                      defined by the future window after each cutoff.

The output remains compatible with enrichment/evaluation scripts because it
writes at least:

    cve, cutoff_date, text

It also writes audit columns:

    builder_target, sampling_role, event_date, event_type,
    prediction_horizon_days, published_date

Important methodology notes:
  * KEV status is used only for labels/audit columns, never as model text.
  * The text block intentionally excludes exact CVE_ID, PUBLISHED, CUTOFF_DATE,
    lastModified, references, vulnStatus, and raw metric keys.
  * The text block keeps relative age features such as DAYS_SINCE_PUBLICATION
    and AGE_BUCKET, because those are realistic at deployment time.
  * This builder is intended to be followed by temporal enrichers such as
    enrich_snapshots_with_epss.py and enrich_snapshots_with_ghsa_osv.py.
"""

import argparse
import bisect
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import pandas as pd

DEFAULT_MAX_TEXT_CHARS = 30000

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
GHSA_RE = re.compile(r"\bGHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}\b", re.IGNORECASE)


@dataclass
class CVERecord:
    cve: str
    source_identifier: str
    published: pd.Timestamp | None
    last_modified: pd.Timestamp | None
    vuln_status: str
    description: str
    cvss_version: str
    cvss_vector: str
    cvss_base_score: Any
    cvss_base_severity: str
    cwes: List[str]
    references: List[dict]
    cpes: List[str]
    raw_metrics_available: List[str]
    cvss_metric_key_used: str


def _to_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts

def scrub_model_text_ids(text: str) -> str:
    """Replace raw CVE/GHSA identifiers in model text while keeping audit columns unchanged."""
    text = CVE_RE.sub("CVE_TOKEN", str(text))
    text = GHSA_RE.sub("GHSA_TOKEN", text)
    return text


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_description(row: dict) -> str:
    desc = row.get("descriptions", "")
    if isinstance(desc, str):
        return desc.strip()
    if isinstance(desc, list):
        for d in desc:
            if isinstance(d, dict) and str(d.get("lang", "")).lower() == "en":
                return str(d.get("value", "") or "").strip()
        for d in desc:
            if isinstance(d, dict):
                return str(d.get("value", "") or "").strip()
    return ""


def _extract_cwes(row: dict) -> List[str]:
    vals = row.get("cwes", [])
    if isinstance(vals, list):
        return sorted({str(x).strip() for x in vals if str(x).strip()})
    if isinstance(vals, str) and vals.strip():
        return [vals.strip()]

    out: List[str] = []
    for weakness in _as_list(row.get("weaknesses")):
        if isinstance(weakness, dict):
            for d in _as_list(weakness.get("description")):
                if isinstance(d, dict):
                    val = str(d.get("value", "") or "").strip()
                    if val:
                        out.append(val)
    return sorted(set(out))


def _extract_cpes(row: dict) -> List[str]:
    vals = row.get("cpes", [])
    if isinstance(vals, list) and vals:
        return sorted({str(x).strip() for x in vals if str(x).strip()})

    vals = _as_list(row.get("affected_products"))
    out: List[str] = []
    for v in vals:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
        elif isinstance(v, dict):
            candidate = v.get("criteria") or v.get("cpe") or v.get("product")
            if candidate:
                out.append(str(candidate).strip())
    return sorted(set(out))


def _extract_cvss_fields(row: dict) -> tuple[str, str, Any, str]:
    cvss_obj = row.get("cvss") or {}
    if not isinstance(cvss_obj, dict):
        cvss_obj = {}
    version = str(row.get("cvss_version") or cvss_obj.get("version") or "").strip()
    vector = str(row.get("cvss_vector") or cvss_obj.get("vectorString") or "").strip()
    score = row.get("cvss_base_score", cvss_obj.get("baseScore", None))
    severity = str(row.get("cvss_base_severity") or cvss_obj.get("baseSeverity") or "").strip()
    return version, vector, score, severity


def load_full_nvd_processed(path: Path) -> dict[str, CVERecord]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "vulnerabilities" in data:
        records = [item.get("cve", {}) for item in data.get("vulnerabilities", []) or []]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported NVD JSON format: {path}")

    out: dict[str, CVERecord] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        cve = str(row.get("id", "") or row.get("cve", "")).strip().upper()
        if not cve:
            continue
        cvss_version, cvss_vector, cvss_score, cvss_severity = _extract_cvss_fields(row)
        out[cve] = CVERecord(
            cve=cve,
            source_identifier=str(row.get("sourceIdentifier", "") or "").strip(),
            published=_to_timestamp(row.get("published")),
            last_modified=_to_timestamp(row.get("lastModified")),
            vuln_status=str(row.get("vulnStatus", "") or "").strip(),
            description=_extract_description(row),
            cvss_version=cvss_version,
            cvss_vector=cvss_vector,
            cvss_base_score=cvss_score,
            cvss_base_severity=cvss_severity,
            cwes=_extract_cwes(row),
            references=list(row.get("references", []) or []),
            cpes=_extract_cpes(row),
            raw_metrics_available=list(row.get("raw_metrics_available", []) or []),
            cvss_metric_key_used=str(row.get("cvss_metric_key_used", "") or "").strip(),
        )
    return out


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cve", "event_type", "event_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["cve"] = df["cve"].astype(str).str.upper().str.strip()
    df["event_type"] = df["event_type"].astype(str).str.strip()
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["cve", "event_date"]).reset_index(drop=True)
    return df.sort_values(["event_date", "cve", "event_type"]).reset_index(drop=True)


def age_bucket(days: int | None) -> str:
    if days is None:
        return "unknown"
    if days < 0:
        return "not_published_yet"
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


def days_since_publication(rec: CVERecord, cutoff: pd.Timestamp) -> int | None:
    if rec.published is None or pd.isna(rec.published):
        return None
    return int((cutoff.normalize() - rec.published.normalize()).days)


def render_nvd_block_cutoff_safer(
    rec: CVERecord,
    cutoff: pd.Timestamp,
    *,
    include_source_identifier: bool = False,
    include_age_context: bool = True,
    max_cpes: int = 40,
) -> str:
    """Render a cutoff-safer NVD text block for the neural text ranker.

    Excluded from model text to reduce shortcut/leakage risk:
      - CVE_ID
      - exact PUBLISHED date
      - exact CUTOFF_DATE
      - lastModified
      - references / URLs / tags
      - vulnStatus
      - raw_metrics_available
      - cvss_metric_key_used
    """
    lines: List[str] = ["NVD_RECORD:"]

    if include_source_identifier and rec.source_identifier:
        lines.append(f"SOURCE_IDENTIFIER: {rec.source_identifier}")

    if include_age_context:
        age_days = days_since_publication(rec, cutoff)
        if age_days is not None:
            lines.append(f"DAYS_SINCE_PUBLICATION: {age_days}")
        lines.append(f"AGE_BUCKET: {age_bucket(age_days)}")

    if rec.description:
        lines.append(f"DESCRIPTION: {scrub_model_text_ids(rec.description)}")

    cvss_parts: List[str] = []
    if rec.cvss_version:
        cvss_parts.append(f"version={rec.cvss_version}")
    if rec.cvss_base_score not in (None, ""):
        cvss_parts.append(f"base_score={rec.cvss_base_score}")
    if rec.cvss_base_severity:
        cvss_parts.append(f"severity={rec.cvss_base_severity}")
    if rec.cvss_vector:
        cvss_parts.append(f"vector={rec.cvss_vector}")
    if cvss_parts:
        lines.append("CVSS: " + "; ".join(cvss_parts))

    if rec.cwes:
        lines.append("CWES: " + "; ".join(str(x) for x in rec.cwes if str(x).strip()))

    if rec.cpes:
        lines.append("CPES:")
        for cpe in rec.cpes[:max_cpes]:
            lines.append(f"- {cpe}")
        if len(rec.cpes) > max_cpes:
            lines.append(f"- ... {len(rec.cpes) - max_cpes} more CPEs omitted")

    return "\n".join(lines).strip()


def build_snapshot_text(
    rec: CVERecord,
    cutoff: pd.Timestamp,
    max_text_chars: int,
    *,
    include_source_identifier: bool,
    include_age_context: bool,
    max_cpes: int,
) -> str:
    text = render_nvd_block_cutoff_safer(
        rec,
        cutoff,
        include_source_identifier=include_source_identifier,
        include_age_context=include_age_context,
        max_cpes=max_cpes,
    )
    if len(text) > max_text_chars:
        text = text[:max_text_chars]
    return text


def is_published_by_cutoff(rec: CVERecord | None, cutoff: pd.Timestamp) -> bool:
    return bool(rec is not None and rec.published is not None and not pd.isna(rec.published) and cutoff >= rec.published)


def build_first_event_map(events: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, str]]:
    out: dict[str, tuple[pd.Timestamp, str]] = {}
    for row in events.itertuples(index=False):
        prev = out.get(row.cve)
        if prev is None or row.event_date < prev[0]:
            out[row.cve] = (row.event_date, row.event_type)
    return out


def parse_optional_date(value: str | None, *, default: pd.Timestamp) -> pd.Timestamp:
    if not value:
        return default
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Could not parse date: {value}")
    return pd.Timestamp(ts)


def week_floor(ts: pd.Timestamp, week_anchor: str) -> pd.Timestamp:
    """Return the weekly anchor date on/before ts.

    week_anchor accepts pandas weekday aliases such as MON, SUN, WED.
    """
    alias = week_anchor.upper()[:3]
    weekday_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    if alias not in weekday_map:
        raise ValueError("--week_anchor must be one of MON,TUE,WED,THU,FRI,SAT,SUN")
    target = weekday_map[alias]
    delta = (ts.weekday() - target) % 7
    return (ts.normalize() - pd.Timedelta(days=delta)).tz_convert("UTC")


def make_weekly_cutoffs(
    events: pd.DataFrame,
    horizon_days: int,
    *,
    start_date: str | None,
    end_date: str | None,
    week_anchor: str,
    include_censored_tail: bool,
) -> list[pd.Timestamp]:
    if events.empty:
        return []

    min_event = pd.Timestamp(events["event_date"].min())
    max_event = pd.Timestamp(events["event_date"].max())

    default_start = week_floor(min_event - pd.Timedelta(days=horizon_days), week_anchor)
    # Avoid right-censored labels by default: a cutoff after max_event-horizon does
    # not have a complete 30-day future window in the observed KEV catalog.
    default_end_raw = max_event if include_censored_tail else max_event - pd.Timedelta(days=horizon_days)
    default_end = week_floor(default_end_raw, week_anchor)

    start = week_floor(parse_optional_date(start_date, default=default_start), week_anchor)
    end = week_floor(parse_optional_date(end_date, default=default_end), week_anchor)
    if end < start:
        raise ValueError(f"Weekly cutoff end {end.date()} is before start {start.date()}")

    freq = f"W-{week_anchor.upper()[:3]}"
    return list(pd.date_range(start=start, end=end, freq=freq, tz="UTC"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build weekly deployment-simulator KEV forecasting snapshots. Each row is a CVE at a "
            "weekly cutoff date. Label is positive if the CVE enters KEV within the next horizon_days."
        )
    )
    ap.add_argument("--repo_root", type=str, default=".")
    ap.add_argument("--events", type=str, required=True, help="KEV event CSV with cve,event_type,event_date")
    ap.add_argument("--output", type=str, default="Data/snapshots_weekly_kev_forecast.csv")
    ap.add_argument("--nvd_json", type=str, default="Data/Structured/CVE/processed/nvdcve-2.0-full.json")
    ap.add_argument("--horizon_days", type=int, default=30, help="Future prediction window after each weekly cutoff")
    ap.add_argument("--negatives_per_positive", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_text_chars", type=int, default=DEFAULT_MAX_TEXT_CHARS)
    ap.add_argument("--max_cpes", type=int, default=40)
    ap.add_argument("--week_anchor", type=str, default="MON", help="Weekly cutoff anchor: MON,TUE,WED,THU,FRI,SAT,SUN")
    ap.add_argument("--start_date", type=str, default=None, help="Optional first cutoff date, YYYY-MM-DD. Rounded down to week anchor.")
    ap.add_argument("--end_date", type=str, default=None, help="Optional last cutoff date, YYYY-MM-DD. Rounded down to week anchor.")
    ap.add_argument(
        "--include_censored_tail",
        action="store_true",
        help="Include cutoffs whose future window extends beyond the latest observed KEV date. Usually keep off for evaluation.",
    )
    ap.add_argument(
        "--keep_zero_positive_cutoffs",
        action="store_true",
        help="Keep negative-only weeks by sampling --background_negatives_per_cutoff rows. Default skips weeks with no positives.",
    )
    ap.add_argument(
        "--background_negatives_per_cutoff",
        type=int,
        default=0,
        help="If keeping zero-positive cutoffs, sample this many negatives for those weeks.",
    )
    ap.add_argument(
        "--recent_negative_window_days",
        type=int,
        default=0,
        help=(
            "Optional: restrict negative candidates to CVEs published within this many days before cutoff. "
            "0 means sample from all eligible published CVEs."
        ),
    )
    ap.add_argument("--no_age_context", action="store_true", help="Do not include DAYS_SINCE_PUBLICATION or AGE_BUCKET in model text.")
    ap.add_argument("--include_source_identifier", action="store_true")
    args = ap.parse_args()

    if args.horizon_days <= 0:
        raise ValueError("--horizon_days must be positive")
    if args.negatives_per_positive < 0:
        raise ValueError("--negatives_per_positive must be nonnegative")
    if args.background_negatives_per_cutoff < 0:
        raise ValueError("--background_negatives_per_cutoff must be nonnegative")
    if args.recent_negative_window_days < 0:
        raise ValueError("--recent_negative_window_days must be nonnegative")

    rng = random.Random(args.seed)
    repo_root = Path(args.repo_root).resolve()

    events = load_events(repo_root / args.events)
    cve_records = load_full_nvd_processed(repo_root / args.nvd_json)
    first_event = build_first_event_map(events)
    cutoffs = make_weekly_cutoffs(
        events,
        args.horizon_days,
        start_date=args.start_date,
        end_date=args.end_date,
        week_anchor=args.week_anchor,
        include_censored_tail=args.include_censored_tail,
    )

    # Pre-sort CVEs by publication date so each cutoff only considers published CVEs.
    published_items: list[tuple[pd.Timestamp, str]] = []
    skipped_missing_pub = 0
    for cve, rec in cve_records.items():
        if rec.published is None or pd.isna(rec.published):
            skipped_missing_pub += 1
            continue
        published_items.append((rec.published, cve))
    published_items.sort(key=lambda x: (x[0], x[1]))
    published_dates = [x[0] for x in published_items]
    published_cves = [x[1] for x in published_items]

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    skipped_cutoffs_no_positive = 0
    active_cutoffs = 0
    total_candidate_negatives = 0
    total_sampled_negatives = 0

    for idx, cutoff in enumerate(cutoffs, start=1):
        future_end = cutoff + pd.Timedelta(days=args.horizon_days)
        cutoff_str = cutoff.date().isoformat()

        n_pub = bisect.bisect_right(published_dates, cutoff)
        eligible_cves = published_cves[:n_pub]

        positives: list[tuple[str, str, str]] = []
        negatives: list[str] = []

        recent_start = cutoff - pd.Timedelta(days=args.recent_negative_window_days) if args.recent_negative_window_days else None

        for cve in eligible_cves:
            ev = first_event.get(cve)
            if ev is not None:
                kev_date, kev_type = ev
                if kev_date <= cutoff:
                    # Already in KEV at cutoff; not a deployment candidate.
                    continue
                if cutoff < kev_date <= future_end:
                    positives.append((cve, kev_date.date().isoformat(), kev_type))
                    continue
                # Else: KEV later than the current prediction horizon. This is
                # valid as a temporal weak negative, if sampled.

            # Non-KEV CVE or KEV-after-window CVE.
            if recent_start is not None:
                rec = cve_records.get(cve)
                if rec is None or rec.published is None or rec.published < recent_start:
                    continue
            negatives.append(cve)

        if positives:
            n_neg = int(round(len(positives) * args.negatives_per_positive))
        elif args.keep_zero_positive_cutoffs:
            n_neg = args.background_negatives_per_cutoff
        else:
            skipped_cutoffs_no_positive += 1
            continue

        active_cutoffs += 1
        total_candidate_negatives += len(negatives)
        sampled_negatives = rng.sample(negatives, k=min(n_neg, len(negatives))) if n_neg > 0 else []
        total_sampled_negatives += len(sampled_negatives)

        for cve, event_date, event_type in positives:
            key = (cve, cutoff_str)
            if key in seen:
                continue
            rec = cve_records.get(cve)
            if not is_published_by_cutoff(rec, cutoff):
                continue
            text = build_snapshot_text(
                rec,
                cutoff,
                args.max_text_chars,
                include_source_identifier=args.include_source_identifier,
                include_age_context=not args.no_age_context,
                max_cpes=args.max_cpes,
            )
            if not text:
                continue
            seen.add(key)
            rows.append(
                {
                    "cve": cve,
                    "cutoff_date": cutoff_str,
                    "text": text,
                    "builder_target": 1,
                    "sampling_role": "weekly_future_kev_positive",
                    "event_date": event_date,
                    "event_type": event_type,
                    "prediction_horizon_days": args.horizon_days,
                    "published_date": rec.published.date().isoformat() if rec and rec.published is not None else "",
                    "cutoff_mode": "weekly",
                }
            )

        for cve in sampled_negatives:
            key = (cve, cutoff_str)
            if key in seen:
                continue
            rec = cve_records.get(cve)
            if not is_published_by_cutoff(rec, cutoff):
                continue
            text = build_snapshot_text(
                rec,
                cutoff,
                args.max_text_chars,
                include_source_identifier=args.include_source_identifier,
                include_age_context=not args.no_age_context,
                max_cpes=args.max_cpes,
            )
            if not text:
                continue

            ev = first_event.get(cve)
            future_event_date = ""
            future_event_type = ""
            if ev is not None:
                # If present, this is a later-than-window KEV date, useful for auditing.
                future_event_date = ev[0].date().isoformat()
                future_event_type = ev[1]

            seen.add(key)
            rows.append(
                {
                    "cve": cve,
                    "cutoff_date": cutoff_str,
                    "text": text,
                    "builder_target": 0,
                    "sampling_role": "weekly_temporal_window_negative",
                    "event_date": future_event_date,
                    "event_type": future_event_type,
                    "prediction_horizon_days": args.horizon_days,
                    "published_date": rec.published.date().isoformat() if rec and rec.published is not None else "",
                    "cutoff_mode": "weekly",
                }
            )

        if idx == 1 or idx % 25 == 0 or idx == len(cutoffs):
            print(
                f"[{idx}/{len(cutoffs)}] cutoff={cutoff_str} published={len(eligible_cves)} "
                f"positives={len(positives)} candidate_negatives={len(negatives)} "
                f"sampled_negatives={len(sampled_negatives)}"
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No rows were created. Check event/NVD paths, date range, horizon, and sampling settings.")

    out = out.drop_duplicates(subset=["cve", "cutoff_date"]).sort_values(
        ["cutoff_date", "builder_target", "cve"], ascending=[True, False, True]
    ).reset_index(drop=True)

    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    n_pos = int((out["builder_target"] == 1).sum())
    n_neg = int((out["builder_target"] == 0).sum())
    repeated_cves = int((out["cve"].value_counts() > 1).sum())
    temporal_neg_later_positive = int(
        ((out["builder_target"] == 0) & out["event_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)).sum()
    )

    print(f"\nWrote weekly KEV forecast snapshots CSV: {out_path}")
    print(f"Rows: {len(out)}")
    print(f"Positive rows: {n_pos}")
    print(f"Negative rows: {n_neg}")
    print(f"Positive ratio: {n_pos / len(out):.4f}")
    print(f"Distinct cutoff dates: {out['cutoff_date'].nunique()}")
    print(f"Unique CVEs: {out['cve'].nunique()}")
    print(f"Repeated CVEs across cutoffs: {repeated_cves}")
    print(f"Temporal negatives that later enter KEV after the window: {temporal_neg_later_positive}")
    print(f"Cutoff min: {out['cutoff_date'].min()}")
    print(f"Cutoff max: {out['cutoff_date'].max()}")
    print("\nDiagnostics:")
    print(f"  event rows loaded: {len(events)}")
    print(f"  CVEs in NVD with usable publication date: {len(published_items)}")
    print(f"  CVEs skipped missing publication date: {skipped_missing_pub}")
    print(f"  weekly cutoff dates considered: {len(cutoffs)}")
    print(f"  active cutoff dates written: {active_cutoffs}")
    print(f"  cutoff dates skipped because no positives in horizon: {skipped_cutoffs_no_positive}")
    print(f"  total sampled negatives: {total_sampled_negatives}")
    print(
        "  average candidate negatives per active cutoff: "
        f"{total_candidate_negatives / max(1, active_cutoffs):.1f}"
    )
    if args.recent_negative_window_days:
        print(f"  recent negative candidate window: {args.recent_negative_window_days} days before cutoff")
    print("\nNext steps:")
    print("  1. Enrich with EPSS as-of cutoff.")
    print("  2. Optionally enrich with GHSA/OSV advisories published by cutoff.")
    print(f"  3. Use --window_days {args.horizon_days} in train/eval.")


if __name__ == "__main__":
    main()
