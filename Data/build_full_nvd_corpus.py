# build_full_nvd_corpus.py
"""
Build one processed full NVD corpus from a folder of NVD 2.0 yearly zip files.

What it does
------------
- Scans a directory for files like:
    nvdcve-2.0-2007.json.zip
    ...
    nvdcve-2.0-2026.json.zip
    nvdcve-2.0-recent.json.zip
    nvdcve-2.0-modified.json.zip
- Reads the JSON directly from each zip file
- Extracts a normalized per-CVE record
- Deduplicates by CVE ID, keeping the record with the latest lastModified
- Writes:
    1) a JSON corpus
    2) optionally a CSV summary

Output schema (JSON)
--------------------
Each output record includes:
- id
- sourceIdentifier
- published
- lastModified
- vulnStatus
- descriptions
- cvss_version
- cvss_vector
- cvss_base_score
- cvss_base_severity
- cwes
- references
- cpes
- raw_metrics_available

This is designed to be much broader than nvdcve-2.0-recent.json and can later
serve as the structured base for a canonical CVE corpus.

Usage example
-------------
python build_full_nvd_corpus.py \
  --input_dir Data/Structured/CVE/raw/nvd \
  --output_json Data/Structured/CVE/processed/nvdcve-2.0-full.json \
  --output_csv Data/Structured/CVE/processed/nvdcve-2.0-full.csv
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def english_description(descs: list[dict]) -> str:
    for d in descs or []:
        if str(d.get("lang", "")).lower() == "en":
            return str(d.get("value", "")).strip()
    if descs:
        return str(descs[0].get("value", "")).strip()
    return ""


def pick_best_cvss(metrics: dict) -> Tuple[str | None, dict | None]:
    """
    Prefer newer / primary NVD metrics when available.
    Priority:
      cvssMetricV40 (Primary first)
      cvssMetricV31 (Primary first)
      cvssMetricV30 (Primary first)
      cvssMetricV2  (Primary first)
    """
    metric_order = ["cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]

    for key in metric_order:
        entries = metrics.get(key, []) or []
        if not entries:
            continue

        primaries = [e for e in entries if str(e.get("type", "")).lower() == "primary"]
        chosen_pool = primaries if primaries else entries
        if chosen_pool:
            return key, chosen_pool[0]

    return None, None


def extract_cwes(weaknesses: list[dict]) -> list[str]:
    out = set()
    for w in weaknesses or []:
        for d in w.get("description", []) or []:
            val = str(d.get("value", "")).strip()
            if val and val.upper() != "NVD-CWE-NOINFO" and val.upper() != "NVD-CWE-OTHER":
                out.add(val)
    return sorted(out)


def extract_references(refs: list[dict]) -> list[dict]:
    out = []
    for r in refs or []:
        out.append(
            {
                "url": str(r.get("url", "")).strip(),
                "source": str(r.get("source", "")).strip(),
                "tags": list(r.get("tags", []) or []),
            }
        )
    return out


def extract_cpes(configurations: list[dict]) -> list[str]:
    """
    Pulls out CPE criteria strings from nested configuration nodes.
    This keeps the parser simple but still useful.
    """
    cpes = set()

    def walk_node(node: dict) -> None:
        for match in node.get("cpeMatch", []) or []:
            crit = str(match.get("criteria", "")).strip()
            if crit:
                cpes.add(crit)
        for child in node.get("children", []) or []:
            walk_node(child)

    for cfg in configurations or []:
        for node in cfg.get("nodes", []) or []:
            walk_node(node)

    return sorted(cpes)


def normalize_record(cve_obj: dict) -> dict:
    descs = cve_obj.get("descriptions", []) or []
    metrics = cve_obj.get("metrics", {}) or {}
    weaknesses = cve_obj.get("weaknesses", []) or []
    references = cve_obj.get("references", []) or []
    configurations = cve_obj.get("configurations", []) or []

    cvss_key, cvss_entry = pick_best_cvss(metrics)

    cvss_version = None
    cvss_vector = None
    cvss_base_score = None
    cvss_base_severity = None

    if cvss_entry is not None:
        cvss_data = cvss_entry.get("cvssData", {}) or {}
        cvss_version = str(cvss_data.get("version", "")).strip() or None
        cvss_vector = str(cvss_data.get("vectorString", "")).strip() or None
        cvss_base_score = cvss_data.get("baseScore", None)
        cvss_base_severity = str(cvss_data.get("baseSeverity", "")).strip() or None
        if not cvss_base_severity and "baseSeverity" in cvss_entry:
            cvss_base_severity = str(cvss_entry.get("baseSeverity", "")).strip() or None

    return {
        "id": str(cve_obj.get("id", "")).strip(),
        "sourceIdentifier": str(cve_obj.get("sourceIdentifier", "")).strip(),
        "published": str(cve_obj.get("published", "")).strip(),
        "lastModified": str(cve_obj.get("lastModified", "")).strip(),
        "vulnStatus": str(cve_obj.get("vulnStatus", "")).strip(),
        "descriptions": english_description(descs),
        "cvss_version": cvss_version,
        "cvss_vector": cvss_vector,
        "cvss_base_score": cvss_base_score,
        "cvss_base_severity": cvss_base_severity,
        "cwes": extract_cwes(weaknesses),
        "references": extract_references(references),
        "cpes": extract_cpes(configurations),
        "raw_metrics_available": sorted(list(metrics.keys())),
        "cvss_metric_key_used": cvss_key,
    }


def iter_zip_json_records(zip_path: Path) -> List[dict]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_members = [n for n in zf.namelist() if n.lower().endswith(".json")]
        if not json_members:
            raise ValueError(f"No JSON file found inside {zip_path}")
        if len(json_members) > 1:
            # still pick the first one, but this is unusual
            member = json_members[0]
        else:
            member = json_members[0]

        data = json.loads(zf.read(member))
        vulns = data.get("vulnerabilities", []) or []
        records = []
        for item in vulns:
            cve_obj = item.get("cve", {})
            if cve_obj:
                records.append(cve_obj)
        return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Build one processed full NVD corpus from yearly .json.zip files.")
    ap.add_argument("--input_dir", required=True, help="Directory containing nvdcve-2.0-*.json.zip files")
    ap.add_argument("--output_json", required=True, help="Path to output processed JSON corpus")
    ap.add_argument("--output_csv", default=None, help="Optional path to output CSV summary")
    ap.add_argument("--include_recent", action="store_true", help="Include files whose names contain 'recent'")
    ap.add_argument("--include_modified", action="store_true", help="Include files whose names contain 'modified'")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    zip_files = sorted(input_dir.glob("nvdcve-2.0-*.json.zip"))
    if not zip_files:
        raise ValueError(f"No NVD .json.zip files found under {input_dir}")

    selected = []
    for p in zip_files:
        name = p.name.lower()
        if "recent" in name and not args.include_recent:
            continue
        if "modified" in name and not args.include_modified:
            continue
        selected.append(p)

    if not selected:
        raise ValueError("No NVD files selected after applying include_recent/include_modified rules.")

    print("Selected NVD files:")
    for p in selected:
        print(" -", p.name)

    by_cve: Dict[str, dict] = {}
    source_counts: Dict[str, int] = {}

    for zp in selected:
        print(f"\nReading {zp.name} ...")
        raw_records = iter_zip_json_records(zp)
        source_counts[zp.name] = len(raw_records)
        print(f"  raw CVE entries: {len(raw_records)}")

        for cve_obj in raw_records:
            rec = normalize_record(cve_obj)
            cve_id = rec["id"]
            if not cve_id:
                continue

            prev = by_cve.get(cve_id)
            if prev is None:
                by_cve[cve_id] = rec
            else:
                prev_ts = pd.to_datetime(prev.get("lastModified"), errors="coerce", utc=True)
                new_ts = pd.to_datetime(rec.get("lastModified"), errors="coerce", utc=True)

                # Keep the latest lastModified
                if pd.isna(prev_ts) and not pd.isna(new_ts):
                    by_cve[cve_id] = rec
                elif not pd.isna(new_ts) and not pd.isna(prev_ts) and new_ts > prev_ts:
                    by_cve[cve_id] = rec

    records = sorted(by_cve.values(), key=lambda r: r["id"])

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote processed JSON corpus to: {output_json}")
    print(f"Unique CVEs: {len(records)}")

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in records:
            rows.append(
                {
                    "id": r["id"],
                    "published": r["published"],
                    "lastModified": r["lastModified"],
                    "vulnStatus": r["vulnStatus"],
                    "descriptions": r["descriptions"],
                    "cvss_version": r["cvss_version"],
                    "cvss_base_score": r["cvss_base_score"],
                    "cvss_base_severity": r["cvss_base_severity"],
                    "cwe_count": len(r["cwes"]),
                    "reference_count": len(r["references"]),
                    "cpe_count": len(r["cpes"]),
                }
            )

        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"Wrote CSV summary to: {output_csv}")

    print("\nPer-file raw counts:")
    for k, v in source_counts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
