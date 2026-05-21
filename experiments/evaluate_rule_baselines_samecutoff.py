#!/usr/bin/env python3
"""
evaluate_rule_baselines_samecutoff.py

Rule/score baselines for ThreatLens weekly CVE prioritization.

This script uses the same label construction and temporal+CVE-disjoint split
logic as the neural evaluator, but does not train a neural model. It evaluates
simple ranking baselines under the same per-cutoff top-K evaluation:

  - EPSS-only ranking
  - CVSS-only ranking
  - public exploit evidence rule/count ranking
  - GHSA/OSV advisory count ranking
  - simple structured score combination
  - random ranking baseline averaged over seeds

Input snapshots should be the final enriched weekly snapshot CSV containing at
least cve, cutoff_date, text, and optionally epss_score, cvss_base_score,
public_exploit_available_by_cutoff/public_exploit_count,
ghsa_osv_available_by_cutoff/ghsa_osv_advisory_count.

Example:
  python agents/evaluate_rule_baselines_samecutoff.py \
    --snapshots Data/snapshots_weekly_kev_forecast_epss_ghsa_exploits.csv \
    --events Data/events_kev.csv \
    --window_days 30 \
    --out_json Data/rule_baselines_samecutoff.json \
    | tee results_rule_baselines_samecutoff.txt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

POSITIVE_EVENTS: Set[str] = {"KEV", "INTHEWILD", "METASPLOIT", "EXPLOITDB"}


def load_snapshots(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cve", "cutoff_date", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Snapshots CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"], utc=True)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cve", "event_type", "event_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["event_type"] = df["event_type"].astype(str).str.upper()
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True)
    return df


def build_labels(snapshots: pd.DataFrame, events: pd.DataFrame, window_days: int) -> pd.DataFrame:
    grouped: Dict[str, List[Tuple[str, pd.Timestamp]]] = {}
    for row in events.itertuples(index=False):
        grouped.setdefault(row.cve, []).append((row.event_type, row.event_date))

    rows = []
    # Preserve all snapshot columns so baseline scoring can use enrichment columns.
    for row in snapshots.itertuples(index=False):
        row_dict = row._asdict()
        cve = row_dict["cve"]
        t = row_dict["cutoff_date"]
        evs = grouped.get(cve, [])
        already_positive = any((etype in POSITIVE_EVENTS) and (edate <= t) for etype, edate in evs)
        if already_positive:
            continue
        future_end = t + pd.Timedelta(days=window_days)
        future_positive = any((etype in POSITIVE_EVENTS) and (t < edate <= future_end) for etype, edate in evs)
        row_dict["target"] = 1 if future_positive else 0
        rows.append(row_dict)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No labeled rows remained after filtering already-positive examples.")
    return out.sort_values(["cutoff_date", "cve"]).reset_index(drop=True)


def temporal_disjoint_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    unique_cutoffs = sorted(df["cutoff_date"].dropna().unique())
    if len(unique_cutoffs) < 3:
        raise ValueError("Need at least 3 cutoff dates.")
    n = len(unique_cutoffs)
    train_end = max(1, int(round(train_frac * n)))
    val_end = max(train_end + 1, int(round((train_frac + val_frac) * n)))
    val_end = min(val_end, n - 1)

    train_cutoffs = set(unique_cutoffs[:train_end])
    val_cutoffs = set(unique_cutoffs[train_end:val_end])
    test_cutoffs = set(unique_cutoffs[val_end:])

    train_df = df[df["cutoff_date"].isin(train_cutoffs)].copy()
    val_df = df[df["cutoff_date"].isin(val_cutoffs)].copy()
    test_df = df[df["cutoff_date"].isin(test_cutoffs)].copy()

    train_cves = set(train_df["cve"].unique())
    val_df = val_df[~val_df["cve"].isin(train_cves)].copy()
    train_val_cves = train_cves | set(val_df["cve"].unique())
    test_df = test_df[~test_df["cve"].isin(train_val_cves)].copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    total_pos = int(labels.sum())
    if total_pos == 0:
        return 0.0
    tp = 0
    fp = 0
    precisions = [1.0]
    recalls = [0.0]
    for y in labels_sorted:
        if y == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_pos)
    auc = 0.0
    for i in range(1, len(precisions)):
        auc += (recalls[i] - recalls[i - 1]) * precisions[i]
    return float(auc)


def sample_same_cutoff_pairs(rng: np.random.Generator, df: pd.DataFrame, num_pairs: int):
    labels = df["target"].to_numpy(dtype=int)
    cutoffs = df["cutoff_date"].tolist()
    neg_by_cutoff: Dict[pd.Timestamp, np.ndarray] = {}
    for cutoff, group in df.groupby("cutoff_date", sort=False):
        neg_idx = group.index[group["target"].to_numpy(dtype=int) == 0].to_numpy(dtype=int)
        if len(neg_idx) > 0:
            neg_by_cutoff[pd.Timestamp(cutoff)] = neg_idx

    eligible_pos = []
    for i, y in enumerate(labels):
        if y == 1 and pd.Timestamp(cutoffs[i]) in neg_by_cutoff:
            eligible_pos.append(i)
    if not eligible_pos:
        return np.array([], dtype=int), np.array([], dtype=int)

    ia = rng.choice(np.array(eligible_pos, dtype=int), size=num_pairs, replace=True)
    ib = np.empty(num_pairs, dtype=int)
    for j, pos_idx in enumerate(ia):
        cutoff = pd.Timestamp(cutoffs[int(pos_idx)])
        ib[j] = rng.choice(neg_by_cutoff[cutoff])
    return ia, ib


def same_cutoff_pair_accuracy(scores: np.ndarray, df: pd.DataFrame, pairs: int = 5000, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    ia, ib = sample_same_cutoff_pairs(rng, df.reset_index(drop=True), min(pairs, max(100, len(df))))
    if len(ia) == 0:
        return 0.0
    return float(np.mean(scores[ia] > scores[ib]))


def cutoff_ranking_metrics(scores: np.ndarray, df: pd.DataFrame, k_values: Tuple[int, ...]) -> Dict[str, float | int]:
    work = df[["cutoff_date", "target"]].copy().reset_index(drop=True)
    work["score"] = scores
    out: Dict[str, float | int] = {}
    valid_cutoffs = 0
    cutoffs_with_positive = 0

    macro_p = {k: [] for k in k_values}
    macro_r = {k: [] for k in k_values}
    micro_hits = {k: 0 for k in k_values}
    micro_slots = {k: 0 for k in k_values}
    micro_pos = {k: 0 for k in k_values}
    pr_aucs = []

    for _, g in work.groupby("cutoff_date", sort=True):
        labels = g["target"].to_numpy(dtype=int)
        sc = g["score"].to_numpy(dtype=float)
        n_pos = int(labels.sum())
        n_neg = int((labels == 0).sum())
        if len(labels) == 0:
            continue
        valid_cutoffs += 1
        if n_pos > 0:
            cutoffs_with_positive += 1
        if n_pos > 0 and n_neg > 0:
            pr_aucs.append(pr_auc(sc, labels))
        order = np.argsort(-sc)
        for k in k_values:
            kk = min(k, len(labels))
            if kk <= 0:
                continue
            hits = int(labels[order[:kk]].sum())
            macro_p[k].append(hits / kk)
            macro_r[k].append(hits / n_pos if n_pos else 0.0)
            micro_hits[k] += hits
            micro_slots[k] += kk
            micro_pos[k] += n_pos

    out["cutoffs"] = valid_cutoffs
    out["cutoffs_with_positive"] = cutoffs_with_positive
    for k in k_values:
        out[f"precision_at_{k}"] = float(np.mean(macro_p[k])) if macro_p[k] else 0.0
        out[f"recall_at_{k}"] = float(np.mean(macro_r[k])) if macro_r[k] else 0.0
        out[f"micro_precision_at_{k}"] = float(micro_hits[k] / micro_slots[k]) if micro_slots[k] else 0.0
        out[f"micro_recall_at_{k}"] = float(micro_hits[k] / micro_pos[k]) if micro_pos[k] else 0.0
    out["macro_pr_auc_by_cutoff"] = float(np.mean(pr_aucs)) if pr_aucs else 0.0
    return out


def evaluate_scores(name: str, scores: np.ndarray, df: pd.DataFrame, k_values: Tuple[int, ...], pair_seed: int = 7):
    labels = df["target"].to_numpy(dtype=int)
    metrics: Dict[str, float | int] = {
        "rows": int(len(df)),
        "unique_cves": int(df["cve"].nunique()),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "pair_acc": same_cutoff_pair_accuracy(scores, df, pairs=5000, seed=pair_seed),
        "global_pr_auc": pr_auc(scores, labels),
    }
    metrics.update(cutoff_ranking_metrics(scores, df, k_values))
    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"{k:>24}: {v}")
    return metrics


CVSS_RE = re.compile(r"base_score=([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def numeric_col(df: pd.DataFrame, candidates: List[str], default: float = 0.0) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def bool_col(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if s.dtype == bool:
                return s.astype(float)
            return s.astype(str).str.lower().isin({"true", "1", "yes", "y"}).astype(float)
    return pd.Series(0.0, index=df.index, dtype=float)


def cvss_from_text(df: pd.DataFrame) -> pd.Series:
    vals = []
    for text in df["text"].fillna("").astype(str):
        m = CVSS_RE.search(text)
        vals.append(float(m.group(1)) if m else 0.0)
    return pd.Series(vals, index=df.index, dtype=float)


def make_scores(df: pd.DataFrame, baseline: str, seed: int = 7) -> np.ndarray:
    baseline = baseline.lower()
    if baseline == "epss":
        return numeric_col(df, ["epss_score", "epss"]).to_numpy(dtype=float)
    if baseline == "epss_percentile":
        return numeric_col(df, ["epss_percentile"]).to_numpy(dtype=float)
    if baseline == "cvss":
        s = numeric_col(df, ["cvss_base_score", "base_score"], default=np.nan)
        if s.isna().all() or (s.fillna(0.0).sum() == 0.0):
            s = cvss_from_text(df)
        return s.fillna(0.0).to_numpy(dtype=float)
    if baseline == "public_exploit_flag":
        return bool_col(df, ["public_exploit_available_by_cutoff"]).to_numpy(dtype=float)
    if baseline == "public_exploit_count":
        count = numeric_col(df, ["public_exploit_count"])
        flag = bool_col(df, ["public_exploit_available_by_cutoff"])
        return (count + flag * 0.1).to_numpy(dtype=float)
    if baseline == "ghsa_count":
        count = numeric_col(df, ["ghsa_osv_advisory_count"])
        flag = bool_col(df, ["ghsa_osv_available_by_cutoff"])
        return (count + flag * 0.1).to_numpy(dtype=float)
    if baseline == "structured_combo":
        # Simple no-training score: normalize main structured signals and sum.
        epss = numeric_col(df, ["epss_score", "epss"]).clip(lower=0)
        cvss = make_scores(df, "cvss", seed=seed) / 10.0
        exploit = (numeric_col(df, ["public_exploit_count"]).clip(lower=0) > 0).astype(float)
        ghsa = (numeric_col(df, ["ghsa_osv_advisory_count"]).clip(lower=0) > 0).astype(float)
        score = epss.to_numpy(dtype=float) + 0.5 * cvss + 0.75 * exploit.to_numpy(dtype=float) + 0.25 * ghsa.to_numpy(dtype=float)
        return score.astype(float)
    if baseline == "random":
        rng = np.random.default_rng(seed)
        return rng.random(len(df))
    raise ValueError(f"Unknown baseline: {baseline}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate simple same-cutoff ranking baselines for ThreatLens.")
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--window_days", type=int, default=30)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--baselines", default="epss,cvss,public_exploit_flag,public_exploit_count,ghsa_count,structured_combo,random")
    ap.add_argument("--k_values", default="10,20,50,100")
    ap.add_argument("--random_seeds", default="1,2,3,4,5", help="Seeds to average for random baseline only")
    ap.add_argument("--out_json", default="Data/rule_baselines_samecutoff.json")
    args = ap.parse_args()

    k_values = tuple(int(x) for x in args.k_values.split(",") if x.strip())
    baselines = [x.strip().lower() for x in args.baselines.split(",") if x.strip()]
    random_seeds = [int(x) for x in args.random_seeds.split(",") if x.strip()]

    snapshots = load_snapshots(Path(args.snapshots))
    events = load_events(Path(args.events))
    labeled = build_labels(snapshots, events, args.window_days)
    train_df, val_df, test_df = temporal_disjoint_split(labeled, args.train_frac, args.val_frac)

    print("Split summary")
    print(f"train rows={len(train_df)} unique_cves={train_df['cve'].nunique()} positives={int(train_df['target'].sum())}")
    print(f"val   rows={len(val_df)} unique_cves={val_df['cve'].nunique()} positives={int(val_df['target'].sum())}")
    print(f"test  rows={len(test_df)} unique_cves={test_df['cve'].nunique()} positives={int(test_df['target'].sum())}")
    print(f"Available columns: {', '.join(labeled.columns)}")

    results: Dict[str, object] = {
        "window_days": args.window_days,
        "split": {
            "train_rows": len(train_df),
            "train_unique_cves": int(train_df["cve"].nunique()),
            "train_positives": int(train_df["target"].sum()),
            "val_rows": len(val_df),
            "val_unique_cves": int(val_df["cve"].nunique()),
            "val_positives": int(val_df["target"].sum()),
            "test_rows": len(test_df),
            "test_unique_cves": int(test_df["cve"].nunique()),
            "test_positives": int(test_df["target"].sum()),
        },
        "baselines": {},
    }

    for b in baselines:
        if b == "random":
            seed_metrics = []
            for seed in random_seeds:
                scores = make_scores(test_df, b, seed=seed)
                seed_metrics.append(evaluate_scores(f"TEST baseline=random seed={seed}", scores, test_df, k_values, pair_seed=seed))
            avg = {}
            for key in seed_metrics[0].keys():
                vals = [m[key] for m in seed_metrics if isinstance(m[key], (int, float))]
                avg[key] = float(np.mean(vals)) if vals else seed_metrics[0][key]
            results["baselines"]["random_avg"] = {"test": avg, "seeds": random_seeds}
            print("\nTEST baseline=random_avg")
            for k, v in avg.items():
                print(f"{k:>24}: {v}")
            continue

        val_scores = make_scores(val_df, b)
        test_scores = make_scores(test_df, b)
        val_metrics = evaluate_scores(f"VAL baseline={b}", val_scores, val_df, k_values)
        test_metrics = evaluate_scores(f"TEST baseline={b}", test_scores, test_df, k_values)
        results["baselines"][b] = {"val": val_metrics, "test": test_metrics}

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote baseline results JSON: {out_path}")
    print("\nJSON metrics")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
