#!/usr/bin/env python3
"""
evaluate_structured_logreg_ablations.py

Structured-logistic ablations for ThreatLens same-cutoff weekly CVE prioritization.

Purpose:
  Evaluate the structured logistic scorer under targeted feature-removal ablations:
    - full:    all structured features
    - no_epss: remove EPSS-derived structured features and EPSS interactions
    - no_cvss: remove CVSS-derived structured features and CVSS interactions

This script uses the ORIGINAL enriched snapshot CSV. It does not require separate
neural/text ablation datasets, because structured logistic consumes explicit
structured columns rather than the model-facing evidence text.

Example:
  python -u agents/evaluate_structured_logreg_ablations.py \
    --snapshots Data/snapshots_weekly_kev_forecast_epss_ghsa_exploits.csv \
    --events Data/events_kev.csv \
    --window_days 30 \
    --variants full,no_epss,no_cvss \
    --out_dir Data/structured_logreg_ablations \
    2>&1 | tee structured_logreg_ablations_run.txt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


POSITIVE_EVENTS: Set[str] = {"KEV", "INTHEWILD", "METASPLOIT", "EXPLOITDB"}
CVSS_RE = re.compile(r"base_score=([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

# Feature groups used for targeted structured-logistic ablations.
EPSS_FEATURES: List[str] = [
    "epss_found",
    "epss_score",
    "epss_percentile",
    "days_since_epss_asof",
    "epss_x_public_exploit",
    "epss_x_advisory",
    "cvss_x_epss",  # interaction contains EPSS
]

CVSS_FEATURES: List[str] = [
    "cvss_base_score",
    "cvss_base_score_norm",
    "cvss_x_epss",  # interaction contains CVSS
]

VARIANT_LABELS: Dict[str, str] = {
    "full": "Structured logistic",
    "no_epss": "No EPSS",
    "no_cvss": "No CVSS",
    "no_epss_no_cvss": "No EPSS/CVSS",
}


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
    """Construct future-window labels while preserving all snapshot columns."""
    grouped: Dict[str, List[Tuple[str, pd.Timestamp]]] = {}
    for row in events.itertuples(index=False):
        grouped.setdefault(row.cve, []).append((row.event_type, row.event_date))

    rows = []
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

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Empty split after temporal + CVE-disjoint filtering.")

    return train_df, val_df, test_df


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


def days_between(later: pd.Series, earlier: pd.Series, default: float = 0.0) -> pd.Series:
    later_dt = pd.to_datetime(later, utc=True, errors="coerce")
    earlier_dt = pd.to_datetime(earlier, utc=True, errors="coerce")
    days = (later_dt - earlier_dt).dt.total_seconds() / 86400.0
    return days.replace([np.inf, -np.inf], np.nan).fillna(default).clip(lower=0.0)


def add_structured_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build the same compact, cutoff-valid tabular features as the learned baseline script."""
    feat = pd.DataFrame(index=df.index)

    feat["epss_found"] = bool_col(df, ["epss_found"])
    feat["epss_score"] = numeric_col(df, ["epss_score", "epss"], default=0.0)
    feat["epss_percentile"] = numeric_col(df, ["epss_percentile"], default=0.0)

    cvss_direct = numeric_col(df, ["cvss_base_score", "base_score"], default=np.nan)
    if cvss_direct.isna().all() or float(cvss_direct.fillna(0.0).sum()) == 0.0:
        feat["cvss_base_score"] = cvss_from_text(df)
    else:
        feat["cvss_base_score"] = cvss_direct.fillna(0.0)
    feat["cvss_base_score_norm"] = feat["cvss_base_score"] / 10.0

    feat["ghsa_osv_available_by_cutoff"] = bool_col(df, ["ghsa_osv_available_by_cutoff"])
    feat["ghsa_osv_advisory_count"] = numeric_col(df, ["ghsa_osv_advisory_count"], default=0.0).clip(lower=0.0)
    feat["log_ghsa_osv_advisory_count"] = np.log1p(feat["ghsa_osv_advisory_count"])

    feat["public_exploit_available_by_cutoff"] = bool_col(df, ["public_exploit_available_by_cutoff"])
    feat["public_exploit_count"] = numeric_col(df, ["public_exploit_count"], default=0.0).clip(lower=0.0)
    feat["log_public_exploit_count"] = np.log1p(feat["public_exploit_count"])
    feat["public_exploit_days_since_first_seen"] = numeric_col(
        df, ["public_exploit_days_since_first_seen"], default=0.0
    ).clip(lower=0.0)
    feat["log_public_exploit_days_since_first_seen"] = np.log1p(feat["public_exploit_days_since_first_seen"])

    if "published_date" in df.columns:
        feat["days_since_publication"] = days_between(df["cutoff_date"], df["published_date"], default=0.0)
        feat["log_days_since_publication"] = np.log1p(feat["days_since_publication"])
    else:
        feat["days_since_publication"] = 0.0
        feat["log_days_since_publication"] = 0.0

    if "epss_asof_date" in df.columns:
        feat["days_since_epss_asof"] = days_between(df["cutoff_date"], df["epss_asof_date"], default=0.0)
    else:
        feat["days_since_epss_asof"] = 0.0

    feat["epss_x_public_exploit"] = feat["epss_score"] * feat["public_exploit_available_by_cutoff"]
    feat["epss_x_advisory"] = feat["epss_score"] * feat["ghsa_osv_available_by_cutoff"]
    feat["cvss_x_epss"] = feat["cvss_base_score_norm"] * feat["epss_score"]

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feature_cols = list(feat.columns)
    return feat, feature_cols


def select_features(all_features: List[str], variant: str) -> List[str]:
    variant = variant.lower().strip()
    drop: Set[str] = set()
    if variant == "full":
        drop = set()
    elif variant == "no_epss":
        drop = set(EPSS_FEATURES)
    elif variant == "no_cvss":
        drop = set(CVSS_FEATURES)
    elif variant == "no_epss_no_cvss":
        drop = set(EPSS_FEATURES) | set(CVSS_FEATURES)
    else:
        raise ValueError(f"Unknown variant: {variant}. Valid variants: {sorted(VARIANT_LABELS)}")
    return [c for c in all_features if c not in drop]


def train_structured_logreg(train_df: pd.DataFrame, feature_cols: List[str]):
    X_train, _ = add_structured_features(train_df)
    y_train = train_df["target"].astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])
    model.fit(X_train[feature_cols], y_train)
    return model


def score_structured(model, feature_cols: List[str], df: pd.DataFrame) -> np.ndarray:
    X, _ = add_structured_features(df)
    return model.predict_proba(X[feature_cols])[:, 1].astype(float)


def metric_percent(x: float) -> float:
    return round(100.0 * float(x), 1)


def make_table(results: Dict[str, object], variants: List[str]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        test = results["variants"][variant]["test"]  # type: ignore[index]
        rows.append({
            "variant": variant,
            "method": VARIANT_LABELS.get(variant, variant),
            "r20": metric_percent(test.get("micro_recall_at_20", 0.0)),
            "r50": metric_percent(test.get("micro_recall_at_50", 0.0)),
            "pr_auc": metric_percent(test.get("global_pr_auc", 0.0)),
            "mpr_auc": metric_percent(test.get("macro_pr_auc_by_cutoff", 0.0)),
            "pair_acc": metric_percent(test.get("pair_acc", 0.0)),
            "features_used": len(results["variants"][variant]["features_used"]),  # type: ignore[index]
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate structured-logistic ablations for ThreatLens.")
    ap.add_argument("--snapshots", required=True, help="Original final enriched snapshot CSV.")
    ap.add_argument("--events", required=True, help="Events CSV, usually Data/events_kev.csv.")
    ap.add_argument("--window_days", type=int, default=30)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--variants", default="full,no_epss,no_cvss",
                    help="Comma-separated: full,no_epss,no_cvss,no_epss_no_cvss")
    ap.add_argument("--k_values", default="10,20,50,100")
    ap.add_argument("--out_dir", default="Data/structured_logreg_ablations")
    args = ap.parse_args()

    k_values = tuple(int(x) for x in args.k_values.split(",") if x.strip())
    variants = [x.strip().lower() for x in args.variants.split(",") if x.strip()]
    for v in variants:
        if v not in VARIANT_LABELS:
            raise ValueError(f"Unknown variant '{v}'. Valid variants: {sorted(VARIANT_LABELS)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data")
    print(f"snapshots={args.snapshots}")
    print(f"events={args.events}")
    snapshots = load_snapshots(Path(args.snapshots))
    events = load_events(Path(args.events))
    print(f"Snapshot rows: {len(snapshots)} unique CVEs={snapshots['cve'].nunique()} cutoffs={snapshots['cutoff_date'].nunique()}")
    print(f"Event types present: {sorted(events['event_type'].dropna().unique().tolist())}")

    labeled = build_labels(snapshots, events, args.window_days)
    train_df, val_df, test_df = temporal_disjoint_split(labeled, args.train_frac, args.val_frac)

    print("\nSplit summary")
    print(f"labeled rows={len(labeled)} unique_cves={labeled['cve'].nunique()} positives={int(labeled['target'].sum())}")
    print(f"train rows={len(train_df)} unique_cves={train_df['cve'].nunique()} positives={int(train_df['target'].sum())}")
    print(f"val   rows={len(val_df)} unique_cves={val_df['cve'].nunique()} positives={int(val_df['target'].sum())}")
    print(f"test  rows={len(test_df)} unique_cves={test_df['cve'].nunique()} positives={int(test_df['target'].sum())}")

    _, all_features = add_structured_features(train_df)
    print("\nFull structured feature set:")
    for col in all_features:
        print(f"  - {col}")

    results: Dict[str, object] = {
        "window_days": args.window_days,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "split": {
            "train_rows": int(len(train_df)),
            "train_unique_cves": int(train_df["cve"].nunique()),
            "train_positives": int(train_df["target"].sum()),
            "val_rows": int(len(val_df)),
            "val_unique_cves": int(val_df["cve"].nunique()),
            "val_positives": int(val_df["target"].sum()),
            "test_rows": int(len(test_df)),
            "test_unique_cves": int(test_df["cve"].nunique()),
            "test_positives": int(test_df["target"].sum()),
        },
        "feature_groups": {
            "epss_features_removed_in_no_epss": EPSS_FEATURES,
            "cvss_features_removed_in_no_cvss": CVSS_FEATURES,
        },
        "variants": {},
    }

    score_out = pd.concat([
        val_df[["cve", "cutoff_date", "target"]].assign(split="val"),
        test_df[["cve", "cutoff_date", "target"]].assign(split="test"),
    ], axis=0, ignore_index=True)

    for variant in variants:
        feature_cols = select_features(all_features, variant)
        removed = sorted(set(all_features) - set(feature_cols))
        print(f"\n=== Variant: {variant} ({VARIANT_LABELS[variant]}) ===")
        print(f"Features used ({len(feature_cols)}): {feature_cols}")
        print(f"Features removed ({len(removed)}): {removed}")

        model = train_structured_logreg(train_df, feature_cols)
        val_scores = score_structured(model, feature_cols, val_df)
        test_scores = score_structured(model, feature_cols, test_df)

        val_metrics = evaluate_scores(f"VAL variant={variant}", val_scores, val_df, k_values)
        test_metrics = evaluate_scores(f"TEST variant={variant}", test_scores, test_df, k_values)

        results["variants"][variant] = {  # type: ignore[index]
            "method": VARIANT_LABELS[variant],
            "model": "LogisticRegression(class_weight='balanced')",
            "features_used": feature_cols,
            "features_removed": removed,
            "val": val_metrics,
            "test": test_metrics,
        }

        combined_scores = np.concatenate([val_scores, test_scores])
        score_out[f"{variant}_score"] = combined_scores

    table = make_table(results, variants)

    out_json = out_dir / "structured_logreg_ablations_summary.json"
    out_scores = out_dir / "structured_logreg_ablations_scores.csv"
    out_table = out_dir / "structured_logreg_ablations_table.csv"

    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    score_out.to_csv(out_scores, index=False)
    table.to_csv(out_table, index=False)

    print("\nWrote outputs:")
    print(out_json)
    print(out_scores)
    print(out_table)

    print("\nCompact paper table source (percentages):")
    print(table.to_string(index=False))

    print("\nJSON metrics")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
