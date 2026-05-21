#!/usr/bin/env python3
"""
evaluate_learned_baselines_samecutoff.py

Learned same-cutoff baselines for ThreatLens weekly CVE prioritization.

This script is intended to sit beside:
  - agents/evaluate_rule_baselines_samecutoff.py
  - agents/evaluate_neural_prioritizer_transformer_samecutoff.py

It uses the same core setup as the active ThreatLens evaluators:
  - build labels from events using a future window
  - filter snapshots whose CVE already had a positive event by the cutoff
  - split forward in time
  - enforce CVE-disjoint validation/test splits
  - evaluate by ranking within each weekly cutoff

Baselines:
  1. structured_hgb
     Learned tabular baseline over structured/cutoff-valid evidence columns.
     Uses HistGradientBoostingClassifier. Predicted probabilities are ranking scores.

  2. structured_logreg
     Simpler learned tabular baseline over the same structured features.
     Uses LogisticRegression. Optional, but useful as a sanity check.

  3. tfidf_logreg
     Shallow text baseline over the same model-facing evidence text used by ThreatLens.
     Uses TF-IDF + class-weighted LogisticRegression. Predicted probabilities are ranking scores.

Example:
  python agents/evaluate_learned_baselines_samecutoff.py \
    --snapshots Data/snapshots_weekly_kev_forecast_epss_ghsa_exploits.csv \
    --events Data/events_kev.csv \
    --window_days 30 \
    --baselines structured_hgb,tfidf_logreg \
    --out_json Data/learned_baselines_samecutoff.json \
    --out_scores Data/learned_baselines_samecutoff_scores.csv
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


POSITIVE_EVENTS: Set[str] = {"KEV", "INTHEWILD", "METASPLOIT", "EXPLOITDB"}
CVSS_RE = re.compile(r"base_score=([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


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
    """Build a compact, cutoff-valid tabular feature matrix from active enriched columns.

    This function intentionally does not use raw identifiers, exact event dates, or the full text.
    It only uses numeric/boolean evidence signals and safe derived age features.
    CVSS is included only as a numeric value extracted from a cvss_base_score/base_score column
    or from the model-facing text pattern 'base_score=...'.
    """
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

    # Cutoff-valid age of the CVE at the review cutoff.
    if "published_date" in df.columns:
        feat["days_since_publication"] = days_between(df["cutoff_date"], df["published_date"], default=0.0)
        feat["log_days_since_publication"] = np.log1p(feat["days_since_publication"])
    else:
        feat["days_since_publication"] = 0.0
        feat["log_days_since_publication"] = 0.0

    # Age since the EPSS value's as-of date, usually small but harmless/safe if present.
    if "epss_asof_date" in df.columns:
        feat["days_since_epss_asof"] = days_between(df["cutoff_date"], df["epss_asof_date"], default=0.0)
    else:
        feat["days_since_epss_asof"] = 0.0

    # Simple interaction terms: these let a shallow tabular model combine obvious signals.
    feat["epss_x_public_exploit"] = feat["epss_score"] * feat["public_exploit_available_by_cutoff"]
    feat["epss_x_advisory"] = feat["epss_score"] * feat["ghsa_osv_available_by_cutoff"]
    feat["cvss_x_epss"] = feat["cvss_base_score_norm"] * feat["epss_score"]

    # Replace any accidental non-finite values before sklearn sees the matrix.
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feature_cols = list(feat.columns)
    return feat, feature_cols


def balanced_sample_weight(y: pd.Series | np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=int)
    n = len(y_arr)
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    if n == 0 or n_pos == 0 or n_neg == 0:
        return np.ones(n, dtype=float)
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    return np.where(y_arr == 1, w_pos, w_neg).astype(float)


def train_structured_hgb(train_df: pd.DataFrame):
    X_train, feature_cols = add_structured_features(train_df)
    y_train = train_df["target"].astype(int)
    weights = balanced_sample_weight(y_train)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1e-4,
            random_state=42,
        )),
    ])
    model.fit(X_train[feature_cols], y_train, clf__sample_weight=weights)
    return model, feature_cols


def train_structured_logreg(train_df: pd.DataFrame):
    X_train, feature_cols = add_structured_features(train_df)
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
    return model, feature_cols


def score_structured(model, feature_cols: List[str], df: pd.DataFrame) -> np.ndarray:
    X, _ = add_structured_features(df)
    return model.predict_proba(X[feature_cols])[:, 1].astype(float)


def train_tfidf_logreg(train_df: pd.DataFrame, max_features: int, min_df: int, ngram_max: int):
    y_train = train_df["target"].astype(int)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, ngram_max),
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    model.fit(train_df["text"].fillna("").astype(str), y_train)
    return model


def score_tfidf(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df["text"].fillna("").astype(str))[:, 1].astype(float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate learned same-cutoff baselines for ThreatLens.")
    ap.add_argument("--snapshots", required=True, help="Final enriched snapshot CSV.")
    ap.add_argument("--events", required=True, help="Events CSV, usually Data/events_kev.csv.")
    ap.add_argument("--window_days", type=int, default=30)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--baselines", default="structured_hgb,tfidf_logreg",
                    help="Comma-separated: structured_hgb, structured_logreg, tfidf_logreg")
    ap.add_argument("--k_values", default="10,20,50,100")
    ap.add_argument("--tfidf_max_features", type=int, default=50000)
    ap.add_argument("--tfidf_min_df", type=int, default=2)
    ap.add_argument("--tfidf_ngram_max", type=int, default=2)
    ap.add_argument("--out_json", default="Data/learned_baselines_samecutoff.json")
    ap.add_argument("--out_scores", default="Data/learned_baselines_samecutoff_scores.csv")
    args = ap.parse_args()

    k_values = tuple(int(x) for x in args.k_values.split(",") if x.strip())
    baselines = [x.strip().lower() for x in args.baselines.split(",") if x.strip()]

    snapshots = load_snapshots(Path(args.snapshots))
    events = load_events(Path(args.events))
    labeled = build_labels(snapshots, events, args.window_days)
    train_df, val_df, test_df = temporal_disjoint_split(labeled, args.train_frac, args.val_frac)

    print("Split summary")
    print(f"train rows={len(train_df)} unique_cves={train_df['cve'].nunique()} positives={int(train_df['target'].sum())}")
    print(f"val   rows={len(val_df)} unique_cves={val_df['cve'].nunique()} positives={int(val_df['target'].sum())}")
    print(f"test  rows={len(test_df)} unique_cves={test_df['cve'].nunique()} positives={int(test_df['target'].sum())}")
    print(f"Available columns: {', '.join(labeled.columns)}")

    _, structured_cols = add_structured_features(train_df)
    print("\nStructured feature columns used by learned tabular baselines:")
    for col in structured_cols:
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
        "baselines": {},
    }

    score_out = pd.concat([
        val_df[["cve", "cutoff_date", "target"]].assign(split="val"),
        test_df[["cve", "cutoff_date", "target"]].assign(split="test"),
    ], axis=0, ignore_index=True)

    for baseline in baselines:
        if baseline == "structured_hgb":
            model, feature_cols = train_structured_hgb(train_df)
            val_scores = score_structured(model, feature_cols, val_df)
            test_scores = score_structured(model, feature_cols, test_df)
            model_info = {"type": "HistGradientBoostingClassifier", "features": feature_cols}

        elif baseline == "structured_logreg":
            model, feature_cols = train_structured_logreg(train_df)
            val_scores = score_structured(model, feature_cols, val_df)
            test_scores = score_structured(model, feature_cols, test_df)
            model_info = {"type": "LogisticRegression", "features": feature_cols}

        elif baseline == "tfidf_logreg":
            model = train_tfidf_logreg(
                train_df,
                max_features=args.tfidf_max_features,
                min_df=args.tfidf_min_df,
                ngram_max=args.tfidf_ngram_max,
            )
            val_scores = score_tfidf(model, val_df)
            test_scores = score_tfidf(model, test_df)
            model_info = {
                "type": "TFIDF_LogisticRegression",
                "max_features": args.tfidf_max_features,
                "min_df": args.tfidf_min_df,
                "ngram_range": [1, args.tfidf_ngram_max],
            }

        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        val_metrics = evaluate_scores(f"VAL baseline={baseline}", val_scores, val_df, k_values)
        test_metrics = evaluate_scores(f"TEST baseline={baseline}", test_scores, test_df, k_values)
        results["baselines"][baseline] = {
            "model": model_info,
            "val": val_metrics,
            "test": test_metrics,
        }

        combined_scores = np.concatenate([val_scores, test_scores])
        score_out[f"{baseline}_score"] = combined_scores

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    out_scores = Path(args.out_scores)
    out_scores.parent.mkdir(parents=True, exist_ok=True)
    score_out.to_csv(out_scores, index=False)

    print(f"\nWrote learned baseline results JSON: {out_json}")
    print(f"Wrote learned baseline scores CSV: {out_scores}")
    print("\nJSON metrics")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
