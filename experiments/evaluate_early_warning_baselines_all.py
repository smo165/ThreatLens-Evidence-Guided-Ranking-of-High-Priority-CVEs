#!/usr/bin/env python3
"""
evaluate_early_warning_baselines.py

Early-warning baselines for ThreatLens.

This script evaluates the non-neural early-warning baselines used for ThreatLens:
  - random
  - cvss
  - public_exploit_count / exploit_count
  - epss
  - structured_combo / structured_rule
  - tfidf_logreg / tfidf_linear
  - structured_hgb
  - structured_logreg

It supports both:
  1. Forward-time Version-B early-warning analysis over test-split target CVEs.
  2. K-fold target-CVE early-warning analysis.

It intentionally does NOT train/evaluate the neural ranker. The neural early-warning
results are already produced by analyze_early_warning_version_b.py and
analyze_early_warning_kfold.py. This script is for baseline comparison only.

Example:
  python agents/evaluate_early_warning_baselines.py \
    --snapshots Data/snapshots_weekly_kev_forecast_epss_ghsa_exploits.csv \
    --events Data/events_kev.csv \
    --window_days 30 \
    --baselines random,cvss,public_exploit_count,epss,structured_combo,tfidf_logreg,structured_hgb,structured_logreg \
    --folds 10 \
    --out_dir Data/early_warning_baselines
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CVSS_RE = re.compile(r"base_score=([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_k_values(text: str) -> List[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals:
        raise argparse.ArgumentTypeError("--k_values must contain at least one integer")
    return vals


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("integer list must contain at least one value")
    return vals


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Loading / labels / splits
# -----------------------------------------------------------------------------

def load_snapshots(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cve", "cutoff_date", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Snapshots CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["cve"] = df["cve"].astype(str)
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["cutoff_date"]).reset_index(drop=True)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cve", "event_type", "event_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["cve"] = df["cve"].astype(str)
    df["event_type"] = df["event_type"].astype(str).str.upper()
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_date"]).reset_index(drop=True)
    return df


def build_labels(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    window_days: int,
    positive_events: Set[str],
) -> pd.DataFrame:
    """Construct future-window labels while preserving all snapshot columns."""
    grouped: Dict[str, List[Tuple[str, pd.Timestamp]]] = {}
    for row in events.itertuples(index=False):
        grouped.setdefault(row.cve, []).append((row.event_type, row.event_date))

    rows: List[Dict[str, object]] = []
    for row in snapshots.itertuples(index=False):
        row_dict = row._asdict()
        cve = str(row_dict["cve"])
        t = row_dict["cutoff_date"]
        evs = grouped.get(cve, [])

        already_positive = any((etype in positive_events) and (edate <= t) for etype, edate in evs)
        if already_positive:
            continue

        future_end = t + pd.Timedelta(days=window_days)
        future_positive = any((etype in positive_events) and (t < edate <= future_end) for etype, edate in evs)
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


def first_target_events(events: pd.DataFrame, target_event_type: str) -> pd.DataFrame:
    target = target_event_type.upper().strip()
    df = events[events["event_type"] == target].copy()
    if df.empty:
        raise ValueError(f"No events found for target_event_type={target!r}")
    df = df.sort_values(["cve", "event_date"]).drop_duplicates("cve", keep="first")
    return df.rename(columns={"event_date": "kev_date", "event_type": "kev_event_type"}).reset_index(drop=True)


def target_cves_with_pre_event_snapshots(snapshots: pd.DataFrame, first_events: pd.DataFrame) -> Set[str]:
    merged = snapshots[["cve", "cutoff_date"]].merge(first_events[["cve", "kev_date"]], on="cve", how="inner")
    merged = merged[merged["cutoff_date"] < merged["kev_date"]]
    return set(merged["cve"].unique())


def select_forward_analysis_cves(scope: str, first_events: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Set[str]:
    event_cves = set(first_events["cve"].unique())
    if scope == "test_cves":
        return set(test_df["cve"].unique()) & event_cves
    if scope == "val_cves":
        return set(val_df["cve"].unique()) & event_cves
    if scope == "val_test_cves":
        return (set(val_df["cve"].unique()) | set(test_df["cve"].unique())) & event_cves
    if scope == "all_event_cves":
        return event_cves
    raise ValueError(f"Unknown analysis_scope: {scope}")


# -----------------------------------------------------------------------------
# Structured features / scoring
# -----------------------------------------------------------------------------

def numeric_col(df: pd.DataFrame, names: Iterable[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def bool_col(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            s = df[name]
            if s.dtype == bool:
                return s.astype(float)
            if np.issubdtype(s.dtype, np.number):
                return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(float)
            return s.astype(str).str.lower().isin({"1", "true", "yes", "y"}).astype(float)
    return pd.Series(0.0, index=df.index, dtype=float)


def days_between(later: pd.Series, earlier: pd.Series, default: float = 0.0) -> pd.Series:
    later_dt = pd.to_datetime(later, utc=True, errors="coerce")
    earlier_dt = pd.to_datetime(earlier, utc=True, errors="coerce")
    out = (later_dt - earlier_dt).dt.days.astype("float")
    return out.fillna(default).clip(lower=0.0)


def extract_cvss_from_text(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    m = CVSS_RE.search(text)
    if not m:
        return 0.0
    try:
        val = float(m.group(1))
        if math.isfinite(val):
            return max(0.0, min(10.0, val))
    except ValueError:
        pass
    return 0.0


def add_structured_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feat = pd.DataFrame(index=df.index)

    feat["epss_found"] = bool_col(df, ["epss_found"])
    feat["epss_score"] = numeric_col(df, ["epss_score", "epss"], default=0.0).clip(lower=0.0, upper=1.0)
    feat["epss_percentile"] = numeric_col(df, ["epss_percentile"], default=0.0).clip(lower=0.0, upper=1.0)

    if "cvss_base_score" in df.columns:
        feat["cvss_base_score"] = numeric_col(df, ["cvss_base_score"], default=0.0).clip(lower=0.0, upper=10.0)
    else:
        feat["cvss_base_score"] = df["text"].map(extract_cvss_from_text).astype(float)
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


def train_structured_logreg(train_df: pd.DataFrame):
    X_train, feature_cols = add_structured_features(train_df)
    y_train = train_df["target"].astype(int)
    if int(y_train.sum()) == 0 or int((y_train == 0).sum()) == 0:
        raise ValueError("structured_logreg needs both positive and negative training rows.")

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


def score_structured_logreg(model, feature_cols: List[str], df: pd.DataFrame) -> np.ndarray:
    X, _ = add_structured_features(df)
    return model.predict_proba(X[feature_cols])[:, 1].astype(float)


def score_epss(df: pd.DataFrame) -> np.ndarray:
    X, _ = add_structured_features(df)
    return X["epss_score"].to_numpy(dtype=float)


def score_cvss(df: pd.DataFrame) -> np.ndarray:
    X, _ = add_structured_features(df)
    return X["cvss_base_score"].to_numpy(dtype=float)


def score_public_exploit_count(df: pd.DataFrame) -> np.ndarray:
    X, _ = add_structured_features(df)
    # Match the same-cutoff rule baseline: count plus a small flag tie-breaker.
    return (X["public_exploit_count"] + 0.1 * X["public_exploit_available_by_cutoff"]).to_numpy(dtype=float)


def score_ghsa_count(df: pd.DataFrame) -> np.ndarray:
    X, _ = add_structured_features(df)
    return (X["ghsa_osv_advisory_count"] + 0.1 * X["ghsa_osv_available_by_cutoff"]).to_numpy(dtype=float)


def score_structured_combo(df: pd.DataFrame) -> np.ndarray:
    X, _ = add_structured_features(df)
    # Match evaluate_rule_baselines_samecutoff.py: simple no-training fusion.
    return (
        X["epss_score"]
        + 0.5 * X["cvss_base_score_norm"]
        + 0.75 * (X["public_exploit_count"] > 0).astype(float)
        + 0.25 * (X["ghsa_osv_advisory_count"] > 0).astype(float)
    ).to_numpy(dtype=float)


def score_random(df: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(len(df)).astype(float)


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
    if int(y_train.sum()) == 0 or int((y_train == 0).sum()) == 0:
        raise ValueError("structured_hgb needs both positive and negative training rows.")
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


def train_tfidf_logreg(train_df: pd.DataFrame, max_features: int, min_df: int, ngram_max: int):
    y_train = train_df["target"].astype(int)
    if int(y_train.sum()) == 0 or int((y_train == 0).sum()) == 0:
        raise ValueError("tfidf_logreg needs both positive and negative training rows.")
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


def baseline_needs_training(baseline: str) -> bool:
    return baseline in {"structured_logreg", "structured_hgb", "tfidf_logreg"}


def canonical_baseline_name(baseline: str) -> str:
    b = baseline.lower().strip()
    aliases = {
        "exploit_count": "public_exploit_count",
        "public_exploit": "public_exploit_count",
        "structured_rule": "structured_combo",
        "rule": "structured_combo",
        "tfidf": "tfidf_logreg",
        "tfidf_linear": "tfidf_logreg",
        "hgb": "structured_hgb",
        "logreg": "structured_logreg",
    }
    return aliases.get(b, b)


# -----------------------------------------------------------------------------
# Ranking / early-warning summaries
# -----------------------------------------------------------------------------

def rank_within_cutoffs(scored: pd.DataFrame) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for _, g in scored.groupby("cutoff_date", sort=True):
        tmp = g.sort_values(["score", "cve"], ascending=[False, True]).copy()
        tmp["rank"] = np.arange(1, len(tmp) + 1)
        tmp["candidate_count_at_cutoff"] = len(tmp)
        parts.append(tmp)
    if not parts:
        raise ValueError("No scored rows to rank.")
    return pd.concat(parts, ignore_index=True)


def summarize_early_warning_per_cve(
    ranked: pd.DataFrame,
    first_events: pd.DataFrame,
    analysis_cves: Set[str],
    k_values: List[int],
    fold_idx: int | None = None,
) -> pd.DataFrame:
    merged = ranked.merge(first_events[["cve", "kev_date", "kev_event_type"]], on="cve", how="inner")
    merged = merged[merged["cve"].isin(analysis_cves)].copy()
    merged = merged[merged["cutoff_date"] < merged["kev_date"]].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["days_before_kev"] = (
        merged["kev_date"].dt.normalize() - merged["cutoff_date"].dt.normalize()
    ).dt.days.astype(int)

    rows: List[Dict[str, object]] = []
    for cve, g in merged.groupby("cve", sort=True):
        g = g.sort_values("cutoff_date")
        best_row = g.loc[g["rank"].idxmin()]
        base: Dict[str, object] = {}
        if fold_idx is not None:
            base["fold"] = int(fold_idx)
        base.update({
            "cve": cve,
            "kev_date": g["kev_date"].iloc[0].date().isoformat(),
            "kev_event_type": g["kev_event_type"].iloc[0],
            "num_pre_kev_cutoffs_available": int(len(g)),
            "earliest_pre_kev_cutoff": g.iloc[0]["cutoff_date"].date().isoformat(),
            "max_available_days_before_kev": int(g.iloc[0]["days_before_kev"]),
            "last_pre_kev_cutoff": g.iloc[-1]["cutoff_date"].date().isoformat(),
            "last_available_days_before_kev": int(g.iloc[-1]["days_before_kev"]),
            "best_rank_before_kev": int(best_row["rank"]),
            "best_rank_cutoff": best_row["cutoff_date"].date().isoformat(),
            "days_before_kev_at_best_rank": int(best_row["days_before_kev"]),
            "best_score_before_kev": float(best_row["score"]),
        })
        for k in k_values:
            hits = g[g["rank"] <= k].sort_values("cutoff_date")
            if hits.empty:
                base[f"hit_top_{k}"] = 0
                base[f"earliest_top_{k}_cutoff"] = ""
                base[f"days_before_kev_top_{k}"] = ""
                base[f"rank_at_earliest_top_{k}"] = ""
                base[f"score_at_earliest_top_{k}"] = ""
            else:
                first = hits.iloc[0]
                base[f"hit_top_{k}"] = 1
                base[f"earliest_top_{k}_cutoff"] = first["cutoff_date"].date().isoformat()
                base[f"days_before_kev_top_{k}"] = int(first["days_before_kev"])
                base[f"rank_at_earliest_top_{k}"] = int(first["rank"])
                base[f"score_at_earliest_top_{k}"] = float(first["score"])
        rows.append(base)

    sort_cols = ["kev_date", "cve"] if fold_idx is None else ["fold", "kev_date", "cve"]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def aggregate_per_cve(per_cve: pd.DataFrame, k_values: List[int], analysis_version: str) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "analysis_version": analysis_version,
        "analysis_rule": "For each target CVE, count top-K hits only at cutoffs before the target event date.",
        "evaluated_cves_with_pre_event_snapshots": int(per_cve["cve"].nunique()) if not per_cve.empty else 0,
        "k_values": k_values,
    }
    for k in k_values:
        hit_col = f"hit_top_{k}"
        day_col = f"days_before_kev_top_{k}"
        if per_cve.empty:
            hits = 0
            hit_rate = 0.0
            vals = pd.Series(dtype=float)
        else:
            hits = int(per_cve[hit_col].sum())
            denom = int(len(per_cve))
            hit_rate = float(hits / denom) if denom else 0.0
            vals = pd.to_numeric(per_cve[day_col], errors="coerce").dropna()
        summary[f"top_{k}"] = {
            "hits": hits,
            "hit_rate_among_cves_with_pre_event_snapshots": hit_rate,
            "median_days_before_event": float(vals.median()) if len(vals) else None,
            "mean_days_before_event": float(vals.mean()) if len(vals) else None,
            "min_days_before_event": int(vals.min()) if len(vals) else None,
            "max_days_before_event": int(vals.max()) if len(vals) else None,
            "p25_days_before_event": float(vals.quantile(0.25)) if len(vals) else None,
            "p75_days_before_event": float(vals.quantile(0.75)) if len(vals) else None,
        }
    return summary



def average_random_summaries(
    summaries: List[Dict[str, object]],
    k_values: List[int],
    *,
    analysis_version: str,
    random_seeds: List[int],
) -> Dict[str, object]:
    """Average early-warning summary metrics across random seeds."""
    if not summaries:
        raise ValueError("Cannot average an empty random summary list")

    out = dict(summaries[0])
    out["analysis_version"] = analysis_version
    out["model"] = {
        "type": "random",
        "random_seeds": [int(s) for s in random_seeds],
        "num_random_seeds": int(len(random_seeds)),
        "aggregation": "mean over random seeds",
    }
    out["evaluated_cves_with_pre_event_snapshots"] = summaries[0].get("evaluated_cves_with_pre_event_snapshots")
    out["random_seed_summaries"] = [
        {
            "seed": int(seed),
            **{f"top_{k}": summaries[i].get(f"top_{k}", {}) for k in k_values},
        }
        for i, seed in enumerate(random_seeds)
    ]

    for k in k_values:
        key = f"top_{k}"
        per_seed = [s.get(key, {}) for s in summaries]

        def mean_field(field: str):
            vals = [b.get(field) for b in per_seed if b.get(field) is not None]
            return float(np.mean(vals)) if vals else None

        out[key] = {
            "hits": mean_field("hits"),
            "hits_by_seed": [b.get("hits") for b in per_seed],
            "hit_rate_among_cves_with_pre_event_snapshots": mean_field("hit_rate_among_cves_with_pre_event_snapshots"),
            "hit_rate_by_seed": [b.get("hit_rate_among_cves_with_pre_event_snapshots") for b in per_seed],
            "median_days_before_event": mean_field("median_days_before_event"),
            "median_days_by_seed": [b.get("median_days_before_event") for b in per_seed],
            "mean_days_before_event": mean_field("mean_days_before_event"),
            "p25_days_before_event": mean_field("p25_days_before_event"),
            "p75_days_before_event": mean_field("p75_days_before_event"),
            "min_days_before_event": mean_field("min_days_before_event"),
            "max_days_before_event": mean_field("max_days_before_event"),
        }

    return out


def per_fold_summary(per_cve: pd.DataFrame, k_values: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if per_cve.empty:
        return pd.DataFrame()
    for fold, g in per_cve.groupby("fold", sort=True):
        row: Dict[str, object] = {"fold": int(fold), "evaluated_cves": int(len(g))}
        for k in k_values:
            hit_col = f"hit_top_{k}"
            day_col = f"days_before_kev_top_{k}"
            hits = int(g[hit_col].sum())
            vals = pd.to_numeric(g[day_col], errors="coerce").dropna()
            row[f"top_{k}_hits"] = hits
            row[f"top_{k}_hit_rate"] = float(hits / len(g)) if len(g) else 0.0
            row[f"top_{k}_median_days"] = float(vals.median()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def relevant_cutoffs_for_targets(snapshots: pd.DataFrame, first_events: pd.DataFrame, target_cves: Set[str]) -> Set[pd.Timestamp]:
    if not target_cves:
        return set()
    target_dates = first_events[first_events["cve"].isin(target_cves)][["cve", "kev_date"]]
    merged = snapshots[["cve", "cutoff_date"]].merge(target_dates, on="cve", how="inner")
    merged = merged[merged["cutoff_date"] < merged["kev_date"]]
    return {pd.Timestamp(x) for x in merged["cutoff_date"].unique()}


def make_folds(items: Sequence[str], folds: int, seed: int) -> List[Set[str]]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    rng = np.random.default_rng(seed)
    arr = np.array(sorted(items), dtype=object)
    rng.shuffle(arr)
    parts = np.array_split(arr, folds)
    return [set(map(str, p.tolist())) for p in parts]


# -----------------------------------------------------------------------------
# Baseline evaluators
# -----------------------------------------------------------------------------

def get_scorer_name(baseline: str) -> str:
    names = {
        "random": "Random",
        "cvss": "CVSS",
        "public_exploit_count": "Exploit count",
        "ghsa_count": "GHSA advisory count",
        "epss": "EPSS",
        "structured_combo": "Structured rule",
        "tfidf_logreg": "TF-IDF linear",
        "structured_hgb": "Structured HGB",
        "structured_logreg": "Structured logistic",
    }
    return names.get(canonical_baseline_name(baseline), baseline)


def score_baseline(
    baseline: str,
    train_df: pd.DataFrame | None,
    scored_df: pd.DataFrame,
    *,
    seed: int,
    tfidf_max_features: int,
    tfidf_min_df: int,
    tfidf_ngram_max: int,
):
    baseline = canonical_baseline_name(baseline)

    if baseline == "random":
        return score_random(scored_df, seed=seed), {"type": "random", "seed": seed}
    if baseline == "cvss":
        return score_cvss(scored_df), {"type": "CVSS base score"}
    if baseline == "public_exploit_count":
        return score_public_exploit_count(scored_df), {"type": "public exploit count + flag tie-breaker"}
    if baseline == "ghsa_count":
        return score_ghsa_count(scored_df), {"type": "GHSA advisory count + flag tie-breaker"}
    if baseline == "epss":
        return score_epss(scored_df), {"type": "EPSS score", "features": ["epss_score"]}
    if baseline == "structured_combo":
        return score_structured_combo(scored_df), {"type": "hand-weighted structured rule"}

    if train_df is None:
        raise ValueError(f"{baseline} requires a training dataframe")

    if baseline == "structured_logreg":
        model, feature_cols = train_structured_logreg(train_df)
        return score_structured_logreg(model, feature_cols, scored_df), {"type": "LogisticRegression", "features": feature_cols}
    if baseline == "structured_hgb":
        model, feature_cols = train_structured_hgb(train_df)
        return score_structured_logreg(model, feature_cols, scored_df), {"type": "HistGradientBoostingClassifier", "features": feature_cols}
    if baseline == "tfidf_logreg":
        model = train_tfidf_logreg(
            train_df,
            max_features=tfidf_max_features,
            min_df=tfidf_min_df,
            ngram_max=tfidf_ngram_max,
        )
        return score_tfidf(model, scored_df), {
            "type": "TFIDF_LogisticRegression",
            "max_features": tfidf_max_features,
            "min_df": tfidf_min_df,
            "ngram_range": [1, tfidf_ngram_max],
        }

    raise ValueError(f"Unknown baseline: {baseline}")


def run_forward_analysis(
    baseline: str,
    snapshots: pd.DataFrame,
    labeled: pd.DataFrame,
    events: pd.DataFrame,
    first_events: pd.DataFrame,
    args,
    out_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    train_df, val_df, test_df = temporal_disjoint_split(labeled, args.train_frac, args.val_frac)
    analysis_cves = select_forward_analysis_cves(args.analysis_scope, first_events, train_df, val_df, test_df)
    if not analysis_cves:
        raise ValueError(f"No target-event CVEs found for forward analysis_scope={args.analysis_scope!r}")

    print(f"\n[forward] baseline={baseline} analysis_scope={args.analysis_scope} analysis_cves={len(analysis_cves)}")
    print(f"[forward] train rows={len(train_df)} unique_cves={train_df['cve'].nunique()} positives={int(train_df['target'].sum())}")
    print(f"[forward] val   rows={len(val_df)} unique_cves={val_df['cve'].nunique()} positives={int(val_df['target'].sum())}")
    print(f"[forward] test  rows={len(test_df)} unique_cves={test_df['cve'].nunique()} positives={int(test_df['target'].sum())}")

    if canonical_baseline_name(baseline) == "random" and len(getattr(args, "random_seeds", [])) > 1:
        print(f"[forward] random baseline: averaging over seeds={args.random_seeds}")
        per_seed_frames: List[pd.DataFrame] = []
        seed_summaries: List[Dict[str, object]] = []

        for random_seed in args.random_seeds:
            scored = snapshots.copy().reset_index(drop=True)
            scored["score"] = score_random(scored, seed=int(random_seed))
            ranked = rank_within_cutoffs(scored)
            per_cve_seed = summarize_early_warning_per_cve(ranked, first_events, analysis_cves, args.k_values)
            per_cve_seed["random_seed"] = int(random_seed)
            per_seed_frames.append(per_cve_seed)

            seed_summary = aggregate_per_cve(per_cve_seed, args.k_values, analysis_version="forward_time_version_b")
            seed_summary.update({
                "baseline": baseline,
                "method": get_scorer_name(baseline),
                "model": {"type": "random", "seed": int(random_seed)},
                "target_event_type": args.target_event_type.upper(),
                "analysis_scope": args.analysis_scope,
                "training_window_days": args.window_days,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "ranked_rows_scored": int(len(ranked)),
                "ranked_cutoffs_scored": int(ranked["cutoff_date"].nunique()),
            })
            seed_summaries.append(seed_summary)

        per_cve = pd.concat(per_seed_frames, ignore_index=True)
        summary = average_random_summaries(
            seed_summaries,
            args.k_values,
            analysis_version="forward_time_version_b",
            random_seeds=args.random_seeds,
        )
        summary.update({
            "baseline": baseline,
            "method": get_scorer_name(baseline),
            "target_event_type": args.target_event_type.upper(),
            "analysis_scope": args.analysis_scope,
            "training_window_days": args.window_days,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "ranked_rows_scored": int(len(snapshots)),
            "ranked_cutoffs_scored": int(snapshots["cutoff_date"].nunique()),
        })

        per_cve_path = out_dir / f"forward_{baseline}_per_cve.csv"
        summary_path = out_dir / f"forward_{baseline}_summary.json"
        per_cve.to_csv(per_cve_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[forward] wrote {per_cve_path}")
        print(f"[forward] wrote {summary_path}")
        print(json.dumps(summary, indent=2))
        return per_cve, summary

    # Match analyze_early_warning_version_b.py: score the full snapshot table, then rank within each cutoff.
    scored = snapshots.copy().reset_index(drop=True)
    scores, model_info = score_baseline(
        baseline,
        train_df if baseline_needs_training(canonical_baseline_name(baseline)) else None,
        scored,
        seed=args.seed,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_ngram_max=args.tfidf_ngram_max,
    )
    scored["score"] = scores
    ranked = rank_within_cutoffs(scored)
    per_cve = summarize_early_warning_per_cve(ranked, first_events, analysis_cves, args.k_values)
    summary = aggregate_per_cve(per_cve, args.k_values, analysis_version="forward_time_version_b")
    summary.update({
        "baseline": baseline,
        "method": get_scorer_name(baseline),
        "model": model_info,
        "target_event_type": args.target_event_type.upper(),
        "analysis_scope": args.analysis_scope,
        "training_window_days": args.window_days,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "ranked_rows_scored": int(len(ranked)),
        "ranked_cutoffs_scored": int(ranked["cutoff_date"].nunique()),
    })

    per_cve_path = out_dir / f"forward_{baseline}_per_cve.csv"
    summary_path = out_dir / f"forward_{baseline}_summary.json"
    per_cve.to_csv(per_cve_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[forward] wrote {per_cve_path}")
    print(f"[forward] wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    return per_cve, summary


def run_kfold_analysis(
    baseline: str,
    snapshots: pd.DataFrame,
    labeled: pd.DataFrame,
    first_events: pd.DataFrame,
    target_cves_available: Set[str],
    args,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if args.fold_universe == "target_event_cves":
        fold_source_cves = sorted(target_cves_available)
    else:
        fold_source_cves = sorted(set(labeled["cve"].unique()) | set(target_cves_available))
    folds = make_folds(fold_source_cves, args.folds, args.seed)

    print(f"\n[kfold] baseline={baseline} folds={args.folds} fold_universe={args.fold_universe}")
    print(f"[kfold] target-event CVEs with pre-event snapshots available: {len(target_cves_available)}")
    print(f"[kfold] CVEs used to define folds: {len(fold_source_cves)}")

    if canonical_baseline_name(baseline) == "random" and len(getattr(args, "random_seeds", [])) > 1:
        print(f"[kfold] random baseline: averaging over seeds={args.random_seeds}")
        all_seed_per_cve: List[pd.DataFrame] = []
        seed_summaries: List[Dict[str, object]] = []
        first_fold_summary: pd.DataFrame | None = None
        first_fold_meta: List[Dict[str, object]] | None = None

        for random_seed in args.random_seeds:
            print(f"[kfold] random seed={random_seed}")
            seed_per_cve_frames: List[pd.DataFrame] = []
            seed_fold_meta: List[Dict[str, object]] = []

            for fold_idx, heldout_cves in enumerate(folds, start=1):
                heldout_targets = heldout_cves & target_cves_available
                if not heldout_targets:
                    print(f"[kfold] fold {fold_idx}: no held-out target CVEs; skipping")
                    continue

                train_df = labeled[~labeled["cve"].isin(heldout_cves)].copy().reset_index(drop=True)

                if args.score_scope == "relevant_cutoffs":
                    cutoffs = relevant_cutoffs_for_targets(snapshots, first_events, heldout_targets)
                    if not cutoffs:
                        print(f"[kfold] fold {fold_idx}: no relevant pre-event cutoffs; skipping")
                        continue
                    scored = snapshots[snapshots["cutoff_date"].isin(cutoffs)].copy().reset_index(drop=True)
                else:
                    scored = snapshots.copy().reset_index(drop=True)

                print(
                    f"[kfold] seed {random_seed} fold {fold_idx}/{args.folds}: "
                    f"heldout_targets={len(heldout_targets)} train_rows={len(train_df)} "
                    f"scored_rows={len(scored)} scored_cutoffs={scored['cutoff_date'].nunique()}"
                )

                # Fold-dependent offset gives each fold a deterministic but distinct random ranking.
                scored["score"] = score_random(scored, seed=int(random_seed) + 100000 * int(fold_idx))
                ranked = rank_within_cutoffs(scored)
                fold_per_cve = summarize_early_warning_per_cve(
                    ranked, first_events, heldout_targets, args.k_values, fold_idx=fold_idx
                )
                if not fold_per_cve.empty:
                    seed_per_cve_frames.append(fold_per_cve)

                seed_fold_meta.append({
                    "fold": int(fold_idx),
                    "heldout_fold_cves": int(len(heldout_cves)),
                    "heldout_target_event_cves": int(len(heldout_targets)),
                    "train_rows": int(len(train_df)),
                    "train_unique_cves": int(train_df["cve"].nunique()),
                    "train_positive_rows": int(train_df["target"].sum()),
                    "scored_rows": int(len(scored)),
                    "scored_cutoffs": int(scored["cutoff_date"].nunique()),
                    "evaluated_target_cves": int(fold_per_cve["cve"].nunique()) if not fold_per_cve.empty else 0,
                })

            if not seed_per_cve_frames:
                raise ValueError(f"No folds produced early-warning results for baseline={baseline}, seed={random_seed}.")

            seed_per_cve = pd.concat(seed_per_cve_frames, ignore_index=True).sort_values(["fold", "kev_date", "cve"]).reset_index(drop=True)
            seed_per_cve["random_seed"] = int(random_seed)
            all_seed_per_cve.append(seed_per_cve)

            seed_fold_summary = per_fold_summary(seed_per_cve, args.k_values)
            if first_fold_summary is None:
                first_fold_summary = seed_fold_summary
                first_fold_meta = seed_fold_meta

            seed_summary = aggregate_per_cve(seed_per_cve, args.k_values, analysis_version="cve_disjoint_kfold_early_warning")
            seed_summary.update({
                "baseline": baseline,
                "method": get_scorer_name(baseline),
                "model": {"type": "random", "seed": int(random_seed)},
                "target_event_type": args.target_event_type.upper(),
                "folds_requested": args.folds,
                "folds_completed": int(seed_per_cve["fold"].nunique()),
                "fold_universe": args.fold_universe,
                "score_scope": args.score_scope,
                "training_window_days": args.window_days,
                "target_event_cves_with_pre_event_snapshots_available": int(len(target_cves_available)),
                "fold_metadata": seed_fold_meta,
            })
            seed_summaries.append(seed_summary)

        per_cve = pd.concat(all_seed_per_cve, ignore_index=True)
        fold_summary = first_fold_summary if first_fold_summary is not None else pd.DataFrame()
        summary = average_random_summaries(
            seed_summaries,
            args.k_values,
            analysis_version="cve_disjoint_kfold_early_warning",
            random_seeds=args.random_seeds,
        )
        summary.update({
            "baseline": baseline,
            "method": get_scorer_name(baseline),
            "target_event_type": args.target_event_type.upper(),
            "folds_requested": args.folds,
            "folds_completed": int(seed_summaries[0].get("folds_completed", 0)),
            "fold_universe": args.fold_universe,
            "score_scope": args.score_scope,
            "training_window_days": args.window_days,
            "target_event_cves_with_pre_event_snapshots_available": int(len(target_cves_available)),
            "fold_metadata": first_fold_meta if first_fold_meta is not None else [],
        })

        per_cve_path = out_dir / f"kfold_{baseline}_per_cve.csv"
        fold_path = out_dir / f"kfold_{baseline}_per_fold.csv"
        summary_path = out_dir / f"kfold_{baseline}_summary.json"
        per_cve.to_csv(per_cve_path, index=False)
        fold_summary.to_csv(fold_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[kfold] wrote {per_cve_path}")
        print(f"[kfold] wrote {fold_path}")
        print(f"[kfold] wrote {summary_path}")
        print(json.dumps(summary, indent=2))
        return per_cve, fold_summary, summary

    all_per_cve: List[pd.DataFrame] = []
    fold_meta: List[Dict[str, object]] = []

    for fold_idx, heldout_cves in enumerate(folds, start=1):
        heldout_targets = heldout_cves & target_cves_available
        if not heldout_targets:
            print(f"[kfold] fold {fold_idx}: no held-out target CVEs; skipping")
            continue

        train_df = labeled[~labeled["cve"].isin(heldout_cves)].copy().reset_index(drop=True)
        if baseline_needs_training(canonical_baseline_name(baseline)) and (
            train_df.empty or int(train_df["target"].sum()) == 0 or int((train_df["target"] == 0).sum()) == 0
        ):
            raise ValueError(f"Fold {fold_idx} has insufficient training data for baseline={baseline}.")

        if args.score_scope == "relevant_cutoffs":
            cutoffs = relevant_cutoffs_for_targets(snapshots, first_events, heldout_targets)
            if not cutoffs:
                print(f"[kfold] fold {fold_idx}: no relevant pre-event cutoffs; skipping")
                continue
            scored = snapshots[snapshots["cutoff_date"].isin(cutoffs)].copy().reset_index(drop=True)
        else:
            scored = snapshots.copy().reset_index(drop=True)

        print(
            f"[kfold] fold {fold_idx}/{args.folds}: heldout_targets={len(heldout_targets)} "
            f"train_rows={len(train_df)} positives={int(train_df['target'].sum())} "
            f"scored_rows={len(scored)} scored_cutoffs={scored['cutoff_date'].nunique()}"
        )

        scores, model_info = score_baseline(
        baseline,
        train_df if baseline_needs_training(canonical_baseline_name(baseline)) else None,
        scored,
        seed=args.seed,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_ngram_max=args.tfidf_ngram_max,
    )
        scored["score"] = scores
        ranked = rank_within_cutoffs(scored)
        fold_per_cve = summarize_early_warning_per_cve(ranked, first_events, heldout_targets, args.k_values, fold_idx=fold_idx)
        if not fold_per_cve.empty:
            all_per_cve.append(fold_per_cve)

        fold_meta.append({
            "fold": int(fold_idx),
            "heldout_fold_cves": int(len(heldout_cves)),
            "heldout_target_event_cves": int(len(heldout_targets)),
            "train_rows": int(len(train_df)),
            "train_unique_cves": int(train_df["cve"].nunique()),
            "train_positive_rows": int(train_df["target"].sum()),
            "scored_rows": int(len(scored)),
            "scored_cutoffs": int(scored["cutoff_date"].nunique()),
            "evaluated_target_cves": int(fold_per_cve["cve"].nunique()) if not fold_per_cve.empty else 0,
        })

    if not all_per_cve:
        raise ValueError(f"No folds produced early-warning results for baseline={baseline}.")

    per_cve = pd.concat(all_per_cve, ignore_index=True).sort_values(["fold", "kev_date", "cve"]).reset_index(drop=True)
    fold_summary = per_fold_summary(per_cve, args.k_values)
    summary = aggregate_per_cve(per_cve, args.k_values, analysis_version="cve_disjoint_kfold_early_warning")
    summary.update({
        "baseline": baseline,
        "method": get_scorer_name(baseline),
        "model": model_info,
        "target_event_type": args.target_event_type.upper(),
        "folds_requested": args.folds,
        "folds_completed": int(per_cve["fold"].nunique()),
        "fold_universe": args.fold_universe,
        "score_scope": args.score_scope,
        "training_window_days": args.window_days,
        "target_event_cves_with_pre_event_snapshots_available": int(len(target_cves_available)),
        "fold_metadata": fold_meta,
    })

    per_cve_path = out_dir / f"kfold_{baseline}_per_cve.csv"
    fold_path = out_dir / f"kfold_{baseline}_per_fold.csv"
    summary_path = out_dir / f"kfold_{baseline}_summary.json"
    per_cve.to_csv(per_cve_path, index=False)
    fold_summary.to_csv(fold_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[kfold] wrote {per_cve_path}")
    print(f"[kfold] wrote {fold_path}")
    print(f"[kfold] wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    return per_cve, fold_summary, summary


def summary_rows_from_json(summary: Dict[str, object], k_values: List[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for k in k_values:
        block = summary.get(f"top_{k}", {})
        rows.append({
            "analysis": summary.get("analysis_version"),
            "baseline": summary.get("baseline"),
            "method": summary.get("method"),
            "k": k,
            "evaluated_cves": summary.get("evaluated_cves_with_pre_event_snapshots"),
            "hits": block.get("hits"),
            "hit_rate": block.get("hit_rate_among_cves_with_pre_event_snapshots"),
            "median_days": block.get("median_days_before_event"),
            "mean_days": block.get("mean_days_before_event"),
            "p25_days": block.get("p25_days_before_event"),
            "p75_days": block.get("p75_days_before_event"),
        })
    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate non-neural early-warning baselines for ThreatLens.")
    ap.add_argument("--snapshots", required=True, help="Final enriched snapshot CSV.")
    ap.add_argument("--events", required=True, help="Events CSV, usually Data/events_kev.csv.")
    ap.add_argument("--window_days", type=int, default=30, help="Training label horizon. Early-warning lead-time analysis is not capped by this horizon.")
    ap.add_argument("--positive_event_types", default="KEV,INTHEWILD,METASPLOIT,EXPLOITDB",
                    help="Comma-separated event types used to build future-window training labels. Default matches existing evaluators.")
    ap.add_argument("--target_event_type", type=str, default="KEV", help="Event type used as the lead-time endpoint; default is KEV.")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--baselines", default="random,cvss,public_exploit_count,epss,structured_combo,tfidf_logreg,structured_hgb,structured_logreg",
                    help="Comma-separated baselines: random, cvss, public_exploit_count/exploit_count, epss, structured_combo/structured_rule, tfidf_logreg/tfidf_linear, structured_hgb, structured_logreg")
    ap.add_argument("--k_values", type=parse_k_values, default=parse_k_values("10,20,50,100"))
    ap.add_argument("--run_forward", action="store_true", help="Run forward-time Version-B early-warning analysis.")
    ap.add_argument("--run_kfold", action="store_true", help="Run K-fold early-warning analysis.")
    ap.add_argument("--analysis_scope", choices=["test_cves", "val_cves", "val_test_cves", "all_event_cves"], default="test_cves")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_universe", choices=["target_event_cves", "all_cves"], default="target_event_cves")
    ap.add_argument("--score_scope", choices=["relevant_cutoffs", "full_snapshots"], default="relevant_cutoffs")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--random_seeds", type=parse_int_list, default=parse_int_list("1,2,3,4,5"),
                    help="Comma-separated random seeds used only for the random baseline. Random summaries are averaged across these seeds.")
    ap.add_argument("--tfidf_max_features", type=int, default=50000)
    ap.add_argument("--tfidf_min_df", type=int, default=2)
    ap.add_argument("--tfidf_ngram_max", type=int, default=2)
    ap.add_argument("--out_dir", default="Data/early_warning_baselines")
    args = ap.parse_args()

    # If neither flag is passed, run both analyses. This keeps the common use simple.
    if not args.run_forward and not args.run_kfold:
        args.run_forward = True
        args.run_kfold = True

    raw_baselines = [b.lower().strip() for b in parse_csv_list(args.baselines)]
    baselines = [canonical_baseline_name(b) for b in raw_baselines]
    allowed = {
        "random",
        "cvss",
        "public_exploit_count",
        "ghsa_count",
        "epss",
        "structured_combo",
        "tfidf_logreg",
        "structured_hgb",
        "structured_logreg",
    }
    unknown = sorted(set(baselines) - allowed)
    if unknown:
        raise ValueError(f"Unknown baselines {unknown}; allowed={sorted(allowed)}")

    positive_events = {x.upper() for x in parse_csv_list(args.positive_event_types)}
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    print("Loading data")
    print(f"snapshots={args.snapshots}")
    print(f"events={args.events}")
    snapshots = load_snapshots(Path(args.snapshots))
    events = load_events(Path(args.events))
    labeled = build_labels(snapshots, events, args.window_days, positive_events=positive_events)
    first_events = first_target_events(events, args.target_event_type)
    target_available = target_cves_with_pre_event_snapshots(snapshots, first_events)

    print(f"Snapshot rows: {len(snapshots)} unique CVEs={snapshots['cve'].nunique()} cutoffs={snapshots['cutoff_date'].nunique()}")
    print(f"Labeled rows:  {len(labeled)} unique CVEs={labeled['cve'].nunique()} positives={int(labeled['target'].sum())}")
    print(f"Positive event types for training labels: {sorted(positive_events)}")
    print(f"Target event type for lead time: {args.target_event_type.upper()}")
    print(f"Target-event CVEs with pre-event snapshots available: {len(target_available)}")

    all_summary_rows: List[Dict[str, object]] = []
    combined: Dict[str, object] = {
        "snapshots": args.snapshots,
        "events": args.events,
        "window_days": args.window_days,
        "positive_event_types": sorted(positive_events),
        "target_event_type": args.target_event_type.upper(),
        "baselines": baselines,
        "k_values": args.k_values,
        "random_seeds": args.random_seeds,
        "analyses": {},
    }

    for baseline in baselines:
        if args.run_forward:
            _, summary = run_forward_analysis(baseline, snapshots, labeled, events, first_events, args, out_dir)
            combined["analyses"][f"forward_{baseline}"] = summary
            all_summary_rows.extend(summary_rows_from_json(summary, args.k_values))

        if args.run_kfold:
            _, _, summary = run_kfold_analysis(baseline, snapshots, labeled, first_events, target_available, args, out_dir)
            combined["analyses"][f"kfold_{baseline}"] = summary
            all_summary_rows.extend(summary_rows_from_json(summary, args.k_values))

    combined_json = out_dir / "early_warning_baselines_combined_summary.json"
    combined_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    combined_csv = out_dir / "early_warning_baselines_table.csv"
    pd.DataFrame(all_summary_rows).to_csv(combined_csv, index=False)

    print("\nWrote combined outputs:")
    print(combined_json)
    print(combined_csv)
    print("\nCompact table:")
    print(pd.DataFrame(all_summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
