from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

# Keep this consistent with evaluate_neural_prioritizer_transformer_samecutoff.py
# and analyze_early_warning_version_b.py.
POSITIVE_EVENTS: Set[str] = {"KEV", "INTHEWILD", "METASPLOIT", "EXPLOITDB"}


# -----------------------------
# Reproducibility / model
# -----------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ScoreHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TransformerPrioritizer(nn.Module):
    """Transformer text encoder + MLP score head.

    This mirrors the main same-cutoff evaluator: structured evidence text is
    tokenized, encoded with the transformer, pooled, normalized, and scored.
    """

    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        enc_dim = int(self.encoder.config.hidden_size)
        self.scorer = ScoreHead(input_dim=enc_dim, hidden_dim=hidden_dim, dropout=dropout)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        if self.pooling == "cls":
            emb = last[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(last)
            emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return torch.nn.functional.normalize(emb, p=2, dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.scorer(emb)


def pairwise_logistic_loss(score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(-(score_pos - score_neg)).mean()


# -----------------------------
# Data loading / labeling
# -----------------------------

def load_snapshots(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cve", "cutoff_date", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Snapshots CSV missing columns: {sorted(missing)}")
    df = df.copy()
    # Do not alter CVE strings; keep this compatible with the existing evaluator.
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["cutoff_date"]).reset_index(drop=True)
    df["text"] = df["text"].fillna("").astype(str)
    return df.sort_values(["cutoff_date", "cve"]).reset_index(drop=True)


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cve", "event_type", "event_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["event_type"] = df["event_type"].astype(str).str.upper()
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_date"]).reset_index(drop=True)
    return df


def build_labels(snapshots: pd.DataFrame, events: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Build future-event labels exactly like the main evaluator.

    Already-positive rows are dropped. A remaining row is positive if a positive
    event occurs after the cutoff and within the forecast horizon.
    """
    grouped: Dict[str, List[Tuple[str, pd.Timestamp]]] = {}
    for row in events.itertuples(index=False):
        grouped.setdefault(row.cve, []).append((row.event_type, row.event_date))

    rows: List[Dict[str, object]] = []
    for row in snapshots.itertuples(index=False):
        evs = grouped.get(row.cve, [])
        t = row.cutoff_date
        already_positive = any((etype in POSITIVE_EVENTS) and (edate <= t) for etype, edate in evs)
        if already_positive:
            continue
        future_end = t + pd.Timedelta(days=window_days)
        future_positive = any((etype in POSITIVE_EVENTS) and (t < edate <= future_end) for etype, edate in evs)
        rows.append({
            "cve": row.cve,
            "cutoff_date": t,
            "text": row.text,
            "target": 1 if future_positive else 0,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No labeled rows remained after filtering already-positive examples.")
    return out.sort_values(["cutoff_date", "cve"]).reset_index(drop=True)


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


# -----------------------------
# K-fold splitting
# -----------------------------

def make_folds(cves: Sequence[str], n_folds: int, seed: int) -> List[Set[str]]:
    if n_folds < 2:
        raise ValueError("--folds must be at least 2")
    cves = sorted(set(cves))
    if len(cves) < n_folds:
        raise ValueError(f"Need at least as many CVEs as folds: {len(cves)} CVEs, {n_folds} folds")
    rng = np.random.default_rng(seed)
    shuffled = np.array(cves, dtype=object)
    rng.shuffle(shuffled)
    return [set(x.tolist()) for x in np.array_split(shuffled, n_folds)]


def describe_train_df(train_df: pd.DataFrame, fold_idx: int) -> None:
    positives = int(train_df["target"].sum())
    print(
        f"fold={fold_idx}: train rows={len(train_df)} "
        f"unique_cves={train_df['cve'].nunique()} positives={positives}"
    )


# -----------------------------
# Same-cutoff pair sampling / training
# -----------------------------

def build_cutoff_pair_index(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, pd.Timestamp]]:
    groups: List[Tuple[np.ndarray, np.ndarray, pd.Timestamp]] = []
    for cutoff, g in df.groupby("cutoff_date", sort=True):
        idx = g.index.to_numpy(dtype=int)
        labels = g["target"].to_numpy(dtype=int)
        pos = idx[labels == 1]
        neg = idx[labels == 0]
        if len(pos) > 0 and len(neg) > 0:
            groups.append((pos, neg, pd.Timestamp(cutoff)))
    if not groups:
        raise ValueError("No cutoff has both positives and negatives for same-cutoff pair sampling.")
    return groups


def sample_same_cutoff_pairs(
    rng: np.random.Generator,
    groups: List[Tuple[np.ndarray, np.ndarray, pd.Timestamp]],
    num_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    gidx = rng.integers(0, len(groups), size=num_pairs)
    pos_out = np.empty(num_pairs, dtype=int)
    neg_out = np.empty(num_pairs, dtype=int)
    for i, gi in enumerate(gidx):
        pos, neg, _ = groups[int(gi)]
        pos_out[i] = int(rng.choice(pos))
        neg_out[i] = int(rng.choice(neg))
    return pos_out, neg_out


def tokenize_batch(tokenizer, texts: List[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    toks = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in toks.items() if k in {"input_ids", "attention_mask"}}


def train_model(
    train_df: pd.DataFrame,
    model_name: str,
    pooling: str,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
    pairs_per_epoch: int,
    max_length: int,
    batch_size: int,
    seed: int,
    gradient_checkpointing: bool,
    device: torch.device,
):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TransformerPrioritizer(
        model_name=model_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pooling=pooling,
    ).to(device)

    if gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        print("Enabled transformer gradient checkpointing.")

    opt = optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.scorer.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    train_df = train_df.reset_index(drop=True)
    texts = train_df["text"].tolist()
    groups = build_cutoff_pair_index(train_df)
    print(f"Same-cutoff training groups with positives and negatives: {len(groups)}")

    for ep in range(1, epochs + 1):
        model.train()
        ia, ib = sample_same_cutoff_pairs(rng, groups, pairs_per_epoch)
        epoch_losses: List[float] = []

        for start in range(0, len(ia), batch_size):
            pos_idx = ia[start:start + batch_size]
            neg_idx = ib[start:start + batch_size]
            batch_texts = [texts[i] for i in pos_idx] + [texts[j] for j in neg_idx]
            toks = tokenize_batch(tokenizer, batch_texts, max_length, device)
            n_pos = len(pos_idx)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                scores = model(**toks)
                loss = pairwise_logistic_loss(scores[:n_pos], scores[n_pos:])

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu().item()))

        if ep == 1 or ep == epochs or ep % 2 == 0:
            print(f"ep={ep:4d}  loss={np.mean(epoch_losses):.4f}")

    return model, tokenizer


@torch.no_grad()
def score_df(
    model: nn.Module,
    tokenizer,
    df: pd.DataFrame,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    texts = df["text"].tolist()
    scores: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        toks = tokenize_batch(tokenizer, batch, max_length, device)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            s = model(**toks)
        scores.append(s.detach().cpu().float().numpy())
    return np.concatenate(scores, axis=0) if scores else np.array([], dtype=float)


# -----------------------------
# Early-warning ranking / summaries
# -----------------------------

def parse_k_values(text: str) -> List[int]:
    vals: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    vals = sorted(set(vals))
    if not vals:
        raise argparse.ArgumentTypeError("--k_values must contain at least one integer")
    return vals


def rank_within_cutoffs(scored: pd.DataFrame) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for cutoff, g in scored.groupby("cutoff_date", sort=True):
        tmp = g.sort_values(["score", "cve"], ascending=[False, True]).copy()
        tmp["rank"] = np.arange(1, len(tmp) + 1)
        tmp["candidate_count_at_cutoff"] = len(tmp)
        parts.append(tmp)
    if not parts:
        raise ValueError("No scored rows to rank.")
    return pd.concat(parts, ignore_index=True)


def relevant_cutoffs_for_targets(snapshots: pd.DataFrame, first_events: pd.DataFrame, target_cves: Set[str]) -> Set[pd.Timestamp]:
    if not target_cves:
        return set()
    target_dates = first_events[first_events["cve"].isin(target_cves)][["cve", "kev_date"]]
    merged = snapshots[["cve", "cutoff_date"]].merge(target_dates, on="cve", how="inner")
    merged = merged[merged["cutoff_date"] < merged["kev_date"]]
    return {pd.Timestamp(x) for x in merged["cutoff_date"].unique()}


def summarize_fold_early_warning(
    ranked: pd.DataFrame,
    first_events: pd.DataFrame,
    analysis_cves: Set[str],
    k_values: List[int],
    fold_idx: int,
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
        base: Dict[str, object] = {
            "fold": int(fold_idx),
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
        }
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

    return pd.DataFrame(rows).sort_values(["fold", "kev_date", "cve"]).reset_index(drop=True)


def aggregate_per_cve(per_cve: pd.DataFrame, k_values: List[int]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "analysis_version": "cve_disjoint_kfold_early_warning",
        "analysis_rule": "For each held-out target CVE, count top-K hits only at cutoffs before the target event date.",
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


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "CVE-disjoint K-fold early-warning analysis for ThreatLens. "
            "Each fold holds out target-event CVEs from training, trains the same-cutoff transformer ranker, "
            "and counts top-K lead-time hits only for held-out target CVEs at pre-event cutoffs."
        )
    )
    ap.add_argument("--snapshots", type=str, required=True, help="CSV produced by weekly builder plus enrichment scripts; must contain cve, cutoff_date, text.")
    ap.add_argument("--events", type=str, required=True, help="Events CSV; must contain cve, event_type, event_date.")
    ap.add_argument("--window_days", type=int, default=30, help="Training label horizon. Early-warning lead-time analysis is not capped by this horizon.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_universe", choices=["target_event_cves", "all_cves"], default="target_event_cves",
                    help=(
                        "target_event_cves: only target-event CVEs are partitioned into folds and held out as targets. "
                        "all_cves: all CVEs are partitioned; each fold excludes all fold CVEs from training, but only target-event CVEs are evaluated."
                    ))
    ap.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--encoder_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pairs", type=int, default=50000, help="Same-cutoff positive/negative pairs sampled per epoch")
    ap.add_argument("--batch_size", type=int, default=8, help="Training batch size in positive-negative pairs; actual text batch is 2x this.")
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--target_event_type", type=str, default="KEV", help="Event type used as the lead-time endpoint; default is KEV.")
    ap.add_argument("--k_values", type=parse_k_values, default=parse_k_values("10,20,50,100"))
    ap.add_argument("--score_scope", choices=["relevant_cutoffs", "full_snapshots"], default="relevant_cutoffs",
                    help=(
                        "relevant_cutoffs scores full weekly candidate pools only for cutoffs where this fold has at least one pre-event target snapshot. "
                        "full_snapshots scores every snapshot row in every fold; this is slower."
                    ))
    ap.add_argument("--out_csv", type=str, default="Data/early_warning_kfold_per_cve.csv")
    ap.add_argument("--out_json", type=str, default="Data/early_warning_kfold_summary.json")
    ap.add_argument("--out_fold_csv", type=str, default="Data/early_warning_kfold_per_fold.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch device: {device}")
    print(f"Transformer model: {args.model_name}")
    print(f"K-fold early-warning analysis: folds={args.folds}, fold_universe={args.fold_universe}")
    print("Training pair sampling: same cutoff date")
    print("Early-warning rule: cutoff_date < target event date; no 30-day lead-time cap")

    snapshots = load_snapshots(Path(args.snapshots))
    events = load_events(Path(args.events))
    labeled = build_labels(snapshots, events, args.window_days)
    first_events = first_target_events(events, args.target_event_type)
    target_cves_available = target_cves_with_pre_event_snapshots(snapshots, first_events)

    if not target_cves_available:
        raise ValueError("No target-event CVEs have pre-event snapshots in the snapshots file.")

    if args.fold_universe == "target_event_cves":
        fold_source_cves = sorted(target_cves_available)
    else:
        fold_source_cves = sorted(set(labeled["cve"].unique()) | set(target_cves_available))

    folds = make_folds(fold_source_cves, args.folds, args.seed)
    print(f"Target-event CVEs with pre-event snapshots available: {len(target_cves_available)}")
    print(f"CVEs used to define folds: {len(fold_source_cves)}")
    print(f"Labeled rows available for training across folds: {len(labeled)}")

    all_per_cve: List[pd.DataFrame] = []
    fold_meta: List[Dict[str, object]] = []

    for fold_idx, heldout_cves in enumerate(folds, start=1):
        fold_seed = args.seed + fold_idx - 1
        heldout_targets = heldout_cves & target_cves_available
        if not heldout_targets:
            print(f"\n=== Fold {fold_idx}/{args.folds}: no held-out target CVEs; skipping ===")
            continue

        print(f"\n=== Fold {fold_idx}/{args.folds} ===")
        print(f"heldout fold CVEs={len(heldout_cves)} heldout target-event CVEs={len(heldout_targets)}")
        train_df = labeled[~labeled["cve"].isin(heldout_cves)].copy().reset_index(drop=True)
        describe_train_df(train_df, fold_idx)

        if train_df.empty or int(train_df["target"].sum()) == 0:
            raise ValueError(f"Fold {fold_idx} has no training data or no positive training rows.")

        model, tokenizer = train_model(
            train_df=train_df,
            model_name=args.model_name,
            pooling=args.pooling,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            epochs=args.epochs,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            pairs_per_epoch=args.pairs,
            max_length=args.max_length,
            batch_size=args.batch_size,
            seed=fold_seed,
            gradient_checkpointing=args.gradient_checkpointing,
            device=device,
        )

        if args.score_scope == "relevant_cutoffs":
            cutoffs = relevant_cutoffs_for_targets(snapshots, first_events, heldout_targets)
            if not cutoffs:
                print(f"Fold {fold_idx}: no relevant pre-event cutoffs after filtering; skipping scoring.")
                continue
            scored = snapshots[snapshots["cutoff_date"].isin(cutoffs)].copy().reset_index(drop=True)
            print(f"Scoring full candidate pools for {len(cutoffs)} relevant cutoffs: {len(scored)} rows")
        else:
            scored = snapshots.copy().reset_index(drop=True)
            print(f"Scoring all snapshot rows: {len(scored)} rows")

        scored["score"] = score_df(model, tokenizer, scored, args.max_length, args.eval_batch_size, device)
        ranked = rank_within_cutoffs(scored)
        fold_per_cve = summarize_fold_early_warning(ranked, first_events, heldout_targets, args.k_values, fold_idx)
        if fold_per_cve.empty:
            print(f"Fold {fold_idx}: no held-out target CVEs with scored pre-event snapshots.")
        else:
            print(f"Fold {fold_idx}: evaluated target CVEs with pre-event snapshots = {fold_per_cve['cve'].nunique()}")
            all_per_cve.append(fold_per_cve)

        fold_meta.append({
            "fold": fold_idx,
            "fold_seed": fold_seed,
            "heldout_fold_cves": int(len(heldout_cves)),
            "heldout_target_event_cves": int(len(heldout_targets)),
            "train_rows": int(len(train_df)),
            "train_unique_cves": int(train_df["cve"].nunique()),
            "train_positive_rows": int(train_df["target"].sum()),
            "scored_rows": int(len(scored)),
            "scored_cutoffs": int(scored["cutoff_date"].nunique()),
            "evaluated_target_cves": int(fold_per_cve["cve"].nunique()) if not fold_per_cve.empty else 0,
        })

        # Free GPU memory before the next fold.
        del model
        del tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_per_cve:
        raise ValueError("No folds produced early-warning per-CVE results.")

    per_cve = pd.concat(all_per_cve, ignore_index=True).sort_values(["fold", "kev_date", "cve"]).reset_index(drop=True)
    fold_summary_df = per_fold_summary(per_cve, args.k_values)
    summary = aggregate_per_cve(per_cve, args.k_values)
    summary.update({
        "model": "BGE-large-en-v1.5+MLP",
        "model_name": args.model_name,
        "target_event_type": args.target_event_type.upper(),
        "folds_requested": args.folds,
        "folds_completed": int(per_cve["fold"].nunique()),
        "fold_universe": args.fold_universe,
        "score_scope": args.score_scope,
        "training_window_days": args.window_days,
        "training_pair_sampling": "same_cutoff",
        "epochs": args.epochs,
        "pairs_per_epoch": args.pairs,
        "batch_size_pairs": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "target_event_cves_with_pre_event_snapshots_available": int(len(target_cves_available)),
        "fold_metadata": fold_meta,
    })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_cve.to_csv(out_csv, index=False)

    out_fold_csv = Path(args.out_fold_csv)
    out_fold_csv.parent.mkdir(parents=True, exist_ok=True)
    fold_summary_df.to_csv(out_fold_csv, index=False)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nK-fold early-warning summary")
    print(json.dumps(summary, indent=2))
    print("\nWrote:")
    print(out_csv)
    print(out_fold_csv)
    print(out_json)


if __name__ == "__main__":
    main()
