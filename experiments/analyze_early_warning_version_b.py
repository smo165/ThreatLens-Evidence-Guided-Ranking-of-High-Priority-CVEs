# analyze_early_warning_version_b.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

# Keep this consistent with the evaluator's label-building logic.
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
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.25) -> None:
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

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class TransformerPrioritizer(nn.Module):
    """BGE/transformer text encoder + MLP scoring head.

    This matches the fine-tuned transformer evaluator pattern:
      text -> tokenizer -> transformer encoder -> pooled embedding -> MLP score
    """

    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        encoder_dim = int(self.encoder.config.hidden_size)
        self.scorer = ScoreHead(input_dim=encoder_dim, hidden_dim=hidden_dim, dropout=dropout)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        if self.pooling == "cls":
            return last[:, 0]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(last)
            return (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.scorer(self.encode(input_ids=input_ids, attention_mask=attention_mask))


def pairwise_logistic_loss(score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(-(score_pos - score_neg)).mean()


# -----------------------------
# Data loading / labeling / split
# -----------------------------

def load_snapshots(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cve", "cutoff_date", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Snapshots CSV missing columns: {sorted(missing)}")
    df = df.copy()
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
    df["event_type"] = df["event_type"].astype(str).str.upper()
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_date"]).reset_index(drop=True)
    return df


def build_labels(snapshots: pd.DataFrame, events: pd.DataFrame, window_days: int) -> pd.DataFrame:
    grouped: Dict[str, List[Tuple[str, pd.Timestamp]]] = {}
    for row in events.itertuples(index=False):
        grouped.setdefault(row.cve, []).append((row.event_type, row.event_date))

    rows = []
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
        raise ValueError("No labeled rows remained after filtering.")
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

    # Match the evaluator's stricter CVE-disjoint filtering.
    train_cves = set(train_df["cve"].unique())
    val_df = val_df[~val_df["cve"].isin(train_cves)].copy()
    train_val_cves = train_cves | set(val_df["cve"].unique())
    test_df = test_df[~test_df["cve"].isin(train_val_cves)].copy()

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Empty split after temporal + CVE-disjoint filtering.")
    return train_df, val_df, test_df, train_cutoffs, val_cutoffs, test_cutoffs


# -----------------------------
# Same-cutoff pair sampling
# -----------------------------

def build_cutoff_pair_index(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, pd.Timestamp]]:
    groups: List[Tuple[np.ndarray, np.ndarray, pd.Timestamp]] = []
    for cutoff, g in df.groupby("cutoff_date", sort=True):
        idx = g.index.to_numpy(dtype=int)
        labels = g["target"].to_numpy(dtype=int)
        pos = idx[labels == 1]
        neg = idx[labels == 0]
        if len(pos) > 0 and len(neg) > 0:
            groups.append((pos, neg, cutoff))
    if not groups:
        raise ValueError("No cutoff has both positives and negatives for same-cutoff pair sampling.")
    return groups


def sample_same_cutoff_pairs(
    rng: np.random.Generator,
    groups: List[Tuple[np.ndarray, np.ndarray, pd.Timestamp]],
    num_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Sample cutoffs uniformly among cutoffs with at least one positive and one negative.
    # Then sample a positive and negative row from that cutoff.
    gidx = rng.integers(0, len(groups), size=num_pairs)
    pos_out = np.empty(num_pairs, dtype=int)
    neg_out = np.empty(num_pairs, dtype=int)
    for i, gi in enumerate(gidx):
        pos, neg, _ = groups[int(gi)]
        pos_out[i] = int(rng.choice(pos))
        neg_out[i] = int(rng.choice(neg))
    return pos_out, neg_out


# -----------------------------
# Tokenization / scoring / training
# -----------------------------

def tokenize_batch(tokenizer, texts: List[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    toks = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in toks.items() if k in {"input_ids", "attention_mask"}}


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
    model = TransformerPrioritizer(model_name=model_name, hidden_dim=hidden_dim, dropout=dropout, pooling=pooling).to(device)

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


# -----------------------------
# Version B early-warning analysis
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


def first_target_events(events: pd.DataFrame, target_event_type: str) -> pd.DataFrame:
    target = target_event_type.upper().strip()
    df = events[events["event_type"] == target].copy()
    if df.empty:
        raise ValueError(f"No events found for target_event_type={target!r}")
    df = df.sort_values(["cve", "event_date"]).drop_duplicates("cve", keep="first")
    return df.rename(columns={"event_date": "kev_date", "event_type": "kev_event_type"}).reset_index(drop=True)


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


def select_analysis_cves(scope: str, first_kev: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Set[str]:
    if scope == "test_cves":
        return set(test_df["cve"].unique()) & set(first_kev["cve"].unique())
    if scope == "val_cves":
        return set(val_df["cve"].unique()) & set(first_kev["cve"].unique())
    if scope == "val_test_cves":
        return (set(val_df["cve"].unique()) | set(test_df["cve"].unique())) & set(first_kev["cve"].unique())
    if scope == "all_event_cves":
        return set(first_kev["cve"].unique())
    raise ValueError(f"Unknown analysis_scope: {scope}")


def summarize_version_b(
    ranked: pd.DataFrame,
    first_kev: pd.DataFrame,
    analysis_cves: Set[str],
    k_values: List[int],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    # Version B rule: any pre-KEV weekly snapshot is valid, regardless of whether it is within the 30-day horizon.
    merged = ranked.merge(first_kev[["cve", "kev_date", "kev_event_type"]], on="cve", how="inner")
    merged = merged[merged["cve"].isin(analysis_cves)].copy()
    merged = merged[merged["cutoff_date"] < merged["kev_date"]].copy()
    merged["days_before_kev"] = (merged["kev_date"].dt.normalize() - merged["cutoff_date"].dt.normalize()).dt.days.astype(int)

    rows: List[Dict[str, object]] = []
    for cve, g in merged.groupby("cve", sort=True):
        g = g.sort_values("cutoff_date")
        best_row = g.loc[g["rank"].idxmin()]
        base: Dict[str, object] = {
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

    per_cve = pd.DataFrame(rows).sort_values(["kev_date", "cve"]).reset_index(drop=True) if rows else pd.DataFrame()

    summary: Dict[str, object] = {
        "analysis_version": "B_all_pre_kev_snapshots",
        "version_b_rule": "cutoff_date < KEV_DATE; no 30-day horizon restriction for lead-time analysis",
        "analysis_cves_requested": int(len(analysis_cves)),
        "analysis_cves_with_pre_kev_snapshots": int(per_cve["cve"].nunique()) if not per_cve.empty else 0,
        "ranked_rows_scored": int(len(ranked)),
        "ranked_cutoffs_scored": int(ranked["cutoff_date"].nunique()),
        "k_values": k_values,
    }

    for k in k_values:
        hit_col = f"hit_top_{k}"
        day_col = f"days_before_kev_top_{k}"
        if per_cve.empty:
            vals = pd.Series(dtype=float)
            hits = 0
            hit_rate = 0.0
        else:
            hits = int(per_cve[hit_col].sum())
            denom = int(len(per_cve))
            hit_rate = float(hits / denom) if denom else 0.0
            vals = pd.to_numeric(per_cve[day_col], errors="coerce").dropna()
        summary[f"top_{k}"] = {
            "hits": hits,
            "hit_rate_among_cves_with_pre_kev_snapshots": hit_rate,
            "median_days_before_kev": float(vals.median()) if len(vals) else None,
            "mean_days_before_kev": float(vals.mean()) if len(vals) else None,
            "min_days_before_kev": int(vals.min()) if len(vals) else None,
            "max_days_before_kev": int(vals.max()) if len(vals) else None,
            "p25_days_before_kev": float(vals.quantile(0.25)) if len(vals) else None,
            "p75_days_before_kev": float(vals.quantile(0.75)) if len(vals) else None,
        }

    return per_cve, summary


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train the same-cutoff BGE-large ThreatLens ranker, then perform Version B early-warning analysis: "
            "for each KEV CVE, find the earliest pre-KEV cutoff where it appears in top-K."
        )
    )
    ap.add_argument("--snapshots", type=str, required=True)
    ap.add_argument("--events", type=str, required=True)
    ap.add_argument("--window_days", type=int, default=30, help="Training label horizon. Version B analysis is not restricted to this horizon.")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--encoder_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pairs", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=64, help="Training batch size in positive-negative pairs; actual text batch is 2x this.")
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--target_event_type", type=str, default="KEV", help="Event type used as the lead-time endpoint; default is KEV.")
    ap.add_argument(
        "--analysis_scope",
        choices=["test_cves", "val_cves", "val_test_cves", "all_event_cves"],
        default="test_cves",
        help=(
            "Which CVEs to include in the lead-time summary. "
            "test_cves is most defensible; all_event_cves is exploratory and can include CVEs seen during training."
        ),
    )
    ap.add_argument("--k_values", type=parse_k_values, default=parse_k_values("10,20,50,100"))
    ap.add_argument("--out_csv", type=str, default="Data/early_warning_version_b_per_cve.csv")
    ap.add_argument("--out_json", type=str, default="Data/early_warning_version_b_summary.json")
    ap.add_argument("--out_ranked_csv", type=str, default="", help="Optional row-level ranked scores CSV; can be large.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch device: {device}")
    print(f"Transformer model: {args.model_name}")
    print("Training pair sampling: same cutoff date")
    print("Early-warning analysis: Version B/all pre-KEV snapshots")
    print("Version B rule: cutoff_date < KEV_DATE, no 30-day lead-time cap")

    snapshots = load_snapshots(Path(args.snapshots))
    events = load_events(Path(args.events))
    labeled = build_labels(snapshots, events, args.window_days)
    train_df, val_df, test_df, train_cutoffs, val_cutoffs, test_cutoffs = temporal_disjoint_split(labeled, args.train_frac, args.val_frac)

    print("Split summary")
    print(f"train rows={len(train_df)} unique_cves={train_df['cve'].nunique()} positives={int(train_df['target'].sum())}")
    print(f"val   rows={len(val_df)} unique_cves={val_df['cve'].nunique()} positives={int(val_df['target'].sum())}")
    print(f"test  rows={len(test_df)} unique_cves={test_df['cve'].nunique()} positives={int(test_df['target'].sum())}")

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
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        device=device,
    )

    first_kev = first_target_events(events, args.target_event_type)
    analysis_cves = select_analysis_cves(args.analysis_scope, first_kev, train_df, val_df, test_df)
    if not analysis_cves:
        raise ValueError(f"No target-event CVEs found for analysis_scope={args.analysis_scope!r}.")

    print(f"Target event type for lead time: {args.target_event_type.upper()}")
    print(f"Analysis scope: {args.analysis_scope}")
    print(f"Analysis CVEs with target event: {len(analysis_cves)}")

    # Score the full snapshot table, then rank within each weekly cutoff. This is what lets Version B
    # look back across all pre-KEV snapshots, not only the 30-day positive-label window.
    scored = snapshots.copy()
    print(f"Scoring all snapshot rows for Version B ranking: {len(scored)} rows")
    scored["score"] = score_df(model, tokenizer, scored, args.max_length, args.eval_batch_size, device)
    ranked = rank_within_cutoffs(scored)

    per_cve, summary = summarize_version_b(ranked, first_kev, analysis_cves, args.k_values)
    summary.update({
        "model": "BGE-large-en-v1.5+MLP",
        "model_name": args.model_name,
        "training_window_days": args.window_days,
        "training_pair_sampling": "same_cutoff",
        "epochs": args.epochs,
        "pairs_per_epoch": args.pairs,
        "batch_size_pairs": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "analysis_scope": args.analysis_scope,
        "target_event_type": args.target_event_type.upper(),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_cve.to_csv(out_csv, index=False)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.out_ranked_csv:
        ranked_out = Path(args.out_ranked_csv)
        ranked_out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["cve", "cutoff_date", "score", "rank", "candidate_count_at_cutoff"]
        ranked[cols].to_csv(ranked_out, index=False)

    print("\nVersion B early-warning summary")
    print(json.dumps(summary, indent=2))
    print("\nWrote:")
    print(out_csv)
    print(out_json)
    if args.out_ranked_csv:
        print(args.out_ranked_csv)


if __name__ == "__main__":
    main()
