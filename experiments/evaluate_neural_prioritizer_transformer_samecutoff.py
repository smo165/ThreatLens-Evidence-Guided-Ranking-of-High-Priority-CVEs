# evaluate_neural_prioritizer_transformer.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

POSITIVE_EVENTS: Set[str] = {"KEV", "INTHEWILD", "METASPLOIT", "EXPLOITDB"}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_snapshots(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cve", "cutoff_date", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Snapshots CSV missing columns: {sorted(missing)}")
    df = df.copy()
    # Keep this consistent with the existing CNN evaluator: do not alter CVE strings.
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
    # Keep this consistent with the existing CNN evaluator: only normalize event_type.
    df["event_type"] = df["event_type"].astype(str).str.upper()
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True)
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
        rows.append({"cve": row.cve, "cutoff_date": t, "text": row.text, "target": 1 if future_positive else 0})

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

    # CVE-disjoint filtering gives a stricter generalization test.
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


def build_cutoff_negative_lookup(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[pd.Timestamp, np.ndarray]]:
    labels = df["target"].to_numpy(dtype=int)
    cutoffs = df["cutoff_date"].to_numpy()
    neg_by_cutoff: Dict[pd.Timestamp, np.ndarray] = {}
    for cutoff in pd.unique(df["cutoff_date"]):
        idx = np.where((cutoffs == np.datetime64(cutoff)) & (labels == 0))[0]
        if len(idx) > 0:
            neg_by_cutoff[pd.Timestamp(cutoff)] = idx
    return labels, neg_by_cutoff


def sample_same_cutoff_pairs(rng: np.random.Generator, df: pd.DataFrame, num_pairs: int):
    """Sample positive-negative pairs where both rows come from the same cutoff date.

    Positives are sampled uniformly from eligible positive rows. For each positive row,
    a negative row is sampled from the same cutoff date. This directly matches the
    weekly ranking task: compare future-KEV and non-future-KEV snapshots available at
    the same weekly cutoff.
    """
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
        raise ValueError("Need at least one cutoff with both positive and negative examples.")

    ia = rng.choice(np.array(eligible_pos, dtype=int), size=num_pairs, replace=True)
    ib = np.empty(num_pairs, dtype=int)
    for j, pos_idx in enumerate(ia):
        cutoff = pd.Timestamp(cutoffs[int(pos_idx)])
        ib[j] = rng.choice(neg_by_cutoff[cutoff])
    return ia, ib


def pairwise_logistic_loss(score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(-(score_pos - score_neg)).mean()


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
    def __init__(self, model_name: str, hidden_dim: int = 256, dropout: float = 0.1, pooling: str = "cls") -> None:
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


def tokenize_texts(tokenizer, texts: List[str], max_length: int, device: torch.device):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}


@torch.no_grad()
def score_df(model: TransformerPrioritizer, tokenizer, df: pd.DataFrame, batch_size: int, max_length: int, device: torch.device) -> np.ndarray:
    model.eval()
    texts = df["text"].tolist()
    scores: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        toks = tokenize_texts(tokenizer, texts[start:start + batch_size], max_length=max_length, device=device)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            s = model(**toks)
        scores.append(s.detach().cpu().float().numpy())
    return np.concatenate(scores, axis=0) if scores else np.array([], dtype=float)


def precision_recall_f1(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.0):
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


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


def same_cutoff_pair_accuracy(scores: np.ndarray, df: pd.DataFrame, pairs: int = 5000, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    ia, ib = sample_same_cutoff_pairs(rng, df.reset_index(drop=True), min(pairs, max(100, len(df))))
    return float(np.mean(scores[ia] > scores[ib]))


def cutoff_ranking_metrics(scores: np.ndarray, df: pd.DataFrame, k_values: Tuple[int, ...] = (10, 20, 50, 100)) -> Dict[str, float | int]:
    """Compute top-K metrics within each cutoff, then report macro and micro averages.

    Macro averages treat each cutoff equally. Micro averages aggregate hits/slots and
    hits/positives across cutoffs. These are the task-aligned top-K metrics for weekly
    forecasting because each cutoff is a separate ranking problem.
    """
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
            prec = hits / kk
            rec = hits / n_pos if n_pos else 0.0
            macro_p[k].append(prec)
            macro_r[k].append(rec)
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


def evaluate_split(name: str, model: TransformerPrioritizer, tokenizer, df: pd.DataFrame, batch_size: int, max_length: int, device: torch.device) -> Dict[str, float | int]:
    df = df.reset_index(drop=True)
    scores = score_df(model, tokenizer, df, batch_size=batch_size, max_length=max_length, device=device)
    labels = df["target"].to_numpy(dtype=int)
    p, r, f1 = precision_recall_f1(scores, labels, threshold=0.0)
    metrics: Dict[str, float | int] = {
        "rows": len(df),
        "unique_cves": int(df["cve"].nunique()),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "pair_acc": same_cutoff_pair_accuracy(scores, df, pairs=5000, seed=7),
        "global_precision": p,
        "global_recall": r,
        "global_f1": f1,
        "global_pr_auc": pr_auc(scores, labels),
    }
    metrics.update(cutoff_ranking_metrics(scores, df))
    print(f"\n{name} metrics")
    for k, v in metrics.items():
        print(f"{k:>24}: {v}")
    return metrics


def train_epoch(
    model: TransformerPrioritizer,
    tokenizer,
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    opt: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    pairs_per_epoch: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> float:
    model.train()
    train_df = train_df.reset_index(drop=True)
    texts = train_df["text"].tolist()
    ia, ib = sample_same_cutoff_pairs(rng, train_df, pairs_per_epoch)
    losses: List[float] = []

    for start in range(0, len(ia), batch_size):
        pos_idx = ia[start:start + batch_size]
        neg_idx = ib[start:start + batch_size]
        batch_texts = [texts[int(i)] for i in pos_idx] + [texts[int(j)] for j in neg_idx]
        toks = tokenize_texts(tokenizer, batch_texts, max_length=max_length, device=device)
        n = len(pos_idx)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            scores = model(**toks)
            loss = pairwise_logistic_loss(scores[:n], scores[n:])

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate fine-tuned transformer text encoder + MLP CVE prioritizer with same-cutoff pair sampling."
    )
    ap.add_argument("--snapshots", type=str, required=True)
    ap.add_argument("--events", type=str, required=True)
    ap.add_argument("--window_days", type=int, default=30)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--encoder_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pairs", type=int, default=50000, help="Same-cutoff positive/negative pairs sampled per epoch")
    ap.add_argument("--batch_size", type=int, default=8, help="Number of same-cutoff pairs per update; actual text batch is 2x this")
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch device: {device}")
    print(f"Transformer model: {args.model_name}")
    print("Pair sampling: same cutoff date")
    print("Top-K evaluation: within each cutoff date, then macro/micro averaged")

    snapshots = load_snapshots(Path(args.snapshots))
    events = load_events(Path(args.events))
    labeled = build_labels(snapshots, events, args.window_days)
    train_df, val_df, test_df = temporal_disjoint_split(labeled, args.train_frac, args.val_frac)

    print("Split summary")
    print(f"train rows={len(train_df)} unique_cves={train_df['cve'].nunique()} positives={int(train_df['target'].sum())}")
    print(f"val   rows={len(val_df)} unique_cves={val_df['cve'].nunique()} positives={int(val_df['target'].sum())}")
    print(f"test  rows={len(test_df)} unique_cves={test_df['cve'].nunique()} positives={int(test_df['target'].sum())}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = TransformerPrioritizer(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)

    if args.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        print("Enabled transformer gradient checkpointing.")

    opt = optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.scorer.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            rng=rng,
            opt=opt,
            scaler=scaler,
            pairs_per_epoch=args.pairs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
        if ep == 1 or ep == args.epochs or ep % 2 == 0:
            print(f"ep={ep:4d}  loss={loss:.4f}")

    val_metrics = evaluate_split("VAL", model, tokenizer, val_df, args.eval_batch_size, args.max_length, device)
    test_metrics = evaluate_split("TEST", model, tokenizer, test_df, args.eval_batch_size, args.max_length, device)

    print("\nJSON metrics")
    print(json.dumps({
        "model": "BGE-large-en-v1.5+MLP",
        "model_name": args.model_name,
        "window_days": args.window_days,
        "pair_sampling": "same_cutoff",
        "topk_evaluation": "per_cutoff_macro_and_micro",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "epochs": args.epochs,
        "pairs_per_epoch": args.pairs,
        "batch_size_pairs": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "val": val_metrics,
        "test": test_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
