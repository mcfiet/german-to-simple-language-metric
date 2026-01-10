"""
Run SBERT fine-tuning for a configurable train size and log validation metrics.

Example:
    python scripts/run_sbert_learning_curve.py \\
        --train-size 1000 \\
        --subset-output data/train_subsets/train_subset_1000.csv \\
        --csv-path data/learning_curve_sizes.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default="data/synthetic_normal_2_labeled.csv",
        help="CSV with columns 'sentence' and 'label'.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of rows to draw (stratified) before the train/val/test split. "
        "If omitted, use the full dataset.",
    )
    parser.add_argument(
        "--subset-output",
        default=None,
        help="Optional path to save the sampled subset for reproducibility.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="HF model id for the encoder.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Val split size (applied on the train fold).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze the transformer and train only the head.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Force device, e.g. 'cuda' or 'cpu'. Defaults to auto-detect.")
    parser.add_argument(
        "--csv-path",
        default=None,
        help="If set, append results as a row to this CSV (train_size, subset_path, val_precision, val_recall, val_f1, val_bal_acc).",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"sentence", "label"}.issubset(df.columns):
        raise ValueError(f"Expected columns 'sentence' and 'label' in {path}")
    df["sentence"] = df["sentence"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def sample_subset(df: pd.DataFrame, train_size: int | None, seed: int) -> pd.DataFrame:
    if train_size is None or train_size >= len(df):
        return df.copy()
    subset, _ = train_test_split(df, train_size=train_size, random_state=seed, stratify=df["label"])
    return subset.reset_index(drop=True)


def build_dataloaders(
    model: SentenceTransformer,
    X_train: pd.Series,
    y_train: pd.Series,
    batch_size: int,
) -> DataLoader:
    train_examples = [InputExample(texts=[s], label=int(l)) for s, l in zip(X_train, y_train)]

    def collate_fn(batch):
        sentences = [ex.texts[0] if isinstance(ex.texts, (list, tuple)) else ex.texts for ex in batch]
        labels = torch.tensor([int(ex.label) for ex in batch], dtype=torch.long)
        features = model.tokenize(sentences)
        return features, labels

    return DataLoader(train_examples, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)


def eval_split(
    model: SentenceTransformer,
    classifier: torch.nn.Module,
    sentences,
    labels,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    classifier.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i : i + batch_size]
            feats = model.tokenize(batch_sents)
            feats = {k: v.to(device) for k, v in feats.items()}
            emb = model(feats)["sentence_embedding"]
            logits = classifier(emb)
            all_logits.append(logits.cpu())
    logits = torch.cat(all_logits)
    preds = logits.argmax(dim=1).numpy()
    labels = np.array(labels)
    acc = accuracy_score(labels, preds)
    bal = balanced_accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"acc": acc, "bal": bal, "precision": precision, "recall": recall, "f1": f1}


def train(
    model: SentenceTransformer,
    classifier: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size: int,
    device: torch.device,
    epochs: int,
    patience: int = 3,
) -> Tuple[SentenceTransformer, torch.nn.Module, Dict[str, float]]:
    best_state = None
    best_val_loss = float("inf")
    epochs_no_improve = 0

    def forward_batch(features):
        features = {k: v.to(device) for k, v in features.items()}
        emb = model(features)["sentence_embedding"]
        logits = classifier(emb)
        return logits

    for epoch in range(epochs):
        model.train()
        classifier.train()
        total_loss = 0.0
        for features, labels in dataloader:
            labels = labels.to(device)
            logits = forward_batch(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation loss for early stopping
        model.eval()
        classifier.eval()
        val_logits = []
        val_targets = []
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_sents = X_val[i : i + batch_size]
                batch_labels = torch.tensor(y_val[i : i + batch_size], device=device)
                feats = model.tokenize(batch_sents)
                logits = forward_batch(feats)
                val_logits.append(logits.cpu())
                val_targets.append(batch_labels.cpu())
        val_logits = torch.cat(val_logits)
        val_targets = torch.cat(val_targets)
        val_loss = criterion(val_logits, val_targets).item()

        val_metrics = eval_split(model, classifier, X_val, y_val, batch_size, device)
        print(
            f"Epoch {epoch + 1}/{epochs} loss={total_loss / len(dataloader):.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['acc']:.3f} "
            f"val_bal={val_metrics['bal']:.3f} val_f1={val_metrics['f1']:.3f}"
        )

        improved = val_loss < (best_val_loss - 1e-4)
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "classifier": {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()},
            }
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    if best_state:
        model.load_state_dict(best_state["model"])
        classifier.load_state_dict(best_state["classifier"])

    final_val_metrics = eval_split(model, classifier, X_val, y_val, batch_size, device)
    return model, classifier, final_val_metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seeds(args.seed)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset at {data_path}")

    df = load_dataset(data_path)
    subset_df = sample_subset(df, args.train_size, args.seed)
    print(f"Total rows: {len(df)}; Using subset: {len(subset_df)}")
    print(subset_df["label"].value_counts(normalize=True))

    if args.subset_output:
        subset_path = Path(args.subset_output)
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_df.to_csv(subset_path, index=False)
    else:
        subset_path = data_path

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        subset_df["sentence"],
        subset_df["label"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=subset_df["label"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=args.val_size, random_state=args.seed, stratify=y_train_full
    )

    model = SentenceTransformer(args.model_name, device=device)
    if args.freeze_encoder:
        transformer = model[0]
        for param in transformer.auto_model.parameters():
            param.requires_grad = False
        print("Encoder frozen; only classifier head will be trained.")

    classifier = torch.nn.Linear(model.get_sentence_embedding_dimension(), 2).to(device)
    optimizer = AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = build_dataloaders(model, X_train, y_train, args.batch_size)
    model, classifier, val_metrics = train(
        model=model,
        classifier=classifier,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        X_train=list(X_train),
        y_train=list(map(int, y_train)),
        X_val=list(X_val),
        y_val=list(map(int, y_val)),
        batch_size=args.batch_size,
        device=device,
        epochs=args.epochs,
    )

    test_metrics = eval_split(model, classifier, list(X_test), list(map(int, y_test)), args.batch_size, device)

    summary = {
        "train_size": len(subset_df),
        "subset_path": str(subset_path),
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_bal_acc": val_metrics["bal"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_bal_acc": test_metrics["bal"],
    }
    print("\n== Final metrics ==")
    print(json.dumps(summary, indent=2))

    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        row = {
            "train_size": summary["train_size"],
            "subset_path": summary["subset_path"],
            "val_precision": summary["val_precision"],
            "val_recall": summary["val_recall"],
            "val_f1": summary["val_f1"],
            "val_bal_acc": summary["val_bal_acc"],
        }
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=write_header, index=False)
        print(f"Appended row to {csv_path}")


if __name__ == "__main__":
    main()
