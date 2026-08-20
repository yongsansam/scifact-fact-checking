#!/usr/bin/env python3
"""Evaluate one-label-per-claim FiD predictions."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_DIR))
from prepare_scifact_fid_claim_label import LABEL_MAP, claim_label, load_rows


def normalize(text):
    value = text.strip().upper()
    return LABEL_MAP.get(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True)
    parser.add_argument("--predictions", required=True, help="<claim-id><TAB><label> file")
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--n-context", type=int, default=3)
    args = parser.parse_args()

    gold = {str(row["id"]): claim_label(row) for row in load_rows(args.claims)}
    predictions = {}
    invalid = 0
    duplicates = 0
    with Path(args.predictions).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"Expected '<id>\\t<label>' at line {line_number}")
            claim_id, raw = line.rstrip("\n").split("\t", 1)
            if claim_id in predictions:
                duplicates += 1
            label = normalize(raw)
            if label is None:
                invalid += 1
                label = "INVALID"
            predictions[claim_id] = label

    ids = list(gold)
    gold_values = [gold[claim_id] for claim_id in ids]
    pred_values = [predictions.get(claim_id, "MISSING") for claim_id in ids]
    labels = ["SUPPORTS", "REFUTES", "NOINFO"]
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_values, pred_values, labels=labels, zero_division=0
    )
    metrics = {
        "task": f"scifact_claim_level_label_only_top{args.n_context}_raw_abstract",
        "n_context": args.n_context,
        "accuracy": float(accuracy_score(gold_values, pred_values)),
        "macro_f1": float(f1_score(gold_values, pred_values, labels=labels, average="macro", zero_division=0)),
        "per_label": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(gold_values, pred_values, labels=labels).tolist(),
        "counts": {
            "gold_claims": len(gold),
            "prediction_rows": len(predictions),
            "missing_predictions": sum(claim_id not in predictions for claim_id in ids),
            "unknown_prediction_ids": sum(claim_id not in gold for claim_id in predictions),
            "invalid_labels": invalid,
            "duplicate_ids": duplicates,
            "gold_labels": dict(Counter(gold_values)),
            "predicted_labels": dict(Counter(pred_values)),
        },
    }
    output = Path(args.metrics_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
