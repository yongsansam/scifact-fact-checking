#!/usr/bin/env python3
"""Shared utilities for SciFact label-only seq2seq readers."""

import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support


LABELS = ("SUPPORTS", "REFUTES", "NOINFO")
BINARY_LABELS = ("SUPPORTS", "REFUTES")
LABEL_MAP = {
    "SUPPORT": "SUPPORTS",
    "SUPPORTS": "SUPPORTS",
    "CONTRADICT": "REFUTES",
    "CONTRADICTS": "REFUTES",
    "CONTRADICTION": "REFUTES",
    "REFUTE": "REFUTES",
    "REFUTES": "REFUTES",
    "NO_EVIDENCE": "NOINFO",
    "NOT_ENOUGH_INFO": "NOINFO",
    "NEI": "NOINFO",
    "NOINFO": "NOINFO",
}


def load_rows(path):
    path = Path(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def normalize_label(value):
    return LABEL_MAP.get(str(value).strip().upper())


def claim_label(claim):
    labels = set()
    for evidence_sets in (claim.get("evidence") or {}).values():
        for evidence in evidence_sets or []:
            label = normalize_label(evidence.get("label", ""))
            if label is None:
                raise ValueError(
                    f"Unknown label {evidence.get('label')!r} in claim {claim['id']}"
                )
            if label != "NOINFO":
                labels.add(label)
    if len(labels) > 1:
        raise ValueError(f"Claim {claim['id']} has conflicting labels: {sorted(labels)}")
    return next(iter(labels), "NOINFO")


def compute_metrics(gold, predicted, ids=None, labels=LABELS):
    if len(gold) != len(predicted):
        raise ValueError("Gold and prediction lengths differ")
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=labels, zero_division=0
    )
    result = {
        "task": "scifact_claim_level_label_only_top3_selected_rationales",
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(
            f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "per_label": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix_labels": list(labels),
        "confusion_matrix": confusion_matrix(gold, predicted, labels=labels).tolist(),
        "counts": {
            "examples": len(gold),
            "gold_labels": dict(Counter(gold)),
            "predicted_labels": dict(Counter(predicted)),
        },
    }
    if ids is not None:
        result["ids_evaluated"] = len(ids)
    return result
