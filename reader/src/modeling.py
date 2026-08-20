#!/usr/bin/env python3
"""Dataset, collation, and constrained label scoring for T5/BART."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .common import LABELS


class ClaimDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class Seq2SeqCollator:
    def __init__(self, tokenizer, max_input_length=1024, max_target_length=8):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, rows):
        encoded = self.tokenizer(
            [row["input"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                [row["target"] for row in rows],
                padding=True,
                truncation=True,
                max_length=self.max_target_length,
                return_tensors="pt",
            )["input_ids"]
        targets[targets == self.tokenizer.pad_token_id] = -100
        encoded["labels"] = targets
        encoded["ids"] = [str(row["id"]) for row in rows]
        encoded["gold"] = [row["target"] for row in rows]
        return encoded


def move_model_inputs(batch, device, include_labels=True):
    keys = ["input_ids", "attention_mask"]
    if include_labels:
        keys.append("labels")
    return {key: batch[key].to(device) for key in keys if key in batch}


def encoded_candidates(tokenizer, device, labels=LABELS):
    with tokenizer.as_target_tokenizer():
        encoded = tokenizer(
            list(labels), padding=True, return_tensors="pt"
        )
    labels = encoded["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return labels.to(device)


def score_labels(model, tokenizer, input_ids, attention_mask, reduction="sum",
                 labels=LABELS):
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be sum or mean")
    candidates = encoded_candidates(tokenizer, input_ids.device, labels=labels)
    scores = []
    batch_size = input_ids.size(0)
    for candidate in candidates:
        # transformers 4.30 T5 uses labels.view() internally, so an expanded
        # zero-stride tensor must be materialized first.
        labels = candidate.unsqueeze(0).expand(batch_size, -1).contiguous()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = output.logits
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        sequence_loss = losses.sum(dim=1)
        if reduction == "mean":
            sequence_loss = sequence_loss / labels.ne(-100).sum(dim=1).clamp_min(1)
        scores.append(-sequence_loss)
    return torch.stack(scores, dim=1)
