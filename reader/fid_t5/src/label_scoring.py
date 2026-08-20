"""Canonical SciFact label likelihood scoring shared by training and test."""

import torch
import torch.nn.functional as F


LABELS = ("SUPPORTS", "REFUTES", "NOINFO")
BINARY_LABELS = ("SUPPORTS", "REFUTES")


def candidate_labels(tokenizer, batch_size, device, labels=LABELS):
    encoded = tokenizer.batch_encode_plus(
        list(labels), pad_to_max_length=True, return_tensors="pt"
    )
    candidates = []
    for row_ids, row_mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        ids = row_ids.unsqueeze(0).expand(batch_size, -1).to(device)
        mask = row_mask.unsqueeze(0).expand(batch_size, -1).bool().to(device)
        # transformers 4.30 T5 uses labels.view() internally.
        candidates.append(ids.masked_fill(~mask, -100).contiguous())
    return candidates


def score_candidate_labels(model, tokenizer, input_ids, attention_mask,
                           reduction="sum", labels=LABELS):
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")
    scores = []
    for candidate in candidate_labels(
        tokenizer, input_ids.size(0), input_ids.device, labels=labels
    ):
        target_labels = candidate
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_labels,
        )
        logits = outputs.logits
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(target_labels)
        sequence_loss = token_loss.sum(dim=1)
        if reduction == "mean":
            sequence_loss = sequence_loss / target_labels.ne(-100).sum(dim=1).clamp_min(1)
        scores.append(-sequence_loss)
    return torch.stack(scores, dim=1)
