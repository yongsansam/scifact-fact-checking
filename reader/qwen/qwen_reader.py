#!/usr/bin/env python3
"""Zero-shot Qwen2.5 reader for SciFact Top-3 raw abstracts."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


LABELS = ("SUPPORTS", "REFUTES", "NOINFO")
SYSTEM_PROMPT = (
    "Classify the scientific claim using all three retrieved abstracts. "
    "Return exactly one label and no explanation: SUPPORTS, REFUTES, or NOINFO."
)


def read_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list in {path}")
    return rows


def canonical_label(row):
    raw = row.get("target", row.get("claim_label", ""))
    if not raw and row.get("answers"):
        raw = row["answers"][0]
    value = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "SUPPORT": "SUPPORTS", "SUPPORTS": "SUPPORTS",
        "CONTRADICT": "REFUTES", "CONTRADICTION": "REFUTES",
        "REFUTE": "REFUTES", "REFUTES": "REFUTES",
        "NOT_ENOUGH_INFO": "NOINFO", "NO_EVIDENCE": "NOINFO",
        "NEI": "NOINFO", "NOINFO": "NOINFO",
    }
    if value not in mapping:
        raise ValueError(f"Unknown label {raw!r} for claim {row.get('id')}")
    return mapping[value]


def row_id(row):
    return str(row.get("claim_id", row.get("id")))


def build_prompt(tokenizer, row, n_context=3, passage_length=341):
    contexts = row.get("ctxs", [])
    if len(contexts) < n_context:
        raise ValueError(
            f"Claim {row_id(row)} has {len(contexts)} contexts; expected {n_context}"
        )
    claim = str(row.get("question", row.get("claim", ""))).strip()
    slots, slot_lengths = [], []
    for rank, context in enumerate(contexts[:n_context], start=1):
        if isinstance(context, str):
            title, text = "", context
        else:
            title = str(context.get("title", ""))
            text = str(context.get("text", context.get("abstract", "")))
        slot = (
            f"Claim: {claim}\n"
            f"Document {rank} title: {title}\n"
            f"Document {rank} abstract: {text}"
        )
        token_ids = tokenizer.encode(
            slot,
            add_special_tokens=False,
            truncation=True,
            max_length=passage_length,
        )
        slots.append(tokenizer.decode(token_ids, skip_special_tokens=True))
        slot_lengths.append(len(token_ids))

    user_prompt = (
        "Read the three evidence passages and classify the claim.\n\n"
        + "\n\n".join(slots)
        + "\n\nOutput only one label: SUPPORTS, REFUTES, or NOINFO."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    return list(prompt_ids), slot_lengths


def load_model(model_path):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization,
        device_map={"": 0},
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )


def score_labels(model, tokenizer, prompt_ids):
    candidate_ids = {
        label: tokenizer.encode(label, add_special_tokens=False)
        for label in LABELS
    }
    device = next(model.parameters()).device
    scores, token_counts = [], []
    start = len(prompt_ids)
    # Score candidates sequentially.  Batching all three duplicates a long
    # Top-20 prompt and materializes [3, sequence, vocabulary] logits, which
    # exceeds a 24 GiB GPU even though the 4-bit model weights fit easily.
    for label in LABELS:
        ids = candidate_ids[label]
        sequence = prompt_ids + ids
        input_ids = torch.tensor(
            [sequence], dtype=torch.long, device=device
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        positions = torch.arange(start - 1, start + len(ids) - 1, device=device)
        targets = torch.tensor(ids, dtype=torch.long, device=device)
        # Convert only the 2-3 label positions to FP32.  Converting the full
        # sequence logits would allocate several additional gigabytes.
        label_logits = outputs.logits[0, positions, :].float()
        token_log_probs = torch.log_softmax(label_logits, dim=-1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        scores.append(float(token_log_probs.mean().item()))
        token_counts.append(len(ids))
        del outputs, label_logits, token_log_probs, input_ids, attention_mask
    return scores, token_counts


def compute_metrics(gold, predicted, labels=LABELS):
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=list(labels), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(np.mean(f1)),
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
        "confusion_matrix": confusion_matrix(
            gold, predicted, labels=list(labels)
        ).tolist(),
        "counts": {
            "examples": len(gold),
            "gold_labels": dict(Counter(gold)),
            "predicted_labels": dict(Counter(predicted)),
        },
    }


def evaluate(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model)
    model.eval()
    rows = read_rows(args.data)
    gold, predicted, score_rows = [], [], []

    for index, row in enumerate(rows, start=1):
        prompt_ids, slot_lengths = build_prompt(
            tokenizer, row, args.n_context, args.passage_length
        )
        scores, token_counts = score_labels(model, tokenizer, prompt_ids)
        prediction = LABELS[int(np.argmax(scores))]
        target = canonical_label(row)
        gold.append(target)
        predicted.append(prediction)
        score_rows.append({
            "claim_id": row_id(row),
            "prediction": prediction,
            "gold": target,
            "scores": dict(zip(LABELS, scores)),
            "label_token_counts": dict(zip(LABELS, token_counts)),
            "passage_token_counts": slot_lengths,
            "prompt_tokens_with_template": len(prompt_ids),
            "score_reduction": "mean",
        })
        if index % 25 == 0:
            print(f"evaluated {index}/{len(rows)}", flush=True)

    metrics = compute_metrics(gold, predicted)
    metrics.update({
        "task": f"scifact_claim_level_label_only_top{args.n_context}_raw_abstract",
        "model": args.model,
        "data": args.data,
        "architecture": "decoder_only",
        "evaluation_mode": "zero_shot",
        "passage_length": args.passage_length,
        "n_context": args.n_context,
        "evidence_token_budget": args.passage_length * args.n_context,
        "score_reduction": "mean",
    })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            f"{row_id(row)}\t{label}\n"
            for row, label in zip(rows, predicted)
        ),
        encoding="utf-8",
    )
    scores_output = Path(args.scores_output)
    scores_output.parent.mkdir(parents=True, exist_ok=True)
    scores_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in score_rows),
        encoding="utf-8",
    )
    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scores-output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--n-context", type=int, default=3)
    parser.add_argument("--passage-length", type=int, default=341)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 4-bit Qwen reader")
    evaluate(arguments)
