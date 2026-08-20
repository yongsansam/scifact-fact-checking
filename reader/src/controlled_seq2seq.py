#!/usr/bin/env python3
"""Controlled T5/BART baseline and FiD-style seq2seq experiments.

The collator creates identical per-document token slots for each matched
baseline/FiD comparison. A baseline model flattens those slots before its
encoder, while a FiD model encodes them independently and concatenates the
encoder outputs for the decoder.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data import WeightedRandomSampler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .common import BINARY_LABELS, LABELS, compute_metrics, load_rows
from .modeling import score_labels
from .fid_bart import FiDBart


def load_tokenizer(model_path):
    """Load a local tokenizer, falling back for older protobuf stacks."""
    try:
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except (TypeError, ValueError):
        return AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=False
        )


class FiDFormatDataset(Dataset):
    def __init__(self, rows, n_context=3):
        self.rows = rows
        self.n_context = n_context

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class FlattenedFiDCollator:
    """Create the same 3xL FiD tokens, then flatten them for standard T5."""

    def __init__(self, tokenizer, n_context=3, passage_length=256,
                 answer_length=8):
        self.tokenizer = tokenizer
        self.n_context = n_context
        self.passage_length = passage_length
        self.answer_length = answer_length

    def __call__(self, rows):
        all_passages = []
        ids, gold = [], []
        for row in rows:
            contexts = row.get("ctxs", [])[:self.n_context]
            if len(contexts) != self.n_context:
                raise ValueError(
                    f"Claim {row.get('id')} has {len(contexts)} contexts; "
                    f"expected {self.n_context}"
                )
            question = "question: " + str(row["question"])
            passages = [
                question + " title: " + str(context.get("title", ""))
                + " context: " + str(context.get("text", ""))
                for context in contexts
            ]
            all_passages.extend(passages)
            ids.append(str(row["id"]))
            gold.append(str(row["target"]).strip().upper())

        encoded = self.tokenizer.batch_encode_plus(
            all_passages,
            max_length=self.passage_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        batch_size = len(rows)
        input_ids = encoded["input_ids"].view(
            batch_size, self.n_context * self.passage_length
        )
        attention_mask = encoded["attention_mask"].view(
            batch_size, self.n_context * self.passage_length
        ).bool()

        targets = self.tokenizer.batch_encode_plus(
            gold,
            max_length=self.answer_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        labels = targets["input_ids"]
        labels = labels.masked_fill(~targets["attention_mask"].bool(), -100)
        return {
            "ids": ids,
            "gold": gold,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_inputs(batch, architecture, n_context, passage_length, device):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if architecture == "fid":
        input_ids = input_ids.view(-1, n_context, passage_length)
        attention_mask = attention_mask.view(-1, n_context, passage_length)
    return input_ids.to(device), attention_mask.to(device)


def evaluate(model, tokenizer, rows, collator, device, batch_size,
             architecture="baseline", labels=LABELS):
    loader = DataLoader(
        FiDFormatDataset(rows, collator.n_context),
        sampler=SequentialSampler(rows),
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collator,
    )
    ids, gold, predicted, score_rows = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = model_inputs(
                batch, architecture, collator.n_context,
                collator.passage_length, device
            )
            scores = score_labels(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                reduction="sum",
                labels=labels,
            )
            predictions = [labels[index] for index in scores.argmax(dim=1).tolist()]
            ids.extend(batch["ids"])
            gold.extend(batch["gold"])
            predicted.extend(predictions)
            for claim_id, prediction, values in zip(
                batch["ids"], predictions, scores.cpu().tolist()
            ):
                score_rows.append({
                    "id": claim_id,
                    "prediction": prediction,
                    "scores": dict(zip(labels, values)),
                })
    return ids, predicted, score_rows, compute_metrics(
        gold, predicted, ids, labels=labels
    )


def save_checkpoint(model, tokenizer, directory, state):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    (directory / "training_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train(args):
    active_labels = BINARY_LABELS if args.binary_labels else LABELS
    set_seed(args.seed)
    train_rows = load_rows(args.train_data)
    dev_rows = load_rows(args.dev_data)
    tokenizer = load_tokenizer(args.model)
    model_class = FiDBart if args.architecture == "fid" else AutoModelForSeq2SeqLM
    model = model_class.from_pretrained(args.model, local_files_only=True).to(args.device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    collator = FlattenedFiDCollator(
        tokenizer, args.n_context, args.passage_length, answer_length=8
    )
    counts = Counter(str(row["target"]).strip().upper() for row in train_rows)
    weights = [1.0 / counts[str(row["target"]).strip().upper()] for row in train_rows]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=generator
    )
    loader = DataLoader(
        FiDFormatDataset(train_rows, args.n_context),
        sampler=sampler,
        batch_size=args.micro_batch_size,
        num_workers=0,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_macro_f1 = -1.0
    best_update = 0
    update = 0
    micro_step = 0
    running_loss = 0.0
    history = []
    optimizer.zero_grad(set_to_none=True)

    while update < args.max_updates:
        for batch in loader:
            model.train()
            input_ids, attention_mask = model_inputs(
                batch, args.architecture, args.n_context,
                args.passage_length, args.device
            )
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch["labels"].to(args.device),
            ).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at update={update}, micro_step={micro_step}"
                )
            (loss / args.accumulation_steps).backward()
            running_loss += loss.item()
            micro_step += 1
            if micro_step % args.accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1

            if update % args.eval_freq == 0 or update == args.max_updates:
                _, _, _, metrics = evaluate(
                    model, tokenizer, dev_rows, collator, args.device,
                    args.eval_batch_size, args.architecture,
                    labels=active_labels,
                )
                result = {
                    "update": update,
                    "train_loss": running_loss / (args.eval_freq * args.accumulation_steps),
                    "dev_accuracy": metrics["accuracy"],
                    "dev_macro_f1": metrics["macro_f1"],
                }
                history.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                running_loss = 0.0
                if metrics["macro_f1"] > best_macro_f1:
                    best_macro_f1 = metrics["macro_f1"]
                    best_update = update
                    save_checkpoint(
                        model, tokenizer, Path(args.output_dir) / "best",
                        {
                            "args": vars(args),
                            "history": history,
                            "best_update": best_update,
                            "best_macro_f1": best_macro_f1,
                            "dev_metrics": metrics,
                        },
                    )
            if update >= args.max_updates:
                break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(json.dumps({
        "args": vars(args),
        "history": history,
        "best_update": best_update,
        "best_macro_f1": best_macro_f1,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def write_evaluation(args):
    active_labels = BINARY_LABELS if args.binary_labels else LABELS
    rows = load_rows(args.data)
    tokenizer = load_tokenizer(args.model)
    model_class = FiDBart if args.architecture == "fid" else AutoModelForSeq2SeqLM
    model = model_class.from_pretrained(args.model, local_files_only=True).to(args.device)
    collator = FlattenedFiDCollator(
        tokenizer, args.n_context, args.passage_length, answer_length=8
    )
    ids, predicted, score_rows, metrics = evaluate(
        model, tokenizer, rows, collator, args.device, args.batch_size,
        args.architecture,
        labels=active_labels,
    )
    metrics.update({
        "task": (
            f"scifact_claim_level_label_only_top{args.n_context}_raw_abstract"
            if "raw_abstract" in str(args.data)
            else f"scifact_claim_level_label_only_top{args.n_context}_selected_rationales"
        ),
        "model": args.model,
        "data": args.data,
        "input_shape": [args.n_context, args.passage_length],
        "total_encoder_tokens": args.n_context * args.passage_length,
        "architecture": args.architecture,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(f"{claim_id}\t{label}\n" for claim_id, label in zip(ids, predicted)),
        encoding="utf-8",
    )
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.scores_output:
        scores_path = Path(args.scores_output)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in score_rows),
            encoding="utf-8",
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--train-data", required=True)
    train_parser.add_argument("--dev-data", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--n-context", type=int, default=3)
    train_parser.add_argument("--passage-length", type=int, default=256)
    train_parser.add_argument("--micro-batch-size", type=int, default=1)
    train_parser.add_argument("--accumulation-steps", type=int, default=8)
    train_parser.add_argument("--eval-batch-size", type=int, default=1)
    train_parser.add_argument("--max-updates", type=int, default=1000)
    train_parser.add_argument("--eval-freq", type=int, default=100)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--clip", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--gradient-checkpointing", action="store_true")
    train_parser.add_argument("--binary-labels", action="store_true")
    train_parser.add_argument(
        "--architecture", choices=("baseline", "fid"), default="baseline"
    )

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--data", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--metrics-output", required=True)
    eval_parser.add_argument("--scores-output", default=None)
    eval_parser.add_argument("--n-context", type=int, default=3)
    eval_parser.add_argument("--passage-length", type=int, default=256)
    eval_parser.add_argument("--batch-size", type=int, default=1)
    eval_parser.add_argument("--device", default="cuda")
    eval_parser.add_argument("--binary-labels", action="store_true")
    eval_parser.add_argument(
        "--architecture", choices=("baseline", "fid"), default="baseline"
    )
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        write_evaluation(args)


if __name__ == "__main__":
    main()
