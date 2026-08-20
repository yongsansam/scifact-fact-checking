#!/usr/bin/env python3
"""Predict one canonical SciFact label with FiD decoder likelihoods.

Supports both a fine-tuned FiD checkpoint and a pretrained T5/FLAN-T5 model
wrapped as FiD without SciFact training.
"""

import argparse
import json
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader, SequentialSampler

import src.data
import src.model
from src.label_scoring import BINARY_LABELS, LABELS, score_candidate_labels


def predict_labels(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    reduction="sum",
):
    """Choose the highest decoder score for each example."""
    scores = score_candidate_labels(
        model,
        tokenizer,
        input_ids,
        attention_mask,
        reduction=reduction,
    )
    best = scores.argmax(dim=1).tolist()
    return [LABELS[index] for index in best]


def initialize_pretrained_fid(model_name, device, local_files_only=False):
    """Wrap pretrained T5 weights in FiD without task fine-tuning."""
    original = transformers.T5ForConditionalGeneration.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = src.model.FiDT5(original.config)
    model.load_t5(original.state_dict())
    del original
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data", required=True)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-path", help="Fine-tuned FiD checkpoint")
    model_group.add_argument(
        "--model-name",
        help="Pretrained T5 model wrapped as FiD without SciFact training",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-size", default="base")
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Tokenizer for a fine-tuned checkpoint (for example t5-base)",
    )
    parser.add_argument("--n-context", type=int, default=3)
    parser.add_argument("--binary-labels", action="store_true")
    parser.add_argument("--text-maxlength", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact Hugging Face; require a complete local cache",
    )
    parser.add_argument(
        "--score-reduction",
        choices=("sum", "mean"),
        default="sum",
        help="Use mean for zero-shot models to remove label token-length bias",
    )
    parser.add_argument(
        "--scores-output",
        default=None,
        help="Optional JSONL with all three candidate scores",
    )
    args = parser.parse_args()
    active_labels = BINARY_LABELS if args.binary_labels else LABELS

    tokenizer_source = args.tokenizer_name or args.model_name or ("t5-" + args.model_size)
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=args.local_files_only,
    )
    examples = src.data.load_data(args.eval_data)
    if args.limit is not None:
        examples = examples[:args.limit]
    dataset = src.data.Dataset(examples, args.n_context)
    collator = src.data.Collator(args.text_maxlength, tokenizer, answer_maxlength=8)
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    if args.model_name:
        model = initialize_pretrained_fid(
            args.model_name,
            args.device,
            local_files_only=args.local_files_only,
        )
    else:
        model = src.model.FiDT5.from_pretrained(args.model_path).to(args.device)
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores_writer = None
    if args.scores_output:
        scores_path = Path(args.scores_output)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_writer = scores_path.open("w", encoding="utf-8")
    with output.open("w", encoding="utf-8") as writer, torch.no_grad():
        for batch in dataloader:
            indices, _, _, input_ids, attention_mask = batch
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            scores = score_candidate_labels(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                reduction=args.score_reduction,
                labels=active_labels,
            )
            best = scores.argmax(dim=1).tolist()
            labels = [active_labels[index] for index in best]
            for index, label in zip(indices.tolist(), labels):
                writer.write(f"{dataset.data[index]['id']}\t{label}\n")
            if scores_writer is not None:
                for index, label, row_scores in zip(indices.tolist(), labels, scores.tolist()):
                    scores_writer.write(json.dumps({
                        "id": str(dataset.data[index]["id"]),
                        "prediction": label,
                        "scores": dict(zip(active_labels, row_scores)),
                    }, ensure_ascii=False) + "\n")
    if scores_writer is not None:
        scores_writer.close()
    print(f"Wrote {len(dataset)} claim labels to {output}")


if __name__ == "__main__":
    main()
