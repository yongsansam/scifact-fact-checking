#!/usr/bin/env python3
"""Brute-force dense retrieval for the small SciFact corpus."""

import argparse
import json
from pathlib import Path

import torch
from transformers import (AutoModel, AutoTokenizer, DPRContextEncoder,
                          DPRContextEncoderTokenizer, DPRQuestionEncoder,
                          DPRQuestionEncoderTokenizer)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def text(document):
    abstract = document.get("abstract", [])
    return f"{document.get('title', '')} {' '.join(abstract) if isinstance(abstract, list) else abstract}"


@torch.no_grad()
def encode(model, tokenizer, texts, batch_size, max_length, device, mean_pool):
    vectors = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(texts[start:start + batch_size], padding=True,
                           truncation=True, max_length=max_length,
                           return_tensors="pt").to(device)
        output = model(**inputs)
        if mean_pool:
            hidden = output.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            value = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            value = output.pooler_output
        vectors.append(torch.nn.functional.normalize(value.float(), dim=1).cpu())
    return torch.cat(vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=("contriever", "dpr"), required=True)
    parser.add_argument("--corpus", required=True); parser.add_argument("--claims", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--model"); parser.add_argument("--question-model"); parser.add_argument("--context-model")
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); device = torch.device(args.device)
    corpus, claims = read_jsonl(args.corpus), read_jsonl(args.claims)
    if args.type == "contriever":
        if not args.model: parser.error("contriever requires --model")
        query_tokenizer = context_tokenizer = AutoTokenizer.from_pretrained(args.model)
        query_model = context_model = AutoModel.from_pretrained(args.model).to(device)
        mean_pool = True
    else:
        if not args.question_model or not args.context_model:
            parser.error("dpr requires --question-model and --context-model")
        query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.question_model)
        context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.context_model)
        query_model = DPRQuestionEncoder.from_pretrained(args.question_model).to(device)
        context_model = DPRContextEncoder.from_pretrained(args.context_model).to(device)
        mean_pool = False
    contexts = encode(context_model, context_tokenizer, [text(x) for x in corpus],
                      args.batch_size, 256, device, mean_pool)
    queries = encode(query_model, query_tokenizer, [x["claim"] for x in claims],
                     args.batch_size, 64, device, mean_pool)
    indices = torch.topk(queries @ contexts.T, k=min(args.topk, len(corpus)), dim=1).indices
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for claim, ranking in zip(claims, indices):
            row = {"claim_id": claim["id"],
                   "doc_ids": [corpus[i]["doc_id"] for i in ranking.tolist()]}
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

