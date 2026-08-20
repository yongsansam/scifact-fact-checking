import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    get_linear_schedule_with_warmup,
)
from pyserini.search.lucene import LuceneSearcher


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}의 {line_number}번째 줄이 올바른 JSON이 아닙니다."
                ) from error

    return rows


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def abstract_text(document):
    abstract = document.get("abstract", [])

    if isinstance(abstract, list):
        return " ".join(str(sentence) for sentence in abstract)

    if abstract is None:
        return ""

    return str(abstract)


def build_train_examples(
    claims,
    corpus,
    searcher,
    hard_negative_pool,
    seed,
):
    rng = random.Random(seed)
    all_doc_ids = list(corpus.keys())
    examples = []

    for claim in tqdm(claims, desc="Mining BM25 hard negatives"):
        evidence = claim.get("evidence") or {}
        gold_doc_ids = {
            str(doc_id)
            for doc_id in evidence.keys()
            if str(doc_id) in corpus
        }

        if not gold_doc_ids:
            continue

        hits = searcher.search(
            str(claim["claim"]),
            hard_negative_pool,
        )

        hard_negative_ids = []

        for hit in hits:
            doc_id = str(hit.docid)

            if doc_id not in corpus:
                continue

            if doc_id in gold_doc_ids:
                continue

            if doc_id not in hard_negative_ids:
                hard_negative_ids.append(doc_id)

        if not hard_negative_ids:
            candidates = [
                doc_id
                for doc_id in all_doc_ids
                if doc_id not in gold_doc_ids
            ]

            if not candidates:
                continue

            hard_negative_ids = [
                rng.choice(candidates)
            ]

        examples.append(
            {
                "claim_id": claim["id"],
                "question": str(claim["claim"]),
                "gold_doc_ids": sorted(gold_doc_ids),
                "hard_negative_ids": hard_negative_ids,
            }
        )

    return examples


class SciFactDPRDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class DPRCollator:
    def __init__(
        self,
        corpus,
        question_tokenizer,
        context_tokenizer,
        question_max_length,
        context_max_length,
    ):
        self.corpus = corpus
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer
        self.question_max_length = question_max_length
        self.context_max_length = context_max_length

    def __call__(self, batch):
        questions = []
        positive_ids = []
        negative_ids = []
        gold_sets = []

        positive_titles = []
        positive_texts = []
        negative_titles = []
        negative_texts = []

        for example in batch:
            positive_id = random.choice(
                example["gold_doc_ids"]
            )

            negative_id = random.choice(
                example["hard_negative_ids"]
            )

            positive_document = self.corpus[positive_id]
            negative_document = self.corpus[negative_id]

            questions.append(example["question"])
            positive_ids.append(positive_id)
            negative_ids.append(negative_id)
            gold_sets.append(set(example["gold_doc_ids"]))

            positive_titles.append(
                str(positive_document.get("title", ""))
            )
            positive_texts.append(
                abstract_text(positive_document)
            )

            negative_titles.append(
                str(negative_document.get("title", ""))
            )
            negative_texts.append(
                abstract_text(negative_document)
            )

        question_inputs = self.question_tokenizer.batch_encode_plus(
            questions,
            add_special_tokens=True,
            max_length=self.question_max_length,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="pt",
        )

        context_pairs = list(
            zip(
                positive_titles + negative_titles,
                positive_texts + negative_texts,
            )
        )

        context_inputs = self.context_tokenizer.batch_encode_plus(
            context_pairs,
            add_special_tokens=True,
            max_length=self.context_max_length,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "question_inputs": question_inputs,
            "context_inputs": context_inputs,
            "positive_ids": positive_ids,
            "negative_ids": negative_ids,
            "gold_sets": gold_sets,
        }


def move_to_device(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def get_vector(model, inputs):
    outputs = model(**inputs)
    return outputs[0]


def mask_false_negatives(
    logits,
    gold_sets,
    candidate_ids,
):
    masked_logits = logits.float().clone()
    mask_value = torch.finfo(masked_logits.dtype).min

    for query_index, gold_doc_ids in enumerate(gold_sets):
        for candidate_index, candidate_id in enumerate(candidate_ids):
            if candidate_index == query_index:
                continue

            if candidate_id in gold_doc_ids:
                masked_logits[query_index, candidate_index] = mask_value

    return masked_logits


def encode_corpus(
    model,
    tokenizer,
    corpus_rows,
    batch_size,
    max_length,
    device,
):
    model.eval()
    vectors = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(corpus_rows), batch_size),
            desc="Encoding dev corpus",
            leave=False,
        ):
            batch_rows = corpus_rows[
                start:start + batch_size
            ]

            pairs = [
                (
                    str(document.get("title", "")),
                    abstract_text(document),
                )
                for document in batch_rows
            ]

            inputs = tokenizer.batch_encode_plus(
                pairs,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=True,
                truncation=True,
                return_tensors="pt",
            )

            inputs = move_to_device(inputs, device)
            batch_vectors = get_vector(model, inputs)

            vectors.append(
                batch_vectors.detach().cpu()
            )

    return torch.cat(vectors, dim=0)


def encode_queries(
    model,
    tokenizer,
    claims,
    batch_size,
    max_length,
    device,
):
    model.eval()
    vectors = []

    with torch.no_grad():
        for start in range(0, len(claims), batch_size):
            batch_claims = claims[
                start:start + batch_size
            ]

            questions = [
                str(claim["claim"])
                for claim in batch_claims
            ]

            inputs = tokenizer.batch_encode_plus(
                questions,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=True,
                truncation=True,
                return_tensors="pt",
            )

            inputs = move_to_device(inputs, device)
            batch_vectors = get_vector(model, inputs)

            vectors.append(
                batch_vectors.detach().cpu()
            )

    return torch.cat(vectors, dim=0)


def evaluate_retrieval(
    question_encoder,
    context_encoder,
    question_tokenizer,
    context_tokenizer,
    corpus_rows,
    dev_claims,
    eval_ks,
    encode_batch_size,
    question_max_length,
    context_max_length,
    device,
):
    context_vectors = encode_corpus(
        model=context_encoder,
        tokenizer=context_tokenizer,
        corpus_rows=corpus_rows,
        batch_size=encode_batch_size,
        max_length=context_max_length,
        device=device,
    )

    query_vectors = encode_queries(
        model=question_encoder,
        tokenizer=question_tokenizer,
        claims=dev_claims,
        batch_size=encode_batch_size,
        max_length=question_max_length,
        device=device,
    )

    scores = torch.matmul(
        query_vectors,
        context_vectors.transpose(0, 1),
    )

    max_k = min(
        max(eval_ks),
        scores.size(1),
    )

    ranked_indices = torch.topk(
        scores,
        k=max_k,
        dim=1,
    ).indices

    corpus_doc_ids = [
        str(document["doc_id"])
        for document in corpus_rows
    ]

    hit_counts = {
        k: 0
        for k in eval_ks
    }

    evidence_claim_count = 0

    for claim_index, claim in enumerate(dev_claims):
        gold_doc_ids = {
            str(doc_id)
            for doc_id in (claim.get("evidence") or {}).keys()
        }

        if not gold_doc_ids:
            continue

        evidence_claim_count += 1

        ranked_doc_ids = [
            corpus_doc_ids[index]
            for index in ranked_indices[claim_index].tolist()
        ]

        for k in eval_ks:
            retrieved_doc_ids = set(
                ranked_doc_ids[:k]
            )

            if retrieved_doc_ids.intersection(gold_doc_ids):
                hit_counts[k] += 1

    metrics = {
        f"EvidenceHit@{k}": (
            hit_counts[k] / evidence_claim_count
            if evidence_claim_count > 0
            else 0.0
        )
        for k in eval_ks
    }

    metrics["EvidenceClaimCount"] = evidence_claim_count

    return metrics


def save_checkpoint(
    output_dir,
    question_encoder,
    context_encoder,
    question_tokenizer,
    context_tokenizer,
    metadata,
):
    output_dir = Path(output_dir)
    question_dir = output_dir / "question_encoder"
    context_dir = output_dir / "ctx_encoder"

    question_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    question_encoder.save_pretrained(
        str(question_dir)
    )

    context_encoder.save_pretrained(
        str(context_dir)
    )

    question_tokenizer.save_pretrained(
        str(question_dir)
    )

    context_tokenizer.save_pretrained(
        str(context_dir)
    )

    with open(
        output_dir / "training_state.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus", required=True)
    parser.add_argument("--train-claims", required=True)
    parser.add_argument("--dev-claims", required=True)
    parser.add_argument("--bm25-index", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--question-model",
        default="facebook/dpr-question_encoder-single-nq-base",
    )

    parser.add_argument(
        "--context-model",
        default="facebook/dpr-ctx_encoder-single-nq-base",
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--question-max-length", type=int, default=64)
    parser.add_argument("--context-max-length", type=int, default=256)
    parser.add_argument("--hard-negative-pool", type=int, default=20)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    corpus_rows = read_jsonl(args.corpus)
    train_claims = read_jsonl(args.train_claims)
    dev_claims = read_jsonl(args.dev_claims)

    corpus = {
        str(document["doc_id"]): document
        for document in corpus_rows
    }

    bm25_searcher = LuceneSearcher(
        args.bm25_index
    )

    bm25_searcher.set_bm25(
        args.k1,
        args.b,
    )

    train_examples = build_train_examples(
        claims=train_claims,
        corpus=corpus,
        searcher=bm25_searcher,
        hard_negative_pool=args.hard_negative_pool,
        seed=args.seed,
    )

    if not train_examples:
        raise ValueError(
            "학습 가능한 evidence claim이 없습니다."
        )

    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        args.question_model
    )

    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        args.context_model
    )

    question_encoder = DPRQuestionEncoder.from_pretrained(
        args.question_model
    ).to(device)

    context_encoder = DPRContextEncoder.from_pretrained(
        args.context_model
    ).to(device)

    dataset = SciFactDPRDataset(
        train_examples
    )

    collator = DPRCollator(
        corpus=corpus,
        question_tokenizer=question_tokenizer,
        context_tokenizer=context_tokenizer,
        question_max_length=args.question_max_length,
        context_max_length=args.context_max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        drop_last=False,
    )

    parameters = list(
        question_encoder.parameters()
    ) + list(
        context_encoder.parameters()
    )

    optimizer = AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(
        len(loader) / args.gradient_accumulation
    )

    total_updates = (
        updates_per_epoch * args.epochs
    )

    warmup_steps = int(
        total_updates * args.warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=args.fp16
    )

    eval_ks = [1, 3, 5, 10, 20]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train examples: {len(train_examples)}")
    print(f"Corpus documents: {len(corpus_rows)}")
    print(f"Dev claims: {len(dev_claims)}")
    print(f"Device: {device}")

    initial_metrics = evaluate_retrieval(
        question_encoder=question_encoder,
        context_encoder=context_encoder,
        question_tokenizer=question_tokenizer,
        context_tokenizer=context_tokenizer,
        corpus_rows=corpus_rows,
        dev_claims=dev_claims,
        eval_ks=eval_ks,
        encode_batch_size=args.encode_batch_size,
        question_max_length=args.question_max_length,
        context_max_length=args.context_max_length,
        device=device,
    )

    print("Zero-shot reference")
    print(
        json.dumps(
            initial_metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    best_hit3 = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        question_encoder.train()
        context_encoder.train()

        optimizer.zero_grad()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch}/{args.epochs}",
        )

        for batch_index, batch in enumerate(progress, start=1):
            question_inputs = move_to_device(
                batch["question_inputs"],
                device,
            )

            context_inputs = move_to_device(
                batch["context_inputs"],
                device,
            )

            batch_size = len(
                batch["positive_ids"]
            )

            with torch.cuda.amp.autocast(
                enabled=args.fp16
            ):
                question_vectors = get_vector(
                    question_encoder,
                    question_inputs,
                )

                context_vectors = get_vector(
                    context_encoder,
                    context_inputs,
                )

                candidate_ids = (
                    batch["positive_ids"]
                    + batch["negative_ids"]
                )

                logits = torch.matmul(
                    question_vectors,
                    context_vectors.transpose(0, 1),
                )

                logits = mask_false_negatives(
                    logits=logits,
                    gold_sets=batch["gold_sets"],
                    candidate_ids=candidate_ids,
                )

                targets = torch.arange(
                    batch_size,
                    device=device,
                )

                loss = F.cross_entropy(
                    logits,
                    targets,
                )

                scaled_loss = (
                    loss / args.gradient_accumulation
                )

            scaler.scale(scaled_loss).backward()

            should_update = (
                batch_index % args.gradient_accumulation == 0
                or batch_index == len(loader)
            )

            if should_update:
                scaler.unscale_(optimizer)

                clip_grad_norm_(
                    parameters,
                    args.max_grad_norm,
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == targets
            ).sum().item()

            total_examples += batch_size
            total_loss += loss.item() * batch_size

            progress.set_postfix(
                loss=f"{total_loss / total_examples:.4f}",
                acc=f"{total_correct / total_examples:.4f}",
            )

        dev_metrics = evaluate_retrieval(
            question_encoder=question_encoder,
            context_encoder=context_encoder,
            question_tokenizer=question_tokenizer,
            context_tokenizer=context_tokenizer,
            corpus_rows=corpus_rows,
            dev_claims=dev_claims,
            eval_ks=eval_ks,
            encode_batch_size=args.encode_batch_size,
            question_max_length=args.question_max_length,
            context_max_length=args.context_max_length,
            device=device,
        )

        epoch_metadata = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": total_loss / total_examples,
            "train_accuracy": total_correct / total_examples,
            "dev_metrics": dev_metrics,
            "question_model": args.question_model,
            "context_model": args.context_model,
            "hard_negative_retriever": "BM25",
            "bm25_k1": args.k1,
            "bm25_b": args.b,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        }

        print(
            json.dumps(
                epoch_metadata,
                ensure_ascii=False,
                indent=2,
            )
        )

        save_checkpoint(
            output_dir=output_dir / "last",
            question_encoder=question_encoder,
            context_encoder=context_encoder,
            question_tokenizer=question_tokenizer,
            context_tokenizer=context_tokenizer,
            metadata=epoch_metadata,
        )

        hit3 = dev_metrics["EvidenceHit@3"]

        if hit3 > best_hit3:
            best_hit3 = hit3

            save_checkpoint(
                output_dir=output_dir / "best",
                question_encoder=question_encoder,
                context_encoder=context_encoder,
                question_tokenizer=question_tokenizer,
                context_tokenizer=context_tokenizer,
                metadata=epoch_metadata,
            )

            print(
                f"Best checkpoint updated: "
                f"EvidenceHit@3={best_hit3:.6f}"
            )

    print(f"Best EvidenceHit@3: {best_hit3:.6f}")
    print(f"Best checkpoint: {output_dir / 'best'}")


if __name__ == "__main__":
    main()