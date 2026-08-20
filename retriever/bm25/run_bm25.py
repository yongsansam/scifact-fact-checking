import argparse
import json
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_eval_ks(value):
    ks = sorted(
        {
            int(k.strip())
            for k in value.split(",")
            if k.strip()
        }
    )

    if not ks:
        raise ValueError("--eval-ks 값이 비어 있습니다.")

    if any(k <= 0 for k in ks):
        raise ValueError("--eval-ks에는 양의 정수만 사용할 수 있습니다.")

    return ks


def convert_doc_id(doc_id):
    doc_id = str(doc_id)

    if doc_id.isdigit():
        return int(doc_id)

    return doc_id


def get_abstract_sentences(document):
    abstract = document.get("abstract", [])

    if isinstance(abstract, list):
        return [str(sentence) for sentence in abstract]

    if abstract is None:
        return []

    return [str(abstract)]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--index", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--reader-topk", type=int, default=3)
    parser.add_argument("--eval-ks", default="1,3,5,10,20")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)

    args = parser.parse_args()

    if args.reader_topk <= 0:
        raise ValueError("--reader-topk는 1 이상이어야 합니다.")

    eval_ks = parse_eval_ks(args.eval_ks)
    search_depth = max(max(eval_ks), args.reader_topk)

    corpus_rows = read_jsonl(args.corpus)
    claims = read_jsonl(args.claims)

    corpus = {
        str(document["doc_id"]): document
        for document in corpus_rows
    }

    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(args.k1, args.b)

    retrieval_outputs = []
    reader_outputs = []

    evidence_claim_count = 0
    hit_counts = {
        k: 0
        for k in eval_ks
    }

    for claim in claims:
        claim_id = claim["id"]
        claim_text = str(claim["claim"])

        hits = searcher.search(
            claim_text,
            search_depth,
        )

        ranked_doc_ids = [
            str(hit.docid)
            for hit in hits
        ]

        reader_hits = hits[:args.reader_topk]
        reader_doc_ids = ranked_doc_ids[:args.reader_topk]

        evidence = claim.get("evidence") or {}

        gold_doc_ids = {
            str(doc_id)
            for doc_id in evidence.keys()
        }

        retrieval_outputs.append(
            {
                "claim_id": claim_id,
                "doc_ids": [
                    convert_doc_id(doc_id)
                    for doc_id in reader_doc_ids
                ],
            }
        )

        contexts = []

        for rank, hit in enumerate(reader_hits, start=1):
            doc_id = str(hit.docid)

            if doc_id not in corpus:
                raise KeyError(
                    f"검색된 doc_id={doc_id}가 corpus.jsonl에 없습니다."
                )

            document = corpus[doc_id]
            abstract_sentences = get_abstract_sentences(document)

            contexts.append(
                {
                    "rank": rank,
                    "doc_id": document["doc_id"],
                    "title": str(document.get("title", "")),
                    "abstract": abstract_sentences,
                    "text": " ".join(abstract_sentences),
                    "score": float(hit.score),
                    "is_gold": doc_id in gold_doc_ids,
                }
            )

        reader_outputs.append(
            {
                "id": claim_id,
                "claim": claim_text,
                "evidence": evidence,
                "cited_doc_ids": claim.get("cited_doc_ids", []),
                "contexts": contexts,
            }
        )

        if not gold_doc_ids:
            continue

        evidence_claim_count += 1

        for k in eval_ks:
            predicted_doc_ids = set(ranked_doc_ids[:k])

            if predicted_doc_ids.intersection(gold_doc_ids):
                hit_counts[k] += 1

    metrics = {
        "corpus_size": len(corpus_rows),
        "claim_count": len(claims),
        "evidence_claim_count": evidence_claim_count,
        "reader_topk": args.reader_topk,
        "evaluation_depth": search_depth,
        "bm25": {
            "k1": args.k1,
            "b": args.b,
        },
    }

    for k in eval_ks:
        score = (
            hit_counts[k] / evidence_claim_count
            if evidence_claim_count > 0
            else 0.0
        )

        metrics[f"EvidenceHit@{k}"] = score
        metrics[f"EvidenceHitCount@{k}"] = hit_counts[k]

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    retrieval_path = Path(
        f"{output_prefix}.retrieval.jsonl"
    )

    reader_path = Path(
        f"{output_prefix}.reader.jsonl"
    )

    metrics_path = Path(
        f"{output_prefix}.metrics.json"
    )

    write_jsonl(
        retrieval_path,
        retrieval_outputs,
    )

    write_jsonl(
        reader_path,
        reader_outputs,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Retrieval output: {retrieval_path}")
    print(f"Reader output: {reader_path}")
    print(f"Metrics output: {metrics_path}")


if __name__ == "__main__":
    main()