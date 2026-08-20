import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_output = output_dir / "corpus.jsonl"
    query_output = output_dir / "queries.tsv"

    corpus_count = 0

    with open(corpus_output, "w", encoding="utf-8") as writer:
        for document in read_jsonl(args.corpus):
            abstract = document.get("abstract", [])

            if isinstance(abstract, list):
                abstract_text = " ".join(str(sentence) for sentence in abstract)
            else:
                abstract_text = str(abstract)

            row = {
                "_id": str(document["doc_id"]),
                "title": str(document.get("title", "")),
                "text": abstract_text,
            }

            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            corpus_count += 1

    query_count = 0

    with open(query_output, "w", encoding="utf-8") as writer:
        for claim in read_jsonl(args.claims):
            claim_id = str(claim["id"])
            claim_text = str(claim["claim"]).replace("\t", " ").replace("\n", " ")

            writer.write(f"{claim_id}\t{claim_text}\n")
            query_count += 1

    print(f"Corpus: {corpus_output} ({corpus_count})")
    print(f"Queries: {query_output} ({query_count})")


if __name__ == "__main__":
    main()