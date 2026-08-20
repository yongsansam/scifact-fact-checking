import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                yield json.loads(line)


def clean_text(value):
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_output = output_dir / "corpus.jsonl"
    queries_output = output_dir / "queries.tsv"

    corpus_count = 0

    with open(corpus_output, "w", encoding="utf-8") as writer:
        for document in read_jsonl(args.corpus):
            title = clean_text(document.get("title", ""))
            abstract = document.get("abstract", [])

            if isinstance(abstract, list):
                abstract_text = " ".join(
                    clean_text(sentence)
                    for sentence in abstract
                )
            else:
                abstract_text = clean_text(abstract)

            row = {
                "id": str(document["doc_id"]),
                "contents": f"{title}\n{abstract_text}",
            }

            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            corpus_count += 1

    query_count = 0

    with open(queries_output, "w", encoding="utf-8") as writer:
        for claim in read_jsonl(args.claims):
            claim_id = str(claim["id"])
            claim_text = clean_text(claim["claim"])

            writer.write(f"{claim_id}\t{claim_text}\n")
            query_count += 1

    print(f"Corpus documents: {corpus_count}")
    print(f"Queries: {query_count}")
    print(f"Corpus output: {corpus_output}")
    print(f"Queries output: {queries_output}")


if __name__ == "__main__":
    main()