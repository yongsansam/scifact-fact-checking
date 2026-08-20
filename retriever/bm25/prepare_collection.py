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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--filename", default="scifact.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.filename

    count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for doc in read_jsonl(args.corpus):
            title = str(doc.get("title", "")).strip()
            abstract = doc.get("abstract", [])

            if isinstance(abstract, list):
                abstract_text = " ".join(
                    str(sentence).strip()
                    for sentence in abstract
                    if str(sentence).strip()
                )
            else:
                abstract_text = str(abstract).strip()

            contents = f"{title}\n{abstract_text}".strip()

            record = {
                "id": str(doc["doc_id"]),
                "contents": contents,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} documents to {output_path}")


if __name__ == "__main__":
    main()