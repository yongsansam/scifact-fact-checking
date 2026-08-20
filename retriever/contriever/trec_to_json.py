import argparse
import json
from collections import defaultdict
from pathlib import Path


def convert_id(value):
    value = str(value)

    if value.isdigit():
        return int(value)

    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, required=True)
    args = parser.parse_args()

    if args.topk <= 0:
        raise ValueError("--topk는 1 이상이어야 합니다.")

    results = defaultdict(list)
    query_order = []

    with open(args.run, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            columns = line.strip().split()

            if not columns:
                continue

            if len(columns) < 6:
                raise ValueError(
                    f"{line_number}번째 줄의 TREC 형식이 올바르지 않습니다."
                )

            claim_id = columns[0]
            doc_id = columns[2]
            rank = int(columns[3])
            score = float(columns[4])

            if claim_id not in results:
                query_order.append(claim_id)

            results[claim_id].append(
                {
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": score,
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as writer:
        for claim_id in query_order:
            ranked = sorted(
                results[claim_id],
                key=lambda item: item["rank"],
            )[:args.topk]

            row = {
                "claim_id": convert_id(claim_id),
                "doc_ids": [
                    convert_id(item["doc_id"])
                    for item in ranked
                ],
            }

            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Output: {output_path}")
    print(f"Claims: {len(query_order)}")
    print(f"Top-k: {args.topk}")


if __name__ == "__main__":
    main()