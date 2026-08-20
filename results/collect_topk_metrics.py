#!/usr/bin/env python3
"""Write compact Accuracy/Macro-F1 summaries from Top-K metrics files."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric", action="append", nargs=2, metavar=("K", "PATH"), required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    results = []
    for raw_k, raw_path in args.metric:
        metrics = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        results.append({
            "k": int(raw_k),
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        })
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "metrics.json"
    if args.merge_existing and output_json.is_file():
        existing = json.loads(output_json.read_text(encoding="utf-8"))
        merged = {int(item["k"]): item for item in existing.get("results", [])}
        merged.update({int(item["k"]): item for item in results})
        results = list(merged.values())
    results.sort(key=lambda row: row["k"])

    payload = {
        "retriever": args.retriever,
        "reader": args.reader,
        "results": results,
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("k", "accuracy", "macro_f1"))
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
