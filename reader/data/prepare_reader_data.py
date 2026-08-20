#!/usr/bin/env python3
"""Build one-label-per-claim SciFact data for the existing FiD reader."""

import argparse
import json
from collections import Counter
from pathlib import Path


LABEL_MAP = {
    "SUPPORT": "SUPPORTS",
    "SUPPORTS": "SUPPORTS",
    "CONTRADICT": "REFUTES",
    "CONTRADICTS": "REFUTES",
    "CONTRADICTION": "REFUTES",
    "REFUTE": "REFUTES",
    "REFUTES": "REFUTES",
    "NO_EVIDENCE": "NOINFO",
    "NEI": "NOINFO",
    "NOINFO": "NOINFO",
}


def load_rows(path):
    path = Path(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def row_id(row):
    for key in ("claim_id", "id"):
        if key in row:
            return str(row[key])
    raise KeyError(f"Retrieval row has no claim_id/id: {row}")


def claim_label(claim):
    labels = set()
    for evidence_sets in (claim.get("evidence") or {}).values():
        for evidence in evidence_sets or []:
            raw_label = str(evidence.get("label", "")).upper()
            if raw_label not in LABEL_MAP:
                raise ValueError(
                    f"Unknown label {raw_label!r} in claim {claim['id']}"
                )
            labels.add(LABEL_MAP[raw_label])
    labels.discard("NOINFO")
    if len(labels) > 1:
        raise ValueError(
            f"Claim {claim['id']} has conflicting labels: {sorted(labels)}"
        )
    return next(iter(labels), "NOINFO")


def ranked_contexts(row, corpus):
    if "contexts" in row:
        values = sorted(
            row["contexts"],
            key=lambda item: int(item.get("rank", 10**9)),
        )
        output = []
        for item in values:
            doc_id = str(item["doc_id"])
            document = corpus.get(doc_id, item)
            output.append((doc_id, document, item.get("score")))
        return output

    values = None
    for key in ("doc_ids", "retrieved_doc_ids", "docs"):
        if key in row:
            values = row[key]
            break
    if values is None:
        raise KeyError(f"Retrieval row has no ranked documents: {row}")

    output = []
    for item in values:
        if isinstance(item, dict):
            doc_id = str(item.get("doc_id", item.get("id")))
            score = item.get("score")
        else:
            doc_id, score = str(item), None
        output.append((doc_id, corpus.get(doc_id), score))
    return output


def abstract_text(document):
    abstract = document.get("abstract", document.get("text", ""))
    if isinstance(abstract, list):
        return " ".join(str(sentence).strip() for sentence in abstract if str(sentence).strip())
    return str(abstract or "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-context", type=int, default=3)
    parser.add_argument("--evidence-only", action="store_true")
    args = parser.parse_args()
    if args.n_context < 1:
        raise ValueError("--n-context must be at least 1")

    claims = load_rows(args.claims)
    corpus = {str(row["doc_id"]): row for row in load_rows(args.corpus)}
    retrieval = {row_id(row): row for row in load_rows(args.retrieval)}
    examples = []
    counts = Counter()
    missing_retrieval = []

    for claim in claims:
        claim_id = str(claim["id"])
        if claim_id not in retrieval:
            missing_retrieval.append(claim_id)
            continue

        selected = []
        seen = set()
        for doc_id, document, score in ranked_contexts(retrieval[claim_id], corpus):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            if document is None:
                raise KeyError(f"Document {doc_id} for claim {claim_id} is absent from corpus")
            selected.append((doc_id, document, score))
            if len(selected) == args.n_context:
                break
        if len(selected) != args.n_context:
            raise ValueError(
                f"Claim {claim_id} has {len(selected)} usable unique contexts; "
                f"expected {args.n_context}"
            )

        label = claim_label(claim)
        if args.evidence_only and label == "NOINFO":
            continue
        contexts = []
        for rank, (_, document, original_score) in enumerate(selected, 1):
            contexts.append({
                "title": str(document.get("title", "")),
                "text": abstract_text(document),
                "score": float(original_score) if original_score is not None else float(args.n_context - rank + 1),
            })
        examples.append({
            "id": claim_id,
            "question": (
                "Read all retrieved abstracts together and classify the scientific "
                "claim. Output exactly one label: "
                + ("SUPPORTS or REFUTES. " if args.evidence_only else
                   "SUPPORTS, REFUTES, or NOINFO. ")
                + f"Claim: {str(claim['claim']).strip()}"
            ),
            "target": label,
            "answers": [label],
            "ctxs": contexts,
            "claim_id": claim_id,
            "ctx_doc_ids": [doc_id for doc_id, _, _ in selected],
            "claim_label": label,
        })
        counts[label] += 1

    if missing_retrieval:
        raise ValueError(
            f"Missing retrieval rows for {len(missing_retrieval)} claims; "
            f"first IDs: {missing_retrieval[:10]}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "claims": len(claims),
        "examples": len(examples),
        "labels": dict(counts),
        "n_context": args.n_context,
        "evidence_only": args.evidence_only,
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
