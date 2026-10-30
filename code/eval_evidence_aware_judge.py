#!/usr/bin/env python3
"""Summarize repeat-level and majority-vote evidence-aware judge results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import DOCS, RECORDS, load_jsonl, write_json


ORDER = {"unsupported": 0, "unknown": 1, "supported": 2}


def majority(rows):
    counts = Counter(row["judge_label"] for row in rows)
    return max(counts, key=lambda label: (counts[label], -ORDER[label]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=RECORDS / "evidence_aware_judge_records.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=RECORDS / "evidence_aware_judge_summary.json",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=DOCS / "EVIDENCE_AWARE_JUDGE.md",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rows = load_jsonl(args.records)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["item_id"])].append(row)
    items = []
    for (dataset, item_id), repeats in sorted(grouped.items()):
        labels = [row["judge_label"] for row in repeats]
        items.append({
            "dataset": dataset,
            "item_id": item_id,
            "reference_label": repeats[0]["reference_label"],
            "majority_label": majority(repeats),
            "repeat_flip": len(set(labels)) > 1,
            "parse_failures": sum(not row["parse_ok"] for row in repeats),
        })
    summaries = {}
    for dataset in sorted({row["dataset"] for row in items}):
        subset = [row for row in items if row["dataset"] == dataset]
        clear = [
            row for row in subset
            if not (dataset == "kramabench_human" and row["reference_label"] == "unknown")
        ]
        summaries[dataset] = {
            "n_items": len(subset),
            "n_clear": len(clear),
            "majority_accuracy": (
                sum(row["majority_label"] == row["reference_label"] for row in clear)
                / len(clear)
                if clear else None
            ),
            "majority_correct": sum(
                row["majority_label"] == row["reference_label"] for row in clear
            ),
            "repeat_flip_items": sum(row["repeat_flip"] for row in subset),
            "repeat_flip_rate": sum(row["repeat_flip"] for row in subset) / len(subset),
            "parse_failures": sum(row["parse_failures"] for row in subset),
            "prediction_counts": dict(Counter(row["majority_label"] for row in subset)),
        }
    payload = {
        "schema_version": 1,
        "model": rows[0]["model"] if rows else None,
        "temperature": 0,
        "repeats": args.repeats,
        "calls": len(rows),
        "tokens_in": sum(row["tokens_in"] for row in rows),
        "tokens_out": sum(row["tokens_out"] for row in rows),
        "latency_s_sum": sum(row["latency_s"] for row in rows),
        "cost_usd": 0.0,
        "parse_failures": sum(not row["parse_ok"] for row in rows),
        "datasets": summaries,
        "ledger_comparator": {
            "model_calls": 0,
            "monetary_cost_usd": 0.0,
            "property": "deterministic audit-layer recomputability, not an LLM semantic verdict",
        },
        "human_evidence_limitation": (
            "The transferred archive lacks raw cited rows/program bodies for most human cases; "
            "the judge received declared columns, reads, execution status, and recomputed output. "
            "Human accuracy is therefore reported as limited-evidence, not a definitive fair ranking."
        ),
    }
    write_json(args.summary, payload)
    lines = [
        "# Evidence-aware local LLM judge",
        "",
        f"- Model: `{payload['model']}`; temperature 0; {args.repeats} repeats.",
        f"- Calls: {payload['calls']}; parse failures: {payload['parse_failures']}.",
        f"- Tokens: {payload['tokens_in']} input / {payload['tokens_out']} output; local cost $0.",
        "",
        "## Results",
        "",
    ]
    for dataset, summary in summaries.items():
        lines.append(
            f"- {dataset}: {summary['majority_correct']}/{summary['n_clear']} "
            f"majority-vote agreement; flip rate {summary['repeat_flip_rate']:.3f}."
        )
    lines.extend([
        "",
        "## Limitation",
        "",
        payload["human_evidence_limitation"],
        "No benchmark gold answer or LEDGER verdict was provided to the judge.",
        "",
    ])
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
