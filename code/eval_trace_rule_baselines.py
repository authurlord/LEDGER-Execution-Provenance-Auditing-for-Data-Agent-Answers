#!/usr/bin/env python3
"""Evaluate each trace rule independently; no undocumented combined heuristic."""

from __future__ import annotations

import json
from collections import defaultdict

from baselines_trace_rules import METHODS
from common import ARTIFACTS, RECORDS, binary_metrics, load_jsonl, write_csv


def summarize(rows, dataset, method):
    applicable = [row for row in rows if row["decisions"].get(method) is not None]
    labeled = [row for row in applicable if row.get("reference_label") in {0, 1}]
    result = {
        "dataset": dataset,
        "method": method,
        "n_total": len(rows),
        "n_applicable": len(applicable),
        "not_applicable": len(rows) - len(applicable),
        "accepted": sum(bool(row["decisions"][method]) for row in applicable),
        "delivered_coverage": (
            sum(bool(row["decisions"][method]) for row in applicable) / len(rows)
            if rows
            else None
        ),
    }
    if labeled:
        result.update(
            binary_metrics(
                [int(row["reference_label"]) for row in labeled],
                [bool(row["decisions"][method]) for row in labeled],
            )
        )
    else:
        result.update(
            {
                "n": 0,
                "support_n": 0,
                "unsupported_n": 0,
                "support_recall": None,
                "support_recall_counts": "0/0",
                "unsupported_detection": None,
                "unsupported_detection_counts": "0/0",
            }
        )
    scores = [
        float(row["official_score"])
        for row in applicable
        if row["decisions"][method] and row.get("official_score") is not None
        and float(row["official_score"]) >= 0
    ]
    result["official_score_on_delivered"] = sum(scores) / len(scores) if scores else None
    result["n_official_scored_delivered"] = len(scores)
    return result


def render_tex(rows):
    selected = [
        row
        for row in rows
        if row["dataset"] in {"kramabench_human", "kramabench_controlled"}
    ]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Rule & Coverage & Support recall & Unsupported detection & N/A \\",
        r"\midrule",
    ]
    for row in selected:
        recall = "--" if row["support_recall"] is None else f"{row['support_recall']:.3f}"
        detection = (
            "--"
            if row["unsupported_detection"] is None
            else f"{row['unsupported_detection']:.3f}"
        )
        lines.append(
            f"{row['dataset']} & {row['method'].replace('_', ' ')} & "
            f"{row['delivered_coverage']:.3f} & {recall} & {detection} & "
            f"{row['not_applicable']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main():
    all_rows = load_jsonl(RECORDS / "trace_rule_baselines.jsonl")
    by_dataset = defaultdict(list)
    for row in all_rows:
        by_dataset[row["dataset"]].append(row)
    summaries = [
        summarize(rows, dataset, method)
        for dataset, rows in sorted(by_dataset.items())
        for method in METHODS
    ]
    write_csv(ARTIFACTS / "table_trace_rule_baselines.csv", summaries)
    (ARTIFACTS / "table_trace_rule_baselines.tex").write_text(
        render_tex(summaries), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
