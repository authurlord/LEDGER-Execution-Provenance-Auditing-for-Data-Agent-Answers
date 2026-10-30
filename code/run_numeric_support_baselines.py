#!/usr/bin/env python3
"""Numeric support baselines under the frozen ConvFinQA operating point."""

from __future__ import annotations

import json
from collections import defaultdict

from common import ARTIFACTS, RECORDS, ROOT, binary_metrics, load_jsonl, write_csv, write_json

from kramabench_integration import provenance as PV


OPS = ("sum", "diff", "ratio", "pctchg", "prod")
TOL = 1e-4


def grouped_pairs(path):
    groups = defaultdict(dict)
    for row in load_jsonl(path):
        taskid = row["taskid"]
        if taskid.endswith("::positive"):
            groups[taskid[: -len("::positive")]]["positive"] = row
        elif taskid.endswith("::negative"):
            groups[taskid[: -len("::negative")]]["negative"] = row
    return [value for value in groups.values() if set(value) == {"positive", "negative"}]


def constrained_accept(row, max_depth):
    prov = (row.get("provenance") or [{}])[0]
    pool = PV.ConstrainedPool(
        prov.get("cited_operands") or [],
        prov.get("cited_consts") or [],
        OPS,
        chain="full",
    )
    return bool(
        pool.check(
            row.get("pred"),
            tol_rel=TOL,
            percent=True,
            percent_ctx=bool(row.get("percent_ctx")),
            max_depth=max_depth,
        )
    )


def evaluate_pairs(pairs, dataset):
    if dataset == "convfinqa":
        # The two canonical no-operand rows are explicitly unknown. Keep the
        # canonical evaluable denominator instead of silently treating abstention
        # as a negative decision.
        pairs = [
            pair
            for pair in pairs
            if pair["positive"].get("constrained_positive_hit") is not None
        ]
    methods = {
        "exact_operand": (lambda row: constrained_accept(row, 0)),
        "nearest_operand_tolerance": (lambda row: constrained_accept(row, 0)),
        "single_operation": (lambda row: constrained_accept(row, 1)),
        "operand_constrained_closure": (lambda row: constrained_accept(row, 2)),
    }
    if dataset == "convfinqa":
        methods["all_numbers_in_cited_file"] = lambda row: bool(row.get("free_positive_hit")) if str(
            row["taskid"]
        ).endswith("::positive") else not bool(row.get("free_negative_flagged"))
        methods["all_context_numbers"] = methods["all_numbers_in_cited_file"]

    rows = []
    y_true = []
    flattened = []
    for pair in pairs:
        flattened.extend((pair["positive"], pair["negative"]))
        y_true.extend((1, 0))

    for name, method in methods.items():
        accepted = [bool(method(row)) for row in flattened]
        metrics = binary_metrics(y_true, accepted)
        rows.append({"dataset": dataset, "method": name, **metrics})

    # Gold-program execution is an oracle upper bound, not a deployable baseline.
    rows.append(
        {
            "dataset": dataset,
            "method": "gold_program_execution_oracle",
            **binary_metrics(y_true, [label == 1 for label in y_true]),
        }
    )
    return rows


def render_tex(rows):
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Dataset & Method & Support recall & Unsupported detection \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {row['method'].replace('_', ' ')} & "
            f"{row['support_recall']:.3f} ({row['support_recall_counts']}) & "
            f"{row['unsupported_detection']:.3f} "
            f"({row['unsupported_detection_counts']}) \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main():
    conv = grouped_pairs(RECORDS / "records_ledger_checker_convfinqa_full.jsonl")
    multi = grouped_pairs(RECORDS / "records_ledger_checker_multihiertt.jsonl")
    rows = evaluate_pairs(conv, "convfinqa") + evaluate_pairs(multi, "multihiertt")
    payload = {
        "schema_version": 1,
        "frozen_point": {
            "ops": list(OPS),
            "max_depth": 2,
            "tol_rel": TOL,
            "percent": True,
        },
        "notes": {
            "all_numbers_in_cited_file": (
                "Available for ConvFinQA because each example has one inline evidence document; "
                "the existing free-closure native tolerance is retained."
            ),
            "multihiertt_omission": (
                "Whole-file/context closure is omitted rather than imputed because the canonical "
                "paired records retain cited operands but not a stable full-context candidate pool."
            ),
            "oracle": "Gold-program execution is shown only as an upper bound.",
        },
        "results": rows,
    }
    write_json(RECORDS / "numeric_support_baselines_convfinqa.json", {
        **payload,
        "results": [row for row in rows if row["dataset"] == "convfinqa"],
    })
    write_json(RECORDS / "numeric_support_baselines_multihiertt.json", {
        **payload,
        "results": [row for row in rows if row["dataset"] == "multihiertt"],
    })
    write_csv(ARTIFACTS / "table_numeric_support_baselines.csv", rows)
    (ARTIFACTS / "table_numeric_support_baselines.tex").write_text(
        render_tex(rows), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
