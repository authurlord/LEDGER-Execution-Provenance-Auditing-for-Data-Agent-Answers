#!/usr/bin/env python3
"""Fail closed if any canonical LEDGER submission value drifts."""

from __future__ import annotations

import csv
from collections import Counter

from common import DOCS, RECORDS, load_json, load_jsonl, write_json


EXPECTED = {
    "convfinqa": {
        "total_remainder": 388,
        "evaluable": 386,
        "support_recall_counts": "360/386",
        "unsupported_detection_counts": "371/386",
        "free_support_recall_counts": "334/388",
        "free_unsupported_detection_counts": "242/388",
    },
    "multihiertt": {
        "total": 7830,
        "evaluable": 6301,
        "support_recall_counts": "5648/6301",
        "unsupported_detection_counts": "6125/6301",
        "recall_misses": 653,
        "misses_depth_gt_2": 472,
    },
    "wrong_operation": {
        "constructible": 382,
        "detected_counts": "9/382",
        "within_depth_detected_counts": "0/347",
    },
    "kramabench": {
        "total": 166,
        "supported": 107,
        "unsupported": 3,
        "unknown": 56,
    },
    "dabstep": {
        "total": 60,
        "supported": 43,
        "unsupported": 0,
        "unknown": 17,
    },
    "human": {
        "rows": 40,
        "clear": 39,
        "yes": 11,
        "no": 28,
        "unclear": 1,
        "replay_and_human_no": 10,
    },
}


def observed_values() -> dict:
    conv = load_json(RECORDS / "convfinqa_checker_fulln_summary.json")
    multi = load_json(RECORDS / "multihiertt_checker_summary.json")
    wrong = load_json(RECORDS / "convfinqa_wrong_operation_negatives_summary.json")

    kb_rows = []
    for path in sorted(RECORDS.glob("records_ledger_n2_kb_*.jsonl")):
        kb_rows.extend(load_jsonl(path))
    kb_counts = Counter(row["grounding_label"] for row in kb_rows)

    dab_rows = load_jsonl(RECORDS / "records_ledger_dabstep_support.jsonl")
    dab_counts = Counter(row["grounding_label"] for row in dab_rows)

    with (DOCS / "annotation" / "KB_SUPPORT_ANNOTATION.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        human_rows = list(csv.DictReader(handle))
    human_counts = Counter(row["human_support_label"].strip().lower() for row in human_rows)
    replay_human_no = sum(
        row["checker_label"].strip().lower() == "supported"
        and row["human_support_label"].strip().lower() == "no"
        for row in human_rows
    )

    return {
        "convfinqa": {
            "total_remainder": conv["full_n_headline"]["n"],
            "evaluable": conv["full_n_headline"]["n_evaluable"],
            "support_recall_counts": conv["full_n_headline"]["constrained"][
                "support_recall_counts"
            ],
            "unsupported_detection_counts": conv["full_n_headline"]["constrained"][
                "unsupported_detection_counts"
            ],
            "free_support_recall_counts": conv["full_n_headline"]["free_closure_baseline"][
                "support_recall_counts"
            ],
            "free_unsupported_detection_counts": conv["full_n_headline"][
                "free_closure_baseline"
            ]["unsupported_detection_counts"],
        },
        "multihiertt": {
            "total": multi["load_stats"]["total"],
            "evaluable": multi["load_stats"]["evaluable"],
            "support_recall_counts": multi["headline"]["support_recall_counts"],
            "unsupported_detection_counts": multi["headline"][
                "unsupported_detection_counts"
            ],
            "recall_misses": multi["prop1_miss_attribution"]["recall_misses"],
            "misses_depth_gt_2": multi["prop1_miss_attribution"]["misses_depth_gt_2"],
        },
        "wrong_operation": {
            "constructible": wrong["wrong_operation_constructible"],
            "detected_counts": wrong["comparison"]["same_operand_wrong_operation"][
                "unsupported_detection_counts"
            ],
            "within_depth_detected_counts": wrong["wrong_operation_depth_strata"][0][
                "unsupported_detection_counts"
            ],
        },
        "kramabench": {
            "total": len(kb_rows),
            "supported": kb_counts["supported"],
            "unsupported": kb_counts["unsupported"],
            "unknown": kb_counts["unknown"],
        },
        "dabstep": {
            "total": len(dab_rows),
            "supported": dab_counts["supported"],
            "unsupported": dab_counts["unsupported"],
            "unknown": dab_counts["unknown"],
        },
        "human": {
            "rows": len(human_rows),
            "clear": len(human_rows) - human_counts["unclear"],
            "yes": human_counts["yes"],
            "no": human_counts["no"],
            "unclear": human_counts["unclear"],
            "replay_and_human_no": replay_human_no,
        },
    }


def main() -> None:
    observed = observed_values()
    mismatches = []
    for group, values in EXPECTED.items():
        for key, expected in values.items():
            actual = observed[group].get(key)
            if actual != expected:
                mismatches.append(
                    {"group": group, "field": key, "expected": expected, "actual": actual}
                )
    result = {
        "schema_version": 1,
        "expected": EXPECTED,
        "observed": observed,
        "mismatches": mismatches,
        "pass": not mismatches,
    }
    write_json(RECORDS / "submission_assertions_rev2.json", result)

    lines = [
        "# LEDGER submission assertions rev2",
        "",
        f"Status: **{'PASS' if not mismatches else 'FAIL'}**.",
        "",
        "This check freezes the July 30 canonical counts before new experiments run.",
        "",
        "| group | field | expected | observed |",
        "|---|---|---:|---:|",
    ]
    for group, values in EXPECTED.items():
        for key, expected in values.items():
            lines.append(f"| {group} | {key} | {expected} | {observed[group][key]} |")
    if mismatches:
        lines.extend(["", "## Mismatches", "", f"`{mismatches}`"])
    (DOCS / "SUBMISSION_ASSERTIONS_REV2.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("PASS" if not mismatches else f"FAIL: {mismatches}")
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
