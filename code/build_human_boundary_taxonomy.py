#!/usr/bin/env python3
"""Prepare an explicit manual-review sheet for replayable human-unsupported cases."""

from __future__ import annotations

import csv
import json
from collections import Counter

from common import ARTIFACTS, DOCS, RECORDS, write_csv, write_json


CATEGORIES = (
    "wrong aggregation",
    "wrong temporal scope",
    "wrong row subset",
    "wrong column or measure",
    "parse or coercion error",
    "hard-coded or fallback output",
    "other",
)


def suggest(note):
    text = note.lower()
    rules = (
        ("temporal scope", "wrong temporal scope"),
        ("year", "wrong temporal scope"),
        ("average", "wrong aggregation"),
        ("mean", "wrong aggregation"),
        ("sum", "wrong aggregation"),
        ("count", "wrong aggregation"),
        ("subset", "wrong row subset"),
        ("filter", "wrong row subset"),
        ("row", "wrong row subset"),
        ("column", "wrong column or measure"),
        ("measure", "wrong column or measure"),
        ("coerc", "parse or coercion error"),
        ("parse", "parse or coercion error"),
        ("hard-coded", "hard-coded or fallback output"),
        ("fallback", "hard-coded or fallback output"),
    )
    return next((category for token, category in rules if token in text), "other")


def main():
    with (DOCS / "annotation" / "KB_SUPPORT_ANNOTATION.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row["human_support_label"].strip().lower() == "no"
        and row["checker_label"].strip().lower() == "supported"
    ]
    review = []
    for row in selected:
        review.append({
            "taskid": row["taskid"],
            "domain": row["domain"],
            "question": row["question"],
            "stated_answer": row["stated_answer"],
            "recomputed_value": row["recomputed_value"],
            "human_note": row["human_note"],
            "suggested_category": suggest(row["human_note"]),
            "allowed_categories": " | ".join(CATEGORIES),
            "confirmed_category": "",
            "reviewer": "",
            "review_status": "PENDING_MANUAL_CONFIRMATION",
        })
    write_csv(
        ROOT_REVIEW := DOCS.parent / "annotation" / "HUMAN_BOUNDARY_TAXONOMY_REVIEW.csv",
        review,
    )
    payload = {
        "schema_version": 1,
        "n_cases": len(review),
        "status": "draft_requires_manual_confirmation",
        "allowed_categories": list(CATEGORIES),
        "suggested_counts": dict(Counter(row["suggested_category"] for row in review)),
        "cases": review,
        "review_file": str(ROOT_REVIEW.relative_to(DOCS.parent)),
    }
    write_json(RECORDS / "human_boundary_taxonomy.json", payload)
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Draft category & Cases \\",
        r"\midrule",
    ]
    for category, count in sorted(payload["suggested_counts"].items()):
        lines.append(f"{category} & {count} \\\\")
    lines.extend([
        r"\midrule",
        rf"Pending manual confirmation & {len(review)} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (ARTIFACTS / "table_human_boundary_taxonomy.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
