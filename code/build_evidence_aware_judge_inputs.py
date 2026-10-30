#!/usr/bin/env python3
"""Build comparable, verdict-blind evidence bundles for the local LLM judge."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

from common import DOCS, RECORDS, ROOT, load_jsonl, write_jsonl


def _convfinqa_index():
    raw = json.load(
        (ROOT / "datasets" / "convfinqa_original" / "data" / "dev.json").open(
            encoding="utf-8"
        )
    )
    return {item["id"]: item for item in raw}


def build_convfinqa():
    index = _convfinqa_index()
    groups = defaultdict(dict)
    for row in load_jsonl(RECORDS / "records_ledger_checker_convfinqa_full.jsonl"):
        taskid = row["taskid"]
        if taskid.endswith("::positive"):
            groups[taskid.removesuffix("::positive")]["positive"] = row
        elif taskid.endswith("::negative"):
            groups[taskid.removesuffix("::negative")]["negative"] = row
    ordered = sorted(
        (rows for rows in groups.values() if set(rows) == {"positive", "negative"}),
        key=lambda rows: int(rows["positive"].get("full_remainder_index", 10**9)),
    )[:100]
    out = []
    for rows in ordered:
        positive = rows["positive"]
        source_id = positive["taskid"].removesuffix("::positive")
        raw_id = source_id.split("::qa", 1)[0]
        item = index[raw_id]
        qa_key = source_id.rsplit("::", 1)[-1]
        qa = item[qa_key]
        evidence = list((qa.get("gold_inds") or {}).values())
        if not evidence:
            evidence = [" | ".join(map(str, row)) for row in item.get("table", [])[:8]]
        program = qa.get("program") or qa.get("program_re") or ""
        recomputed = positive.get("pred")
        operands = (positive.get("provenance") or [{}])[0]
        for label_name, reference in (("positive", "supported"), ("negative", "unsupported")):
            row = rows[label_name]
            out.append({
                "dataset": "convfinqa_constructed",
                "item_id": row["taskid"],
                "question": qa["question"],
                "stated_answer": row.get("pred"),
                "declared_files": ["inline-table"],
                "evidence_snippets": evidence[:5],
                "actual_file_reads": ["inline-table"],
                "program_summary": program,
                "execution_status": "ok",
                "recomputed_output": recomputed,
                "cited_operands": operands.get("cited_operands") or [],
                "cited_constants": operands.get("cited_consts") or [],
                "reference_label": reference,
            })
    return out


def build_human():
    bases = {
        row["taskid"]: row
        for row in load_jsonl(RECORDS / "kb_labels_rederived.jsonl")
        if row.get("_setting") == "discovery"
    }
    with (DOCS / "annotation" / "KB_SUPPORT_ANNOTATION.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        annotations = list(csv.DictReader(handle))
    out = []
    for row in annotations:
        base = bases.get(row["taskid"], {})
        reference = {
            "yes": "supported",
            "no": "unsupported",
            "unclear": "unknown",
        }[row["human_support_label"].strip().lower()]
        operand_descriptions = [
            value.strip()
            for value in row.get("cited_operands", "").replace("|", ";").split(";")
            if value.strip()
        ]
        out.append({
            "dataset": "kramabench_human",
            "item_id": row["taskid"],
            "question": row["question"],
            "stated_answer": row["stated_answer"],
            "declared_files": [
                value for value in row.get("cited_files", "").split(";") if value
            ],
            # The archive lacks the underlying KramaBench tables for most cases.
            # Column/operand descriptions are retained as compact evidence, and
            # this limitation is explicitly reported rather than using the human
            # adjudication note (which would leak the label).
            "evidence_snippets": [
                f"declared operand or column: {value}" for value in operand_descriptions
            ],
            "actual_file_reads": base.get("files_read") or [],
            "program_summary": (
                f"Archived program path: {row['snippet_paths']}"
                if row.get("snippet_paths")
                else "Program body unavailable in transferred archive."
            ),
            "execution_status": row["exec_status"],
            "recomputed_output": row["recomputed_value"],
            "cited_operands": operand_descriptions,
            "cited_constants": [],
            "reference_label": reference,
            "evidence_limitation": "raw cited rows unavailable for most transferred live cases",
        })
    return out


def build_wikisql():
    path = RECORDS / "wikisql_semantic_validation.jsonl"
    if not path.exists():
        return []
    table_path = ROOT / "external" / "raw" / "wikisql" / "data" / "dev.tables.jsonl"
    tables = {
        table["id"]: table
        for table in (
            json.loads(line)
            for line in table_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    by_class = defaultdict(list)
    for row in load_jsonl(path):
        if row["perturbation_class"] != "same_original_output_counterfactual":
            by_class[row["perturbation_class"]].append(row)
    classes = sorted(by_class)
    selected = []
    cursor = 0
    while len(selected) < 50 and classes:
        perturb_class = classes[cursor % len(classes)]
        index = cursor // len(classes)
        if index < len(by_class[perturb_class]):
            selected.append(by_class[perturb_class][index])
        cursor += 1
    out = []
    for row in selected:
        table = tables[row["table_id"]]
        snippets = [
            " | ".join(map(str, table["header"])),
            *[" | ".join(map(str, values)) for values in table["rows"][:4]],
        ]
        out.append({
            "dataset": "wikisql_semantic_perturbations",
            "item_id": row["case_id"],
            "question": row["question"],
            "stated_answer": row["candidate_output"],
            "declared_files": [f"dev.db::{row['table_id']}"],
            "evidence_snippets": snippets,
            "actual_file_reads": [f"dev.db::{row['table_id']}"],
            "program_summary": row["candidate_sql"],
            "execution_status": "ok",
            "recomputed_output": row["candidate_output"],
            "cited_operands": [],
            "cited_constants": [],
            "reference_label": "unsupported",
            "perturbation_class": row["perturbation_class"],
        })
    return out


def main():
    rows = build_human() + build_convfinqa() + build_wikisql()
    write_jsonl(RECORDS / "evidence_aware_judge_inputs.jsonl", rows)
    print(f"wrote {len(rows)} judge inputs")


if __name__ == "__main__":
    main()
