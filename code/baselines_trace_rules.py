#!/usr/bin/env python3
"""Construct ordinary trace/provenance-rule decisions without using LEDGER verdict fields."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from common import (
    DOCS,
    RECORDS,
    extract_recomputed,
    load_jsonl,
    values_equal,
    write_jsonl,
)


METHODS = (
    "citations_resolve",
    "declared_covers_actual_reads",
    "execution_success",
    "comparable_output_exists",
    "unrestricted_replay_match",
    "strict_locatability",
    "restricted_replay_match",
)


def _canonical_kb_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(RECORDS.glob("records_ledger_n2_kb_*.jsonl")):
        match = re.search(r"_kb_([^_]+)_(oracle|discovery)\.jsonl$", path.name)
        if not match:
            continue
        domain, setting = match.groups()
        for row in load_jsonl(path):
            rows.append({**row, "_domain": domain, "_setting": setting})
    return rows


def _old_gate_map():
    return {
        (row["taskid"], row.get("setting", "discovery")): not bool(row["old_gate_blocked"])
        for row in load_jsonl(RECORDS / "old_gate_replay_kb.jsonl")
    }


def _basic_decisions(row: dict[str, Any], strict_locatable: bool | None = None):
    cited = {str(value) for value in row.get("files_cited") or []}
    read = {str(value) for value in row.get("files_read") or []}
    offered = {str(value) for value in row.get("offered_files") or []}
    citations_resolve = bool(cited) and (
        not offered or all(value in offered or value in read for value in cited)
    )
    declared_covers = bool(read) and read.issubset(cited)
    exec_value = row.get("exec")
    if isinstance(exec_value, dict):
        execution_success = exec_value.get("returncode") == 0 and not exec_value.get("timeout")
    else:
        execution_success = not bool(row.get("exec_failed"))
    recomputed = extract_recomputed(row.get("support_detail"))
    comparable = row.get("pred") not in (None, "") and recomputed is not None
    unrestricted = (
        execution_success
        and comparable
        and values_equal(row.get("pred"), recomputed)
    )
    restricted = unrestricted and citations_resolve and declared_covers
    return {
        "citations_resolve": citations_resolve,
        "declared_covers_actual_reads": declared_covers,
        "execution_success": execution_success,
        "comparable_output_exists": comparable,
        "unrestricted_replay_match": unrestricted,
        "strict_locatability": strict_locatable,
        "restricted_replay_match": restricted,
    }


def build_kramabench():
    old_gate = _old_gate_map()
    out = []
    for row in _canonical_kb_rows():
        setting = row["_setting"]
        decisions = _basic_decisions(
            row, old_gate.get((row["taskid"], setting))
        )
        out.append(
            {
                "dataset": "kramabench_all",
                "case_id": f"{row['taskid']}::{setting}",
                "taskid": row["taskid"],
                "domain": row["_domain"],
                "setting": setting,
                "reference_label": None,
                "official_score": row.get("score"),
                "decisions": decisions,
            }
        )
    return out


def build_human():
    kb_map = {
        row["taskid"]: row
        for row in load_jsonl(RECORDS / "kb_labels_rederived.jsonl")
        if row.get("_setting") == "discovery"
    }
    old_gate = _old_gate_map()
    with (DOCS / "annotation" / "KB_SUPPORT_ANNOTATION.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        annotations = list(csv.DictReader(handle))
    out = []
    for annotation in annotations:
        taskid = annotation["taskid"]
        base = kb_map.get(taskid, {})
        cited = [x for x in annotation["cited_files"].split(";") if x]
        recomputed = annotation["recomputed_value"].strip().strip("'\"")
        comparable = recomputed.lower() not in {"", "none", "null", "nan"}
        unrestricted = (
            annotation["exec_status"].strip().lower() == "ok"
            and comparable
            and values_equal(annotation["stated_answer"], recomputed)
        )
        offered = set(base.get("offered_files") or [])
        reads = set(base.get("files_read") or [])
        cited_set = set(cited)
        citations_resolve = bool(cited_set) and (
            not offered or all(path in offered or path in reads for path in cited_set)
        )
        declared_covers = bool(reads) and reads.issubset(cited_set)
        decisions = {
            "citations_resolve": citations_resolve,
            "declared_covers_actual_reads": declared_covers,
            "execution_success": annotation["exec_status"].strip().lower() == "ok",
            "comparable_output_exists": comparable,
            "unrestricted_replay_match": unrestricted,
            "strict_locatability": old_gate.get((taskid, "discovery")),
            "restricted_replay_match": (
                unrestricted and citations_resolve and declared_covers
            ),
        }
        label_text = annotation["human_support_label"].strip().lower()
        reference = {"yes": 1, "no": 0}.get(label_text)
        out.append(
            {
                "dataset": "kramabench_human",
                "case_id": taskid,
                "taskid": taskid,
                "domain": annotation["domain"],
                "setting": "discovery",
                "reference_label": reference,
                "reference_label_raw": label_text,
                "official_score": float(annotation["official_score"]),
                "decisions": decisions,
            }
        )
    return out


def build_dabstep():
    out = []
    for row in load_jsonl(RECORDS / "records_ledger_dabstep_support.jsonl"):
        decisions = _basic_decisions(row, None)
        out.append(
            {
                "dataset": "dabstep",
                "case_id": row["taskid"],
                "taskid": row["taskid"],
                "domain": row.get("level"),
                "setting": "fixed_files",
                "reference_label": None,
                "official_score": row.get("score"),
                "decisions": decisions,
            }
        )
    return out


def build_controlled():
    out = []
    for row in load_jsonl(RECORDS / "kb_perturbation_suite.jsonl"):
        kind = row["perturb_class"]
        citations = kind != "c2"
        covers = kind != "c1"
        execution = kind not in {"b", "d", "unknown"}
        comparable = kind not in {"a2", "b", "d", "unknown"}
        unrestricted = kind in {"c1", "c2"}
        restricted = unrestricted and citations and covers
        out.append(
            {
                "dataset": "kramabench_controlled",
                "case_id": f"{row['base_taskid']}::{row['setting']}::{kind}",
                "taskid": row["base_taskid"],
                "domain": row["domain"],
                "setting": row["setting"],
                "reference_label": None if kind == "unknown" else 0,
                "reference_label_raw": kind,
                "official_score": None,
                "decisions": {
                    "citations_resolve": citations,
                    "declared_covers_actual_reads": covers,
                    "execution_success": execution,
                    "comparable_output_exists": comparable,
                    "unrestricted_replay_match": unrestricted,
                    "strict_locatability": None,
                    "restricted_replay_match": restricted,
                },
            }
        )
    return out


def main():
    rows = build_human() + build_kramabench() + build_dabstep() + build_controlled()
    write_jsonl(RECORDS / "trace_rule_baselines.jsonl", rows)
    print(f"wrote {len(rows)} cases")


if __name__ == "__main__":
    main()
