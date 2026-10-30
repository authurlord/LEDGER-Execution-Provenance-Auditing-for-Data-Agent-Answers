#!/usr/bin/env python3
"""Build leakage-audited pre-verdict features for ordinary anomaly baselines."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

from common import DOCS, RECORDS, finite_float, load_jsonl, relative_distance, write_jsonl
from kramabench_integration.provenance import ConstrainedPool


FEATURES = (
    "log_abs_stated_answer",
    "num_declared_files",
    "num_actual_files",
    "citation_resolution_fraction",
    "undeclared_read_count",
    "execution_success",
    "comparable_output_exists",
    "num_operands",
    "min_relative_distance_to_operand",
    "median_relative_distance_to_operands",
    "exact_operand_match",
    "one_step_derivable",
    "output_type_numeric",
    "program_length",
    "num_tool_calls",
)


def _feature_row(
    *,
    value: Any,
    operands: list[Any],
    consts: list[Any],
    cited: list[str],
    actual: list[str],
    offered: list[str],
    execution_success: bool,
    comparable: bool,
    program: str,
    tool_calls: Any,
    percent_ctx: bool,
) -> dict[str, float]:
    number = finite_float(value)
    numeric_operands = [x for x in (finite_float(v) for v in operands) if x is not None]
    distances = sorted(
        abs(number - operand) / max(abs(number), 1.0)
        for operand in numeric_operands
    ) if number is not None else []
    cited_set, actual_set, offered_set = set(cited), set(actual), set(offered)
    resolved = sum(path in offered_set or path in actual_set for path in cited_set)
    pool = ConstrainedPool(numeric_operands, consts, ("sum", "diff", "ratio", "pctchg", "prod"))
    exact = bool(number is not None and pool.check(
        number, tol_rel=1e-4, percent=True, percent_ctx=percent_ctx, max_depth=0
    ))
    one_step = bool(number is not None and pool.check(
        number, tol_rel=1e-4, percent=True, percent_ctx=percent_ctx, max_depth=1
    ))
    return {
        "log_abs_stated_answer": math.log1p(abs(number)) if number is not None else 0.0,
        "num_declared_files": float(len(cited_set)),
        "num_actual_files": float(len(actual_set)),
        "citation_resolution_fraction": (
            resolved / len(cited_set) if cited_set else 0.0
        ),
        "undeclared_read_count": float(len(actual_set - cited_set)),
        "execution_success": float(bool(execution_success)),
        "comparable_output_exists": float(bool(comparable)),
        "num_operands": float(len(numeric_operands)),
        "min_relative_distance_to_operand": min(distances) if distances else 1.0,
        "median_relative_distance_to_operands": (
            distances[len(distances) // 2] if distances else 1.0
        ),
        "exact_operand_match": float(exact),
        "one_step_derivable": float(one_step),
        "output_type_numeric": float(number is not None),
        "program_length": float(len(re.findall(r"\w+|[^\s\w]", program or ""))),
        "num_tool_calls": float(finite_float(tool_calls) or 0.0),
    }


def _constructed_rows(path: Path, split: str, distribution: str) -> list[dict[str, Any]]:
    out = []
    for row in load_jsonl(path):
        provenance = (row.get("provenance") or [{}])[0]
        taskid = str(row["taskid"])
        if taskid.endswith("::positive"):
            label = 1
        elif taskid.endswith("::negative") or taskid.endswith("::wrong_operation_negative"):
            label = 0
        else:
            continue
        out.append({
            "case_id": taskid,
            "dataset": distribution,
            "split": split,
            "reference_label": label,
            "features": _feature_row(
                value=row.get("pred"),
                operands=provenance.get("cited_operands") or [],
                consts=provenance.get("cited_consts") or [],
                cited=row.get("files_cited") or [],
                actual=row.get("files_read") or [],
                offered=row.get("offered_files") or [],
                execution_success=not bool(row.get("exec_failed")),
                comparable=row.get("pred") not in (None, ""),
                program=str(row.get("program") or row.get("mutated_program") or ""),
                tool_calls=row.get("tool_calls"),
                percent_ctx=bool(row.get("percent_ctx")),
            ),
        })
    return out


def _wrong_operation_rows() -> list[dict[str, Any]]:
    """Create matched original/mutated pairs from the same operands and program."""

    out = []
    for row in load_jsonl(RECORDS / "convfinqa_wrong_operation_negatives.jsonl"):
        provenance = (row.get("provenance") or [{}])[0]
        shared = {
            "operands": provenance.get("cited_operands") or [],
            "consts": provenance.get("cited_consts") or [],
            "cited": row.get("files_cited") or [],
            "actual": row.get("files_read") or [],
            "offered": row.get("offered_files") or [],
            "execution_success": not bool(row.get("exec_failed")),
            "comparable": True,
            "tool_calls": row.get("tool_calls"),
            "percent_ctx": bool(row.get("percent_ctx")),
        }
        base = str(row["taskid"]).removesuffix("::wrong_operation_negative")
        out.append({
            "case_id": f"{base}::original_program_positive",
            "dataset": "convfinqa_wrong_operation",
            "split": "semantic_boundary_test",
            "reference_label": 1,
            "features": _feature_row(
                value=row.get("exe_ans"),
                program=str(row.get("program") or ""),
                **shared,
            ),
        })
        out.append({
            "case_id": f"{base}::wrong_operation_negative",
            "dataset": "convfinqa_wrong_operation",
            "split": "semantic_boundary_test",
            "reference_label": 0,
            "features": _feature_row(
                value=row.get("pred"),
                program=str(row.get("mutated_program") or ""),
                **shared,
            ),
        })
    return out


def _human_rows() -> list[dict[str, Any]]:
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
        label = {"yes": 1, "no": 0}.get(row["human_support_label"].strip().lower())
        if label is None:
            continue
        base = bases.get(row["taskid"], {})
        operands = [
            value.strip()
            for value in re.split(r"[;,|]", row.get("cited_operands", ""))
            if value.strip()
        ]
        recomputed = row.get("recomputed_value", "").strip().lower()
        out.append({
            "case_id": row["taskid"],
            "dataset": "kramabench_human",
            "split": "zero_tune_test",
            "reference_label": label,
            "features": _feature_row(
                value=row.get("stated_answer"),
                operands=operands,
                consts=[],
                cited=[v for v in row.get("cited_files", "").split(";") if v],
                actual=base.get("files_read") or [],
                offered=base.get("offered_files") or [],
                execution_success=row.get("exec_status", "").strip().lower() == "ok",
                comparable=recomputed not in {"", "none", "null", "nan"},
                program="",
                tool_calls=base.get("tool_calls"),
                percent_ctx="%" in str(row.get("question", "")),
            ),
        })
    return out


def main() -> None:
    rows = []
    rows.extend(_constructed_rows(
        RECORDS / "records_ledger_checker_convfinqa.jsonl",
        "calibration",
        "convfinqa_seed",
    ))
    rows.extend(_constructed_rows(
        RECORDS / "records_ledger_checker_convfinqa_full.jsonl",
        "test",
        "convfinqa_remainder",
    ))
    rows.extend(_constructed_rows(
        RECORDS / "records_ledger_checker_multihiertt.jsonl",
        "transfer_test",
        "multihiertt",
    ))
    rows.extend(_wrong_operation_rows())
    rows.extend(_human_rows())
    write_jsonl(RECORDS / "audit_feature_matrix.jsonl", rows)
    print(f"wrote {len(rows)} rows with {len(FEATURES)} pre-verdict features")


if __name__ == "__main__":
    main()
