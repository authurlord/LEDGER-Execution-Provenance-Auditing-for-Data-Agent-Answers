#!/usr/bin/env python3
"""Deterministic CPU-only E0 scorer sanity audit for LEDGER KramaBench records."""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RECORD_GLOB = "records/records_ledger_n2_kb_*_*.jsonl"
FLAGS_PATH = ROOT / "records" / "score_sanity_flags.jsonl"
REPORT_PATH = ROOT / "docs" / "SCORE_SANITY_REPORT.md"
ADJUDICATION_PATH = ROOT / "docs" / "annotation" / "SCORE_ADJUDICATION.csv"
EXPECTED_FILES = 8
EXPECTED_RECORDS = 166


@dataclass(frozen=True)
class EqResult:
    equal: bool
    match_kind: str
    ordered_list_equal: bool = False
    unordered_list_equal: bool = False


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, Decimal)) and not isinstance(x, bool)


def _as_decimal(x: Any) -> Decimal | None:
    if _is_number(x):
        try:
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return Decimal(str(x))
        except (InvalidOperation, ValueError):
            return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return Decimal(s)
        except InvalidOperation:
            return None
    return None


def _number_close(a: Any, b: Any) -> bool:
    da = _as_decimal(a)
    db = _as_decimal(b)
    if da is None or db is None:
        return False
    return abs(da - db) <= Decimal("0.000001")


def _norm_scalar(x: Any) -> Any:
    if isinstance(x, str):
        return x.strip().lower()
    if _is_number(x):
        d = _as_decimal(x)
        if d is not None:
            return d.normalize()
    return x


def _canonical_for_unordered(x: Any) -> str:
    if isinstance(x, list):
        normalized = [_canonical_for_unordered(v) for v in x]
        return json.dumps({"list": normalized}, sort_keys=True, ensure_ascii=False)
    if isinstance(x, tuple):
        normalized = [_canonical_for_unordered(v) for v in x]
        return json.dumps({"list": normalized}, sort_keys=True, ensure_ascii=False)
    number = _as_decimal(x)
    if number is not None:
        return json.dumps({"number": str(number.normalize())}, sort_keys=True, ensure_ascii=False)
    return json.dumps({"scalar": _norm_scalar(x)}, sort_keys=True, ensure_ascii=False, default=str)


def _norm_value(x: Any) -> Any:
    if isinstance(x, list):
        return [_norm_value(v) for v in x]
    if isinstance(x, tuple):
        return [_norm_value(v) for v in x]
    return _norm_scalar(x)


def _scalar_eq(a: Any, b: Any) -> EqResult:
    if a is None or b is None:
        return EqResult(False, "none")
    if _number_close(a, b):
        if _is_number(a) and _is_number(b):
            return EqResult(True, "numeric_tolerance")
        if _is_number(a) != _is_number(b):
            return EqResult(True, "numeric_string_coercion")
    if isinstance(a, str) and isinstance(b, str):
        if a.strip().lower() == b.strip().lower():
            return EqResult(True, "string_normalized")
    if a == b:
        return EqResult(True, "exact")
    return EqResult(False, "different")


def compare_norm(a: Any, b: Any) -> EqResult:
    """Normalize and compare values; list comparisons report ordered/unordered hits."""
    if a is None or b is None:
        return EqResult(False, "none")
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)):
            return EqResult(False, "list_vs_nonlist")
        ordered = len(a) == len(b) and all(compare_norm(x, y).equal for x, y in zip(a, b))
        unordered = False
        if len(a) == len(b):
            left = sorted(_canonical_for_unordered(x) for x in a)
            right = sorted(_canonical_for_unordered(x) for x in b)
            unordered = left == right
        if ordered:
            return EqResult(True, "list_ordered", True, unordered)
        if unordered:
            return EqResult(True, "list_unordered_only", False, True)
        return EqResult(False, "list_different", False, unordered)
    return _scalar_eq(a, b)


def eq_norm(a: Any, b: Any) -> bool:
    return compare_norm(a, b).equal


def _has_numeric_string_coercion(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return any(_has_numeric_string_coercion(x, y) for x, y in zip(a, b))
    return (_is_number(a) and isinstance(b, str) and _number_close(a, b)) or (
        isinstance(a, str) and _is_number(b) and _number_close(a, b)
    )


def _has_float_format(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return any(_has_float_format(x, y) for x, y in zip(a, b))
    if _is_number(a) and _is_number(b) and a != b and _number_close(a, b):
        return True
    if isinstance(a, str) and isinstance(b, str) and a.strip() != b.strip() and _number_close(a, b):
        return True
    return False


def _has_string_case(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return any(_has_string_case(x, y) for x, y in zip(a, b))
    return (
        isinstance(a, str)
        and isinstance(b, str)
        and a.strip().lower() == b.strip().lower()
        and a.strip() != b.strip()
    )


def explain_scorer_anomaly(pred: Any, gold: Any, eq: EqResult) -> str:
    if eq.match_kind == "list_unordered_only":
        return "list_order_difference"
    if _has_numeric_string_coercion(pred, gold):
        return "type_coercion_numeric_string"
    if _has_float_format(pred, gold):
        return "float_format"
    if _has_string_case(pred, gold):
        return "string_case"
    return "other"


def mode_from_path(path: str) -> str:
    name = os.path.basename(path)
    return name.rsplit("_", 1)[-1].removesuffix(".jsonl")


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def is_zero_score(value: Any) -> bool:
    try:
        if isinstance(value, float) and math.isnan(value):
            return False
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def load_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    paths = sorted(glob.glob(str(ROOT / RECORD_GLOB)))
    seen_by_mode: dict[tuple[str, str], str] = {}
    seen_by_file: dict[tuple[str, str], int] = defaultdict(int)

    for path in paths:
        mode = mode_from_path(path)
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append({"type": "json_decode_error", "file": path, "line": line_no, "detail": str(exc)})
                    continue
                taskid = rec.get("taskid")
                if not taskid:
                    errors.append({"type": "missing_taskid", "file": path, "line": line_no, "detail": ""})
                else:
                    file_key = (path, str(taskid))
                    seen_by_file[file_key] += 1
                    if seen_by_file[file_key] > 1:
                        errors.append(
                            {"type": "duplicate_taskid_in_file", "file": path, "line": line_no, "taskid": taskid}
                        )
                    mode_key = (mode, str(taskid))
                    previous = seen_by_mode.get(mode_key)
                    if previous and previous != path:
                        errors.append(
                            {
                                "type": "duplicate_taskid_in_mode",
                                "file": path,
                                "line": line_no,
                                "taskid": taskid,
                                "previous_file": previous,
                            }
                        )
                    seen_by_mode.setdefault(mode_key, path)
                rec["_file"] = os.path.basename(path)
                rec["_path"] = path
                rec["_line"] = line_no
                rec["_mode"] = mode
                rows.append(rec)
    return rows, errors


def make_flag_row(rec: dict[str, Any], flag: str, explanation: str, eq: EqResult) -> dict[str, Any]:
    return {
        "file": rec.get("_file"),
        "line": rec.get("_line"),
        "taskid": rec.get("taskid"),
        "domain": rec.get("domain"),
        "mode": rec.get("_mode"),
        "pred": rec.get("pred"),
        "gold": rec.get("gold"),
        "score": rec.get("score"),
        "grounding_label": rec.get("grounding_label"),
        "unsupported_class": rec.get("unsupported_class"),
        "support_detail": rec.get("support_detail"),
        "provenance": rec.get("provenance"),
        "offered_files": rec.get("offered_files"),
        "trace_path": rec.get("trace_path"),
        "flag": flag,
        "explanation": explanation,
        "eq_match_kind": eq.match_kind,
        "list_ordered_equal": eq.ordered_list_equal,
        "list_unordered_equal": eq.unordered_list_equal,
    }


def build_report(
    rows: list[dict[str, Any]],
    flags: list[dict[str, Any]],
    join_errors: list[dict[str, Any]],
    paths_count: int,
) -> str:
    anomaly_rows = [r for r in flags if "pred_equals_gold_but_score_zero" in r["flag"]]
    unsupported_rows = [r for r in flags if "unsupported_flag" in r["flag"]]
    category_counts = Counter(r["explanation"] for r in anomaly_rows)
    mode_taskids = {(r.get("_mode"), r.get("taskid")) for r in rows if r.get("taskid")}
    bare_task_counts = Counter(r.get("taskid") for r in rows if r.get("taskid"))
    cross_mode_repeats = sum(1 for _, count in bare_task_counts.items() if count > 1)

    lines = [
        "# SCORE SANITY REPORT",
        "",
        "## Summary",
        "",
        f"- Input glob: `{RECORD_GLOB}`",
        f"- Files read: {paths_count} (expected {EXPECTED_FILES})",
        f"- Total records: {len(rows)} (expected {EXPECTED_RECORDS})",
        f"- Unique `(mode, taskid)` records: {len(mode_taskids)}",
        f"- Taskids appearing in both modes: {cross_mode_repeats} (expected for oracle/discovery pairs; not a G0 failure)",
        f"- `pred == gold` after normalization and `score == 0.0`: {len(anomaly_rows)}",
        f"- Unsupported flags: {len(unsupported_rows)}",
        f"- G0 record-join integrity: {'FAIL' if join_errors else 'PASS'}",
        "",
        "## Anomaly Classification",
        "",
    ]
    if category_counts:
        for key, value in sorted(category_counts.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none: 0")

    lines.extend(["", "## `pred == gold` and `score == 0.0` Rows", ""])
    if anomaly_rows:
        lines.append("| taskid | mode | domain | match | ordered list hit | unordered list hit | explanation | pred | gold |")
        lines.append("|---|---|---|---|---:|---:|---|---|---|")
        for row in anomaly_rows:
            lines.append(
                "| {taskid} | {mode} | {domain} | {match} | {ordered} | {unordered} | {explanation} | `{pred}` | `{gold}` |".format(
                    taskid=row.get("taskid"),
                    mode=row.get("mode"),
                    domain=row.get("domain"),
                    match=row.get("eq_match_kind"),
                    ordered=row.get("list_ordered_equal"),
                    unordered=row.get("list_unordered_equal"),
                    explanation=row.get("explanation"),
                    pred=json_cell(row.get("pred")),
                    gold=json_cell(row.get("gold")),
                )
            )
    else:
        lines.append("No rows matched this condition.")

    lines.extend(["", "## Unsupported Flags", ""])
    if unsupported_rows:
        lines.append("| taskid | mode | domain | class | pred | gold | score | support_detail |")
        lines.append("|---|---|---|---|---|---|---:|---|")
        for row in unsupported_rows:
            detail = str(row.get("support_detail") or "").replace("|", "\\|")
            lines.append(
                "| {taskid} | {mode} | {domain} | {cls} | `{pred}` | `{gold}` | {score} | {detail} |".format(
                    taskid=row.get("taskid"),
                    mode=row.get("mode"),
                    domain=row.get("domain"),
                    cls=row.get("unsupported_class"),
                    pred=json_cell(row.get("pred")),
                    gold=json_cell(row.get("gold")),
                    score=row.get("score"),
                    detail=detail,
                )
            )
    else:
        lines.append("No unsupported flags.")

    lines.extend(["", "## Record-Join Integrity", ""])
    if join_errors:
        lines.append("G0 FAIL: missing or duplicate task identifiers were found.")
        lines.append("")
        lines.append("| type | file | line | taskid | detail |")
        lines.append("|---|---|---:|---|---|")
        for err in join_errors:
            lines.append(
                "| {type} | `{file}` | {line} | {taskid} | {detail} |".format(
                    type=err.get("type"),
                    file=os.path.relpath(err.get("file", ""), ROOT) if err.get("file") else "",
                    line=err.get("line", ""),
                    taskid=err.get("taskid", ""),
                    detail=str(err.get("detail") or err.get("previous_file") or "").replace("|", "\\|"),
                )
            )
    else:
        lines.append("G0 PASS: no missing taskid, no duplicate taskid within a file, and no duplicate `(mode, taskid)` joins.")

    lines.extend(
        [
            "",
            "## Scale Note",
            "",
            "Expected scale from the brief is about 9 scorer anomalies plus 3 unsupported flags, with `legal-hard-30` hitting both categories. Actual counts above are reported directly rather than forced to that expectation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    paths = sorted(glob.glob(str(ROOT / RECORD_GLOB)))
    rows, join_errors = load_records()
    flags: list[dict[str, Any]] = []

    for rec in rows:
        eq = compare_norm(rec.get("pred"), rec.get("gold"))
        flags_for_row: list[str] = []
        explanation = ""
        if eq.equal and rec.get("pred") is not None and is_zero_score(rec.get("score")):
            flags_for_row.append("pred_equals_gold_but_score_zero")
            explanation = explain_scorer_anomaly(rec.get("pred"), rec.get("gold"), eq)
        if rec.get("grounding_label") == "unsupported":
            flags_for_row.append("unsupported_flag")
            if not explanation:
                explanation = "unsupported_flag_only"
        if flags_for_row:
            flags.append(make_flag_row(rec, ";".join(flags_for_row), explanation, eq))

    flags.sort(key=lambda r: (str(r.get("mode")), str(r.get("domain")), str(r.get("taskid")), str(r.get("file"))))

    FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FLAGS_PATH, "w", encoding="utf-8") as handle:
        for row in flags:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(rows, flags, join_errors, len(paths)), encoding="utf-8")

    ADJUDICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ADJUDICATION_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "taskid",
                "pred",
                "gold",
                "score",
                "scorer_explanation",
                "human_correctness_label",
                "note",
            ],
        )
        writer.writeheader()
        for row in flags:
            writer.writerow(
                {
                    "taskid": row.get("taskid"),
                    "pred": json_cell(row.get("pred")),
                    "gold": json_cell(row.get("gold")),
                    "score": row.get("score"),
                    "scorer_explanation": row.get("explanation"),
                    "human_correctness_label": "",
                    "note": "",
                }
            )

    anomaly_count = sum(1 for r in flags if "pred_equals_gold_but_score_zero" in r["flag"])
    unsupported_count = sum(1 for r in flags if "unsupported_flag" in r["flag"])
    category_counts = Counter(r["explanation"] for r in flags if "pred_equals_gold_but_score_zero" in r["flag"])

    print("score_sanity_audit")
    print(f"files_read={len(paths)} records={len(rows)} expected_records={EXPECTED_RECORDS}")
    print(f"flagged_records={len(flags)} pred_equals_gold_score_zero={anomaly_count} unsupported_flags={unsupported_count}")
    print("anomaly_categories=" + json.dumps(dict(sorted(category_counts.items())), sort_keys=True))
    print(f"record_join_errors={len(join_errors)} g0={'FAIL' if join_errors else 'PASS'}")
    print(f"wrote={FLAGS_PATH.relative_to(ROOT)}")
    print(f"wrote={REPORT_PATH.relative_to(ROOT)}")
    print(f"wrote={ADJUDICATION_PATH.relative_to(ROOT)}")

    return 1 if join_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
