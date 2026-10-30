#!/usr/bin/env python3
"""Measure deterministic audit overhead on a fixed 100-record sample."""

from __future__ import annotations

import json
import time

import numpy as np

from common import (
    ARTIFACTS,
    RECORDS,
    extract_recomputed,
    load_jsonl,
    values_equal,
    write_json,
)
from kramabench_integration.provenance import ConstrainedPool


REPEATS = 5
N = 100


def describe(values):
    data = np.asarray(values, dtype=float) * 1000.0
    return {
        "n": len(data),
        "mean_ms": float(data.mean()),
        "median_ms": float(np.median(data)),
        "p90_ms": float(np.quantile(data, 0.90)),
        "p95_ms": float(np.quantile(data, 0.95)),
        "max_ms": float(data.max()),
    }


def main():
    raw_lines = [
        line for line in
        (RECORDS / "records_ledger_checker_convfinqa_full.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ][:N]
    numeric = [json.loads(line) for line in raw_lines]
    kb = []
    for path in sorted(RECORDS.glob("records_ledger_n2_kb_*.jsonl")):
        kb.extend(load_jsonl(path))
    kb = kb[:N]
    timings = {
        "trace_parsing": [],
        "citation_resolution": [],
        "restricted_replay": [],
        "operand_extraction": [],
        "closure": [],
        "record_serialization": [],
        "total_audit": [],
    }
    repeat_decisions = []
    for _ in range(REPEATS):
        decisions = []
        for index in range(N):
            total_started = time.perf_counter()
            started = time.perf_counter()
            parsed = json.loads(raw_lines[index])
            timings["trace_parsing"].append(time.perf_counter() - started)

            kb_row = kb[index]
            started = time.perf_counter()
            cited = set(kb_row.get("files_cited") or [])
            read = set(kb_row.get("files_read") or [])
            offered = set(kb_row.get("offered_files") or [])
            citations_resolve = bool(cited) and all(
                path in offered or path in read for path in cited
            )
            timings["citation_resolution"].append(time.perf_counter() - started)

            started = time.perf_counter()
            recomputed = extract_recomputed(kb_row.get("support_detail"))
            replay = (
                not kb_row.get("exec_failed")
                and recomputed is not None
                and values_equal(kb_row.get("pred"), recomputed)
                and citations_resolve
                and bool(read)
                and read.issubset(cited)
            )
            timings["restricted_replay"].append(time.perf_counter() - started)

            started = time.perf_counter()
            provenance = (parsed.get("provenance") or [{}])[0]
            operands = list(provenance.get("cited_operands") or [])
            consts = list(provenance.get("cited_consts") or [])
            timings["operand_extraction"].append(time.perf_counter() - started)

            started = time.perf_counter()
            pool = ConstrainedPool(
                operands,
                consts,
                ("sum", "diff", "ratio", "pctchg", "prod"),
                chain="full",
            )
            closure = bool(pool.check(
                parsed.get("pred"),
                tol_rel=1e-4,
                percent=True,
                percent_ctx=bool(parsed.get("percent_ctx")),
                max_depth=2,
            ))
            timings["closure"].append(time.perf_counter() - started)

            record = {
                "case_id": parsed["taskid"],
                "citations_resolve": citations_resolve,
                "restricted_replay_match": replay,
                "closure_match": closure,
            }
            started = time.perf_counter()
            json.dumps(record, sort_keys=True)
            timings["record_serialization"].append(time.perf_counter() - started)
            timings["total_audit"].append(time.perf_counter() - total_started)
            decisions.append((replay, closure))
        repeat_decisions.append(decisions)
    reference = repeat_decisions[0]
    agreements = [
        sum(current == expected for current, expected in zip(run, reference)) / N
        for run in repeat_decisions[1:]
    ]
    judge_path = RECORDS / "evidence_aware_judge_summary.json"
    judge = json.load(judge_path.open()) if judge_path.exists() else {}
    payload = {
        "schema_version": 1,
        "sample_rule": "first 100 ConvFinQA remainder and first 100 canonical KramaBench records",
        "repeats": REPEATS,
        "components": {name: describe(values) for name, values in timings.items()},
        "deterministic_repeat_agreement": {
            "pairwise_to_first": agreements,
            "all_exact": all(value == 1.0 for value in agreements),
            "records_per_repeat": N,
        },
        "model_calls": 0,
        "llm_judge_calls": judge.get("calls"),
        "llm_judge_tokens": (
            (judge.get("tokens_in") or 0) + (judge.get("tokens_out") or 0)
            if judge else None
        ),
    }
    write_json(RECORDS / "audit_runtime_summary.json", payload)
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Audit component & Median (ms) & p95 (ms) \\",
        r"\midrule",
    ]
    for name, summary in payload["components"].items():
        lines.append(
            f"{name.replace('_', ' ')} & {summary['median_ms']:.3f} & "
            f"{summary['p95_ms']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ARTIFACTS / "table_runtime.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
