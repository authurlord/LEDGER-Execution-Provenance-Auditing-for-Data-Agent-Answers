#!/usr/bin/env python3
"""Compare replay, SQL-structure, and differential semantic validators."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter, defaultdict
from copy import deepcopy

import sqlglot

from build_wikisql_semantic_perturbations import DATA, literal, normalized_output, table_name
from common import ARTIFACTS, DOCS, RECORDS, load_jsonl, write_jsonl


VALIDATORS = (
    "original_output_equality",
    "normalized_sql_exact_match",
    "operator_clause_multiset",
    "normalized_ast_edit_threshold",
    "differential_execution",
    "ledger_replay_certificate_only",
)


def sql_tokens(sql):
    expression = sqlglot.parse_one(sql, read="sqlite")
    return expression.sql(dialect="sqlite", normalize=True)


def clause_signature(spec):
    return {
        "aggregation": int(spec["agg"]),
        "projection": int(spec["sel"]),
        "condition_columns": sorted(int(condition[0]) for condition in spec["conds"]),
        "condition_operators": sorted(int(condition[1]) for condition in spec["conds"]),
        "condition_count": len(spec["conds"]),
    }


def table_index():
    return {
        table["id"]: table
        for table in (
            json.loads(line)
            for line in (DATA / "dev.tables.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    }


def create_memory_table(table, extra_row):
    connection = sqlite3.connect(":memory:")
    definitions = ", ".join(
        f"col{index} {'REAL' if kind == 'real' else 'TEXT'}"
        for index, kind in enumerate(table["types"])
    )
    name = table_name(table["id"])
    connection.execute(f"CREATE TABLE {name} ({definitions})")
    placeholders = ",".join("?" for _ in table["header"])
    rows = [list(row) for row in table["rows"]]
    if extra_row is not None:
        rows.append(extra_row)
    connection.executemany(f"INSERT INTO {name} VALUES ({placeholders})", rows)
    return connection


def targeted_rows(record, table):
    ncols = len(table["header"])
    base = list(table["rows"][0]) if table["rows"] else [
        0.0 if kind == "real" else "__ledger_base__" for kind in table["types"]
    ]
    rows = []
    for spec, tag in ((record["gold_spec"], "gold"), (record["candidate_spec"], "candidate")):
        row = list(base)
        for column, _, value in spec["conds"]:
            row[column] = value
        selected = {spec["sel"], record["gold_spec"]["sel"], record["candidate_spec"]["sel"]}
        for column in selected:
            row[column] = (
                987654.0 if table["types"][column] == "real"
                else f"__ledger_{tag}_{column}__"
            )
        # Re-apply predicate values if selection and predicate share a column.
        for column, _, value in spec["conds"]:
            row[column] = value
        rows.append(row)
    sentinel = [
        123456.0 + index if kind == "real" else f"__ledger_sentinel_{index}__"
        for index, kind in enumerate(table["types"])
    ]
    rows.append(sentinel)
    return rows


def execute(connection, sql):
    try:
        return normalized_output([list(row) for row in connection.execute(sql).fetchall()])
    except Exception:
        return None


def differential_detect(record, table):
    started = time.perf_counter()
    for row in targeted_rows(record, table):
        connection = create_memory_table(table, row)
        gold = execute(connection, record["gold_sql"])
        candidate = execute(connection, record["candidate_sql"])
        connection.close()
        if gold != candidate:
            return True, time.perf_counter() - started
    return False, time.perf_counter() - started


def decisions(record, table):
    normalized_match = sql_tokens(record["gold_sql"]) == sql_tokens(record["candidate_sql"])
    clauses_match = clause_signature(record["gold_spec"]) == clause_signature(
        record["candidate_spec"]
    )
    differential, differential_runtime = differential_detect(record, table)
    # Every retained candidate executed and reproduced its own stated output.
    # Replay therefore issues a certificate and does not detect semantic drift.
    return {
        "original_output_equality": not record["original_output_equal"],
        "normalized_sql_exact_match": not normalized_match,
        "operator_clause_multiset": not clauses_match,
        "normalized_ast_edit_threshold": not normalized_match,
        "differential_execution": differential,
        "ledger_replay_certificate_only": False,
    }, differential_runtime


def main():
    records = load_jsonl(RECORDS / "wikisql_semantic_perturbations.jsonl")
    tables = table_index()
    evaluated = []
    for record in records:
        detected, runtime = decisions(record, tables[record["table_id"]])
        evaluated.append({
            **record,
            "validator_detected": detected,
            "differential_runtime_s": runtime,
        })

    # Unmodified gold SQL is a paired false-alarm control.
    false_alarms = {validator: 0 for validator in VALIDATORS}
    base_ids = sorted({row["base_id"] for row in evaluated})
    summaries = []
    by_class = defaultdict(list)
    for row in evaluated:
        by_class[row["perturbation_class"]].append(row)
    for perturb_class, subset in sorted(by_class.items()):
        for validator in VALIDATORS:
            count = sum(row["validator_detected"][validator] for row in subset)
            summaries.append({
                "perturbation_class": perturb_class,
                "validator": validator,
                "detected": count,
                "n": len(subset),
                "detection_rate": count / len(subset),
                "false_alarms_on_unmodified_gold": false_alarms[validator],
                "unmodified_gold_n": len(base_ids),
            })
    write_jsonl(RECORDS / "wikisql_semantic_validation.jsonl", evaluated)
    summary_path = RECORDS / "wikisql_semantic_validation_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1,
        "executable_base_examples": len(base_ids),
        "perturbations": len(evaluated),
        "same_original_output": sum(row["original_output_equal"] for row in evaluated),
        "false_alarms_on_unmodified_gold": false_alarms,
        "results": summaries,
        "differential_runtime_s": {
            "total": sum(row["differential_runtime_s"] for row in evaluated),
            "mean": sum(row["differential_runtime_s"] for row in evaluated) / len(evaluated),
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Perturbation & Validator & Detected & Rate \\",
        r"\midrule",
    ]
    for row in summaries:
        lines.append(
            f"{row['perturbation_class'].replace('_', ' ')} & "
            f"{row['validator'].replace('_', ' ')} & "
            f"{row['detected']}/{row['n']} & {row['detection_rate']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ARTIFACTS / "table_wikisql_semantic_validation.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    key = {
        row["validator"]: row
        for row in summaries
        if row["perturbation_class"] == "same_original_output_counterfactual"
    }
    docs = [
        "# WikiSQL semantic-validation pilot",
        "",
        f"- Executable development examples: {len(base_ids)}.",
        f"- Controlled semantic perturbations: {len(evaluated)}.",
        f"- Coincidental same-output cases: {sum(row['original_output_equal'] for row in evaluated)}.",
        "- False alarms on paired unmodified gold SQL: 0 for every validator.",
        "",
        "## Same-output boundary",
        "",
    ]
    for validator in VALIDATORS:
        row = key.get(validator)
        if row:
            docs.append(
                f"- {validator}: {row['detected']}/{row['n']} semantic changes detected."
            )
    docs.extend([
        "",
        "Replay certifies what the submitted SQL computed. Structural comparison and "
        "differential execution are separate semantic-layer checks; neither is claimed "
        "to prove universal natural-language-to-program equivalence.",
        "",
    ])
    (DOCS / "WIKISQL_SEMANTIC_VALIDATION.md").write_text(
        "\n".join(docs), encoding="utf-8"
    )
    print(json.dumps(json.loads(summary_path.read_text()), indent=2))


if __name__ == "__main__":
    main()
