#!/usr/bin/env python3
"""Construct independently labeled WikiSQL semantic perturbations."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from common import RECORDS, ROOT, write_jsonl


DATA = ROOT / "external" / "raw" / "wikisql" / "data"
AGG = ("", "MAX", "MIN", "COUNT", "SUM", "AVG")
COND = ("=", ">", "<")
AGG_SWAP = {1: 2, 2: 1, 3: 4, 4: 5, 5: 4}


def table_name(table_id):
    return "table_" + table_id.replace("-", "_")


def literal(value, column_type):
    if value is None:
        return "NULL"
    if column_type == "real":
        try:
            return repr(float(str(value).replace(",", "")))
        except ValueError:
            pass
    text = str(value).lower().replace("'", "''")
    return f"'{text}'"


def render_sql(spec, table):
    selected = f"col{spec['sel']}"
    expression = (
        f"{AGG[spec['agg']]}({selected})" if spec["agg"] else selected
    )
    clauses = []
    for column, operator, value in spec["conds"]:
        clauses.append(
            f"col{column} {COND[operator]} {literal(value, table['types'][column])}"
        )
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return f"SELECT {expression} FROM {table_name(table['id'])}{where}"


def execute(connection, sql):
    started = __import__("time").perf_counter()
    try:
        rows = connection.execute(sql).fetchall()
        return [list(row) for row in rows], None, __import__("time").perf_counter() - started
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:120]}", __import__("time").perf_counter() - started


def normalized_output(value):
    if value is None:
        return None
    normalized = []
    for row in value:
        new_row = []
        for item in row:
            if isinstance(item, float):
                new_row.append(round(item, 8))
            elif isinstance(item, str):
                new_row.append(item.strip().lower())
            else:
                new_row.append(item)
        normalized.append(new_row)
    return sorted(normalized, key=repr)


def variants(spec, table):
    ncols = len(table["header"])
    produced = []
    if spec["agg"] in AGG_SWAP:
        candidate = deepcopy(spec)
        candidate["agg"] = AGG_SWAP[spec["agg"]]
        produced.append(("wrong_aggregate", candidate))
    if ncols > 1:
        candidate = deepcopy(spec)
        candidate["sel"] = (spec["sel"] + 1) % ncols
        produced.append(("wrong_projection_column", candidate))
    if spec["conds"] and ncols > 1:
        candidate = deepcopy(spec)
        column, operator, value = candidate["conds"][0]
        candidate["conds"][0] = ((column + 1) % ncols, operator, value)
        produced.append(("wrong_predicate_column", candidate))

        alternatives = [
            row[column]
            for row in table["rows"]
            if str(row[column]).lower() != str(value).lower()
        ]
        if alternatives:
            candidate = deepcopy(spec)
            candidate["conds"][0] = (column, operator, alternatives[0])
            produced.append(("wrong_predicate_value", candidate))

        candidate = deepcopy(spec)
        candidate["conds"] = candidate["conds"][1:]
        produced.append(("dropped_predicate", candidate))
    used = {condition[0] for condition in spec["conds"]}
    unused = [column for column in range(ncols) if column not in used]
    if unused and table["rows"]:
        column = unused[0]
        candidate = deepcopy(spec)
        candidate["conds"].append((column, 0, table["rows"][0][column]))
        produced.append(("added_irrelevant_predicate", candidate))
    return produced


def main():
    examples = [
        json.loads(line)
        for line in (DATA / "dev.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tables = {
        table["id"]: table
        for table in (
            json.loads(line)
            for line in (DATA / "dev.tables.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    }
    connection = sqlite3.connect(DATA / "dev.db")
    records = []
    base_count = 0
    for example_index, example in enumerate(examples):
        if base_count >= 500:
            break
        table = tables[example["table_id"]]
        gold_spec = deepcopy(example["sql"])
        gold_spec["conds"] = [list(condition) for condition in gold_spec["conds"]]
        gold_sql = render_sql(gold_spec, table)
        gold_output, error, gold_runtime = execute(connection, gold_sql)
        if error is not None:
            continue
        built = []
        for perturb_class, candidate_spec in variants(gold_spec, table):
            if candidate_spec == gold_spec:
                continue
            candidate_sql = render_sql(candidate_spec, table)
            candidate_output, candidate_error, candidate_runtime = execute(
                connection, candidate_sql
            )
            if candidate_error is not None:
                continue
            built.append({
                "dataset": "wikisql_dev",
                "base_id": f"dev-{example_index}",
                "case_id": f"dev-{example_index}::{perturb_class}",
                "table_id": example["table_id"],
                "question": example["question"],
                "perturbation_class": perturb_class,
                "semantic_change_label": 1,
                "label_source": "controlled structured-SQL transformation",
                "gold_spec": gold_spec,
                "candidate_spec": candidate_spec,
                "gold_sql": gold_sql,
                "candidate_sql": candidate_sql,
                "gold_output": normalized_output(gold_output),
                "candidate_output": normalized_output(candidate_output),
                "original_output_equal": (
                    normalized_output(gold_output) == normalized_output(candidate_output)
                ),
                "gold_runtime_s": gold_runtime,
                "candidate_runtime_s": candidate_runtime,
            })
        if not built:
            continue
        records.extend(built)
        base_count += 1

    # Class 7 explicitly isolates coincidental original-output equality without
    # changing the independently constructed SQL pair.
    same_output = [record for record in records if record["original_output_equal"]]
    for index, source in enumerate(same_output[:100]):
        clone = deepcopy(source)
        clone["case_id"] = f"{source['base_id']}::same_output_counterfactual::{index}"
        clone["source_perturbation_class"] = source["perturbation_class"]
        clone["perturbation_class"] = "same_original_output_counterfactual"
        records.append(clone)
    write_jsonl(RECORDS / "wikisql_semantic_perturbations.jsonl", records)
    print(json.dumps({
        "executable_base_examples": base_count,
        "perturbations": len(records),
        "class_counts": dict(Counter(row["perturbation_class"] for row in records)),
        "same_original_output": sum(row["original_output_equal"] for row in records),
    }, indent=2))


if __name__ == "__main__":
    main()
