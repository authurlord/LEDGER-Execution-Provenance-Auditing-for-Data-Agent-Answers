"""Schema-compatible distractor tables for U-S2 (supported-but-task-incorrect).

A distractor is a DIFFERENT table on which the same structured query (same column indices, agg, cond
ops) can execute. Since WikiSQL columns are ``col0..colN`` all TEXT, "schema-compatible" reduces to
"has enough columns for every index the query references". Selection is deterministic: the first
compatible table by sorted table id whose result differs from the original.

Run self-test: python3 -m experiments.sqlledger.construct.distractors
"""
from __future__ import annotations
import sqlite3
from ..adapters.wikisql import build_sql, table_name
from ..runtime.canonicalize import equal


def build_index(dataset):
    """Map n_columns -> sorted list of table_ids present in the dataset's tables.jsonl."""
    idx = {}
    for tid, meta in dataset.tables().items():
        nc = len(meta.get("header", []))
        idx.setdefault(nc, []).append(tid)
    for nc in idx:
        idx[nc].sort()
    return idx


def referenced_cols(sql_struct: dict):
    cols = {sql_struct["sel"]}
    for c, _op, _v in sql_struct.get("conds", []):
        cols.add(c)
    return cols


def find_distractor(con: sqlite3.Connection, index, dataset, parent, gold_rows, prefer_nonempty=True):
    """Return (distractor_table_id, distractor_sql, distractor_rows) or None.

    The distractor must (a) have every referenced column index, (b) be a different table, (c) execute,
    and (d) yield a result different from ``gold_rows``. When ``prefer_nonempty`` (default), a
    distractor with a NON-EMPTY differing result is preferred (codex #5: avoid a degenerate s2 class
    dominated by empty results); an empty differing result is used only as a fallback.
    """
    need = max(referenced_cols(parent.sql_struct)) + 1
    fallback = None
    for nc in sorted(k for k in index if k >= need):
        for tid in index[nc]:
            if tid == parent.db_id:
                continue
            dsql = build_sql(parent.sql_struct, tid)
            try:
                rows = con.execute(dsql).fetchall()
            except sqlite3.Error:
                continue
            if equal(rows, gold_rows):
                continue
            if rows or not prefer_nonempty:
                return tid, dsql, rows
            if fallback is None:
                fallback = (tid, dsql, rows)
    return fallback


def _selftest():
    from ..adapters.wikisql import WikiSQLDataset, WIKI
    if not (WIKI / "dev.db").exists():
        print("SKIP distractors selftest: data absent"); return
    ds = WikiSQLDataset("dev")
    con = ds.db()
    idx = build_index(ds)
    found = 0
    for p in ds.iter_parents(limit=30):
        try:
            gold = con.execute(p.sql).fetchall()
        except sqlite3.Error:
            continue
        d = find_distractor(con, idx, ds, p, gold)
        if d:
            found += 1
    con.close()
    assert found >= 20, f"only {found}/30 parents got a distractor"
    print(f"distractors selftest OK ({found}/30 parents matched)")


if __name__ == "__main__":
    _selftest()
