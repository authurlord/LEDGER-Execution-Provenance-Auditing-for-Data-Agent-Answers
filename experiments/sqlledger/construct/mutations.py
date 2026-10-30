"""Deterministic SQL mutation operators for U-a1 (non-equivalent executable mutant) and the U-a3
vacuous-query builder. No randomness: mutants are tried in a fixed order and the first one that both
executes and produces a canonically-different result is returned.

Run self-test: python3 -m experiments.sqlledger.construct.mutations
"""
from __future__ import annotations
import sqlite3
from ..adapters.wikisql import build_sql, AGG_OPS, COND_OPS
from ..runtime.canonicalize import equal


def _variants(sql_struct: dict, n_cols: int):
    """Yield candidate mutated sql_structs in a fixed deterministic order."""
    base = sql_struct
    sel, agg, conds = base["sel"], base["agg"], list(base.get("conds", []))
    # 1) aggregate replacement (cycle to a different agg, incl. none<->COUNT)
    for new_agg in (3, 4, 5, 1, 2, 0):        # COUNT,SUM,AVG,MAX,MIN,none
        if new_agg != agg:
            yield {"sel": sel, "agg": new_agg, "conds": conds}
    # 2) selected-column change
    for new_sel in range(n_cols):
        if new_sel != sel:
            yield {"sel": new_sel, "agg": agg, "conds": conds}
    # 3) condition operator change
    for ci, (col, op, val) in enumerate(conds):
        for new_op in (0, 1, 2):
            if new_op != op:
                nc = list(conds); nc[ci] = [col, new_op, val]
                yield {"sel": sel, "agg": agg, "conds": nc}
    # 4) drop a condition
    for ci in range(len(conds)):
        nc = conds[:ci] + conds[ci + 1:]
        yield {"sel": sel, "agg": agg, "conds": nc}


def find_a1_mutant(con: sqlite3.Connection, sql_struct: dict, table_id: str, n_cols: int, gold_rows):
    """Return (mutant_sql, mutant_struct, mutant_rows) for the first mutant whose result differs from
    ``gold_rows`` under canonical comparison, or None if none found."""
    for mut in _variants(sql_struct, n_cols):
        msql = build_sql(mut, table_id)
        try:
            rows = con.execute(msql).fetchall()
        except sqlite3.Error:
            continue
        if not equal(rows, gold_rows):
            return msql, mut, rows
    return None


def find_sql_a1_mutant(con, sql, gold_rows, ordered=False):
    """Dataset-agnostic a1 mutant via sqlglot AST edits (for raw-SQL traces, e.g. Spider). Applies
    deterministic transformations and returns (mutant_sql, mutant_rows) for the first that executes
    and differs from ``gold_rows``, else None."""
    import sqlglot
    from sqlglot import exp
    try:
        base = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return None
    swaps = {"COUNT": "SUM", "SUM": "COUNT", "MAX": "MIN", "MIN": "MAX", "AVG": "SUM"}
    candidates = []

    # 1) swap the first aggregate function that has a real column argument (not COUNT(*))
    t = base.copy()
    for a in t.find_all(exp.AggFunc):
        name = a.key.upper(); new = swaps.get(name)
        arg = a.this
        if new and arg is not None and not isinstance(arg, exp.Star):
            try:
                a.replace(exp.func(new, arg.copy()))
                candidates.append(t)
            except Exception:  # noqa: BLE001
                pass
            break
    # 2) flip the first comparison operator
    for OpA, OpB in ((exp.GT, exp.LT), (exp.GTE, exp.LTE), (exp.EQ, exp.NEQ)):
        t = base.copy()
        found = list(t.find_all(OpA))
        if found:
            f = found[0]; f.replace(OpB(this=f.this, expression=f.expression))
            candidates.append(t); break
    # 3) drop the WHERE clause
    t = base.copy()
    w = t.find(exp.Where)
    if w:
        w.pop(); candidates.append(t)
    # 4) drop LIMIT
    t = base.copy()
    lim = t.find(exp.Limit)
    if lim:
        lim.pop(); candidates.append(t)

    from ..runtime.canonicalize import equal as _eq
    from ..runtime.sqlite_reads import capture_reads
    for cand in candidates:
        msql = cand.sql(dialect="sqlite")
        if msql.strip().lower() == sql.strip().lower():
            continue
        rows, _cap, err = capture_reads(con, msql)   # timed: a mutant (e.g. dropped WHERE) may blow up
        if err is not None:
            continue
        if not _eq(rows, gold_rows, ordered=ordered):
            return msql, rows
    return None


def build_a3_sql(table: str) -> str:
    """A data-reading but value-invariant query over ``table``: reads the table (COUNT(*)) yet the
    output is constant regardless of the data."""
    return f'SELECT (COUNT(*) * 0) + 7 AS answer FROM "{table}"'


def _selftest():
    con = sqlite3.connect(":memory:")
    con.execute('CREATE TABLE "table_t"(col0 TEXT, col1 TEXT, col2 TEXT)')
    con.executemany('INSERT INTO "table_t" VALUES(?,?,?)',
                    [("a", "1", "x"), ("b", "2", "y"), ("a", "3", "z")])
    gold_struct = {"sel": 0, "agg": 0, "conds": [[2, 0, "x"]]}       # SELECT col0 WHERE col2='x'
    gold = con.execute(build_sql(gold_struct, "t")).fetchall()
    assert gold == [("a",)], gold
    m = find_a1_mutant(con, gold_struct, "t", 3, gold)
    assert m is not None, "should find a differing mutant"
    msql, mut, mrows = m
    assert not equal(mrows, gold), (mrows, gold)
    # a3 invariance
    r1 = con.execute(build_a3_sql("table_t")).fetchall()
    con.execute('INSERT INTO "table_t" VALUES("c","9","w")')
    r2 = con.execute(build_a3_sql("table_t")).fetchall()
    assert r1 == r2 == [(7,)], (r1, r2)
    print("mutations selftest OK")


if __name__ == "__main__":
    _selftest()
