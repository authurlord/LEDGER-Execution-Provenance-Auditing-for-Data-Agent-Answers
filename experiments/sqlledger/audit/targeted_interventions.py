"""Targeted, typed interventions for data-dependence (A3) evidence.

Replaces the crude global ``x1.37`` + 10%-row-delete counterfactual (kept only as legacy baseline
B6). For a trace query over declared tables we apply up to ``max_interventions`` deterministic, typed,
one-cell/one-row interventions on the ACTUALLY-READ columns, re-execute, and look for a comparable
output change:

    numeric cell delta      -> add a scale-aware delta to a read numeric cell
    extremum mutation       -> mutate the actual MIN/MAX cell of a read numeric column
    qualifying-row deletion -> delete the first data row

Rules (guidance): a sensitivity witness requires a SUCCESSFUL comparable output that CHANGES; a crash
is ``intervention_error`` (never sensitivity); if no valid successful intervention exists, the result
is ``inconclusive`` (never unsupported).

Returns a dict: {"verdict": sensitive|invariant_under_test|inconclusive, "trials": [...]}.

Run self-test: python3 -m experiments.sqlledger.audit.targeted_interventions
"""
from __future__ import annotations
import sqlite3

import sqlglot
from sqlglot import exp

from ..runtime.sqlite_reads import capture_reads
from ..runtime.canonicalize import equal


def _eq_constraints(sql):
    """Extract column = literal equality constraints from the WHERE clause (best-effort, sqlite)."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for eq in tree.find_all(exp.EQ):
        col = eq.left if isinstance(eq.left, exp.Column) else (eq.right if isinstance(eq.right, exp.Column) else None)
        lit = eq.right if isinstance(eq.right, exp.Literal) else (eq.left if isinstance(eq.left, exp.Literal) else None)
        if col is not None and lit is not None:
            out[col.name] = lit.this if hasattr(lit, "this") else str(lit)
    return out


def _fresh_copy(src, tables):
    from ..runtime.restricted_replay import materialize
    dst = materialize(src, list(tables))        # cached serialize/deserialize (fast for big tables)
    real_cols = {t: [c[1] for c in dst.execute(f'PRAGMA table_xinfo("{t}")').fetchall() if c[6] == 0]
                 for t in tables}
    return dst, real_cols


def _numeric_cols(con, table, cols):
    out = []
    for c in cols:
        vals = con.execute(f'SELECT "{c}" FROM "{table}" WHERE "{c}" IS NOT NULL LIMIT 50').fetchall()
        num = 0
        for (v,) in vals:
            try:
                float(v); num += 1
            except (TypeError, ValueError):
                pass
        if vals and num == len(vals):
            out.append(c)
    return out


def _base_rows(src, sql, f_claim):
    dst, _ = _fresh_copy(src, f_claim)
    try:
        rows, _cap, err = capture_reads(dst, sql)
        return rows, err
    finally:
        dst.close()


def _apply_and_run(src, sql, f_claim, mutate):
    """mutate(dst, real_cols) applies a deterministic change; returns (rows, err)."""
    dst, real_cols = _fresh_copy(src, f_claim)
    try:
        try:
            mutate(dst, real_cols)
            dst.commit()
        except sqlite3.Error as e:
            return None, f"mutate_error: {e}"
        rows, _cap, err = capture_reads(dst, sql)
        return rows, err
    finally:
        dst.close()


SENTINEL = "ZZ_SENTINEL_ZZ"


def assess(src, sql, f_claim, read_tables, read_columns, max_interventions=8):
    base_rows, berr = _base_rows(src, sql, f_claim)
    if berr is not None or base_rows is None:
        return {"verdict": "inconclusive", "reason": "base_failed", "error": berr, "trials": []}
    if len(base_rows) == 0:
        # an empty result cannot have its data-dependence established by cell/row mutation; do not
        # risk a false a3 (guidance: inconclusive, never unsupported)
        return {"verdict": "inconclusive", "reason": "empty_base_result", "trials": []}

    tables = [t for t in read_tables if t in f_claim] or list(f_claim)
    trials = []
    n_valid = 0
    eq = _eq_constraints(sql)

    def run(kind, mutate):
        nonlocal n_valid
        rows, err = _apply_and_run(src, sql, f_claim, mutate)
        if err is not None:
            trials.append({"kind": kind, "status": "intervention_error", "error": err}); return None
        n_valid += 1
        changed = not equal(rows, base_rows)
        trials.append({"kind": kind, "status": "ok", "changed": changed}); return changed

    for t in tables:
        # 0) qualifying-row insertion: build a row satisfying the WHERE equalities so that a
        #    count/existence/filter result that is currently empty-or-zero can change (guidance:
        #    count/existence intervention). Distinguishes a genuine 0-count from a vacuous constant.
        real_cols_t = [c[1] for c in src.execute(f'PRAGMA table_xinfo("{t}")').fetchall() if c[6] == 0]
        if real_cols_t:
            def mut_ins(dst, rc, _t=t, _cols=real_cols_t):
                vals = [eq.get(c, SENTINEL) for c in _cols]
                ph = ",".join(["?"] * len(_cols))
                dst.execute(f'INSERT INTO "{_t}"({",".join(chr(34)+c+chr(34) for c in _cols)}) VALUES ({ph})', vals)
                dst.execute(f'INSERT INTO "{_t}"({",".join(chr(34)+c+chr(34) for c in _cols)}) VALUES ({ph})', vals)
            if run(f"qualifying_insert:{t}", mut_ins):
                return {"verdict": "sensitive", "trials": trials}

        if len([x for x in trials if x["status"] == "ok"]) >= max_interventions:
            break
        # target the columns ACTUALLY READ (incl. text); fall back to all columns for COUNT(*)-style
        read_cols = [c for (tt, c) in read_columns if tt == t and c]
        cols = read_cols or [c[1] for c in src.execute(f'PRAGMA table_xinfo("{t}")').fetchall() if c[6] == 0]
        numset = set(_numeric_cols(src, t, cols))

        # 1) per-read-column typed replacement across all rows (does the output depend on this column?)
        for col in cols:
            if numset and col in numset:
                def mut(dst, rc, _t=t, _col=col):
                    dst.execute(f'UPDATE "{_t}" SET "{_col}" = CAST("{_col}" AS REAL) + 1000.0')
            else:
                def mut(dst, rc, _t=t, _col=col):
                    dst.execute(f'UPDATE "{_t}" SET "{_col}" = ?', (SENTINEL,))
            if run(f"replace:{t}.{col}", mut):
                return {"verdict": "sensitive", "trials": trials}

        # 2) extremum mutation on a numeric read column
        for col in (numset & set(cols)):
            def mut_ext(dst, rc, _t=t, _col=col):
                dst.execute(f'UPDATE "{_t}" SET "{_col}" = CAST("{_col}" AS REAL) + 1e9 '
                            f'WHERE CAST("{_col}" AS REAL) = (SELECT MAX(CAST("{_col}" AS REAL)) FROM "{_t}")')
            if run(f"extremum:{t}.{col}", mut_ext):
                return {"verdict": "sensitive", "trials": trials}
            break

        # 3) row-count dependence: delete the first row
        def mut_del(dst, rc, _t=t):
            dst.execute(f'DELETE FROM "{_t}" WHERE rowid = (SELECT MIN(rowid) FROM "{_t}")')
        if run(f"row_delete:{t}", mut_del):
            return {"verdict": "sensitive", "trials": trials}

    if n_valid == 0:
        return {"verdict": "inconclusive", "reason": "no_valid_intervention", "trials": trials}
    return {"verdict": "invariant_under_test", "trials": trials}


def _selftest():
    src = sqlite3.connect(":memory:")
    src.execute('CREATE TABLE "t"(col0 TEXT, col1 TEXT)')      # col1 numeric-as-text
    src.executemany('INSERT INTO "t" VALUES(?,?)', [("a", "10"), ("b", "20"), ("c", "30")])
    src.commit()
    # sensitive: MAX depends on data
    r = assess(src, 'SELECT MAX(CAST(col1 AS REAL)) FROM "t"', ["t"], ["t"], [("t", "col1")])
    assert r["verdict"] == "sensitive", r
    # vacuous: constant output
    r = assess(src, 'SELECT (COUNT(*)*0)+7 FROM "t"', ["t"], ["t"], [("t", "col0")])
    assert r["verdict"] == "invariant_under_test", r
    print("targeted_interventions selftest OK")


if __name__ == "__main__":
    _selftest()
