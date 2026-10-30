"""Deterministic canonicalization and tolerance-aware comparison of SQL result sets.

Two distinct needs are served by two functions (codex review #8/#9):

* ``canonicalize(rows)`` — an EXACT, type-preserving, hashable form used for (a) the rerun-determinism
  gate (a deterministic query returns identical output across runs, so exact hashing is correct) and
  (b) the stored ``*_canonical`` strings on records. NaN maps to a stable token so a repeated NaN is
  seen as deterministic.

* ``equal(a, b)`` — COMPARABLE equality used for support determination (stated vs replay). It is
  numeric-tolerant, cross-type numeric (int 1 == float 1.0), NFC for strings, and treats NaN as
  NON-comparable (returns False). It does structural pairwise matching (ordered) or greedy multiset
  matching (bag), so tolerance need not form a hash bucket.

Tolerance is type-based (relative for floats), NEVER inferred from either value.

Run self-test: python3 -m experiments.sqlledger.runtime.canonicalize
"""
from __future__ import annotations
import math
import unicodedata
from collections import Counter
from decimal import Decimal

DEFAULT_NUM_TOL = 1e-9   # relative tolerance for float comparison


# ---------- exact canonical form (determinism hashing / stored strings) ----------

def _canon_cell(v):
    if v is None:
        return ("null",)
    if isinstance(v, bool):
        return ("int", int(v))
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, Decimal):
        return ("dec", str(v))               # preserve full precision, do not lose to binary float
    if isinstance(v, float):
        if math.isnan(v):
            return ("nan",)
        if math.isinf(v):
            return ("inf", math.copysign(1, v))
        return ("num", repr(v))              # repr() round-trips the exact float
    if isinstance(v, (bytes, bytearray)):
        return ("bytes", bytes(v))
    return ("str", unicodedata.normalize("NFC", str(v)))


def canon_row(row):
    return tuple(_canon_cell(c) for c in row)


def canonicalize(rows, ordered=False):
    """Exact hashable canonical form of a result set (for determinism, not tolerance comparison)."""
    if rows is None:
        return ("error",)
    crows = [canon_row(r) for r in rows]
    if ordered:
        return ("ordered", tuple(crows))
    bag = Counter(crows)
    return ("bag", tuple(sorted(bag.items(), key=lambda kv: repr(kv[0]))))


def scalar_of(rows):
    """Extract the single scalar answer from a 1x1 result set, else None."""
    if not rows or len(rows) != 1 or len(rows[0]) != 1:
        return None
    return rows[0][0]


# ---------- tolerance-aware comparable equality (support determination) ----------

def _as_number(v):
    """Return v as float if it is a numeric cell, else None. bool is numeric; str is NOT coerced."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, int):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, float):
        return v
    return None


def cells_equal(x, y, num_tol=DEFAULT_NUM_TOL):
    """Comparable-equality of two scalar cells. NaN is non-comparable (False)."""
    if x is None and y is None:
        return True
    if x is None or y is None:
        return False
    nx, ny = _as_number(x), _as_number(y)
    if nx is not None and ny is not None:
        if math.isnan(nx) or math.isnan(ny):
            return False                      # NaN non-comparable
        if math.isinf(nx) or math.isinf(ny):
            return nx == ny
        return abs(nx - ny) <= num_tol * max(1.0, abs(nx), abs(ny))
    if (nx is None) != (ny is None):
        return False                          # number vs non-number
    # both non-numeric
    if isinstance(x, (bytes, bytearray)) or isinstance(y, (bytes, bytearray)):
        return bytes(x) == bytes(y) if type(x) == type(y) else False
    return unicodedata.normalize("NFC", str(x)) == unicodedata.normalize("NFC", str(y))


def _row_equal(r1, r2, num_tol):
    return len(r1) == len(r2) and all(cells_equal(a, b, num_tol) for a, b in zip(r1, r2))


def equal(rows_a, rows_b, ordered=False, num_tol=DEFAULT_NUM_TOL):
    """Comparable-equality of two result sets (tolerance-aware, cross-type numeric, NaN-noncomparable)."""
    if rows_a is None or rows_b is None:
        return False
    if len(rows_a) != len(rows_b):
        return False
    if ordered:
        return all(_row_equal(a, b, num_tol) for a, b in zip(rows_a, rows_b))
    # bag: greedy matching (result sets are small)
    remaining = list(rows_b)
    for a in rows_a:
        for i, b in enumerate(remaining):
            if _row_equal(a, b, num_tol):
                remaining.pop(i); break
        else:
            return False
    return not remaining


def _selftest():
    # bag equality ignores order, preserves duplicates
    assert equal([(1,), (2,), (2,)], [(2,), (1,), (2,)])
    assert not equal([(1,), (2,)], [(1,), (2,), (2,)])
    # ordered respects order
    assert not equal([(1,), (2,)], [(2,), (1,)], ordered=True)
    # cross-type numeric: int 1 == float 1.0  (codex #9)
    assert equal([(1,)], [(1.0,)])
    assert cells_equal(1, 1.0)
    # relative tolerance actually applied (codex #8)
    assert equal([(1.0,)], [(1.0 + 5e-10,)])           # within 1e-9 rel
    assert not equal([(1.0,)], [(1.0 + 5e-8,)])        # outside
    assert not equal([(1.0000000000001,)], [(1.5,)])
    # NULL equals NULL; NaN never equals NaN
    assert equal([(None,)], [(None,)])
    assert not equal([(float("nan"),)], [(float("nan"),)])
    # number vs string not equal
    assert not equal([(1,)], [("1",)])
    # unicode NFC; no lowercasing
    assert equal([("é",)], [("é",)])
    assert not equal([("ABC",)], [("abc",)])
    # exact canonical stable for determinism; Decimal precision preserved
    assert canonicalize([(1,), (2,)]) == canonicalize([(2,), (1,)])
    assert canonicalize([(Decimal("1.10"),)]) != canonicalize([(Decimal("1.1"),)])
    assert scalar_of([(42,)]) == 42 and scalar_of([(1,), (2,)]) is None
    print("canonicalize selftest OK")


if __name__ == "__main__":
    _selftest()
