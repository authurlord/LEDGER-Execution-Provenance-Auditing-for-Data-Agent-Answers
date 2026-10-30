"""Support-determination baselines B0-B6 (ablation ladder) over SQLLedger records.

Each baseline maps a record -> {"label": supported|unsupported|unknown, "subtype": ..., "evidence": {}}
using ONLY the trace program, the declared sources, the stated answer, and re-execution against the
source database. None of them look at the gold label. Run all on identical eligible IDs.

    B0  citation-exists              declared sources non-empty
    B1  static SQL table extraction  every statically-referenced table is declared (SQLGlot)
    B2  unrestricted replay + match  run trace on full DB; output matches stated
    B3  restricted replay + match    run trace under declared-only catalog; success & matches stated
    B4  B3 + read/declaration consistency (F_read subset of F_claim)
    B5  B4 + targeted-intervention data-dependence (vacuous -> unsupported)   [== full LEDGER]
    B6  B4 + legacy global-scale/row-delete counterfactual                    [ablation]

Run self-test: python3 -m experiments.sqlledger.audit.baselines
"""
from __future__ import annotations
import sqlite3
import time

import sqlglot
from sqlglot import exp

from ..runtime.canonicalize import canonicalize
from ..runtime.restricted_replay import restricted_replay
from ..runtime.sqlite_reads import capture_reads
from . import targeted_interventions as TI

SUPPORTED, UNSUPPORTED, UNKNOWN = "supported", "unsupported", "unknown"


def _canon_repr(rows):
    return repr(canonicalize(rows))


def _static_tables(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return None
    names = set()
    for t in tree.find_all(exp.Table):
        names.add(t.name)
    return names


def _norm(names):
    return {str(n).strip().strip('"').strip("'").strip("`").lower() for n in names}


def b0(rec, src):
    ok = bool(rec.get("claimed_sources"))
    return {"label": SUPPORTED if ok else UNSUPPORTED, "subtype": None if ok else "c2",
            "evidence": {"declared": rec.get("claimed_sources")}}


def b1(rec, src):
    st = _static_tables(rec["trace_program"])
    if st is None:
        return {"label": UNKNOWN, "subtype": None, "evidence": {"parse": "failed"}}
    declared = _norm(rec.get("claimed_sources", []))
    ref = _norm(st)
    missing = ref - declared
    if not declared:
        return {"label": UNSUPPORTED, "subtype": "c2", "evidence": {"referenced": sorted(ref)}}
    if missing:
        return {"label": UNSUPPORTED, "subtype": "c1", "evidence": {"undeclared": sorted(missing)}}
    return {"label": SUPPORTED, "subtype": None, "evidence": {"referenced": sorted(ref)}}


def b2(rec, src):
    if rec.get("trace_status") != "ok":
        return {"label": UNSUPPORTED, "subtype": "b", "evidence": {"trace_status": rec.get("trace_status")}}
    rows, _cap, err = capture_reads(src, rec["trace_program"])   # full DB, unrestricted
    if err is not None:
        return {"label": UNSUPPORTED, "subtype": "b", "evidence": {"error": err}}
    match = _canon_repr(rows) == rec["stated_answer_canonical"]
    return {"label": SUPPORTED if match else UNSUPPORTED, "subtype": None if match else "a1",
            "evidence": {"match": match}}


def _restricted(rec, src):
    return restricted_replay(src, rec["trace_program"], f_claim=rec.get("claimed_sources", []))


def b3(rec, src):
    if rec.get("trace_status") != "ok":
        return {"label": UNSUPPORTED, "subtype": "b", "evidence": {"trace_status": rec.get("trace_status")}}
    r = _restricted(rec, src)
    if r.status == "missing_declared_dependency":
        sub = "c2" if (not rec.get("claimed_sources")) or TI is None else "c1"
        # distinguish: empty/invalid declaration -> c2, omitted real table -> c1
        decl = rec.get("claimed_sources", [])
        sub = "c2" if (not decl or all(not _exists(src, d) for d in decl)) else "c1"
        return {"label": UNSUPPORTED, "subtype": sub, "evidence": {"missing": r.missing_table}}
    if r.status == "reads_catalog":
        return {"label": UNKNOWN, "subtype": None, "evidence": {"reads_catalog": True}}
    if r.status != "success_comparable":
        return {"label": UNSUPPORTED, "subtype": "b", "evidence": {"status": r.status, "error": r.error}}
    match = repr(r.canonical) == rec["stated_answer_canonical"]
    return {"label": SUPPORTED if match else UNSUPPORTED, "subtype": None if match else "a1",
            "evidence": {"match": match}}


def _exists(src, name):
    name = str(name).strip().strip('"').strip("'").strip("`")
    return src.execute("SELECT 1 FROM sqlite_master WHERE name=? COLLATE NOCASE", (name,)).fetchone() is not None


def b4(rec, src):
    base = b3(rec, src)
    if base["label"] != SUPPORTED:
        return base
    f_read = _norm(rec.get("runtime_read_tables", []))
    f_claim = _norm(rec.get("claimed_sources", []))
    if not f_read <= f_claim:
        return {"label": UNSUPPORTED, "subtype": "c1",
                "evidence": {"undeclared_reads": sorted(f_read - f_claim)}}
    return base


def b5(rec, src):
    base = b4(rec, src)
    if base["label"] != SUPPORTED:
        return base
    ti = TI.assess(src, rec["trace_program"], rec.get("claimed_sources", []),
                   rec.get("runtime_read_tables", []), rec.get("runtime_read_columns", []))
    if ti["verdict"] == "invariant_under_test":
        return {"label": UNSUPPORTED, "subtype": "a3", "evidence": {"intervention": "invariant"}}
    return base  # sensitive or inconclusive -> keep supported


def b6(rec, src):
    base = b4(rec, src)
    if base["label"] != SUPPORTED:
        return base
    # legacy: global x1.37 + drop-10%-rows; invariant under both -> unsupported (a3)
    f_claim = rec.get("claimed_sources", [])
    r0 = restricted_replay(src, rec["trace_program"], f_claim=f_claim)
    if r0.status != "success_comparable":
        return base
    insens = 0
    for mode in ("scale", "drop"):
        rows, err = _legacy_perturb(src, rec["trace_program"], f_claim, mode)
        if err is None and rows is not None and repr(canonicalize(rows)) == repr(r0.canonical):
            insens += 1
    if insens == 2:
        return {"label": UNSUPPORTED, "subtype": "a3", "evidence": {"legacy_invariant": True}}
    return base


def _legacy_perturb(src, sql, f_claim, mode):
    dst = sqlite3.connect(":memory:")
    try:
        for t in f_claim:
            row = src.execute("SELECT sql FROM sqlite_master WHERE name=? AND type='table'", (t,)).fetchone()
            if not row:
                continue
            dst.execute(row[0])
            cols = [c[1] for c in src.execute(f'PRAGMA table_xinfo("{t}")').fetchall() if c[6] == 0]
            data = src.execute(f'SELECT {",".join(chr(34)+c+chr(34) for c in cols)} FROM "{t}"').fetchall()
            if data:
                ph = ",".join(["?"] * len(cols))
                dst.executemany(f'INSERT INTO "{t}" VALUES ({ph})', data)
            if mode == "scale":
                for c in cols:
                    dst.execute(f'UPDATE "{t}" SET "{c}" = CAST("{c}" AS REAL)*1.37 '
                                f'WHERE typeof(CAST("{c}" AS REAL))="real" AND "{c}" GLOB "*[0-9]*"')
            else:
                dst.execute(f'CREATE TEMP TABLE _r AS SELECT rowid rid, row_number() OVER () rn FROM "{t}"')
                dst.execute(f'DELETE FROM "{t}" WHERE rowid IN (SELECT rid FROM _r WHERE rn % 10 = 0)')
                dst.execute("DROP TABLE _r")
        dst.commit()
        rows, _cap, err = capture_reads(dst, sql)
        return rows, err
    except sqlite3.Error as e:
        return None, str(e)
    finally:
        dst.close()


BASELINES = {"B0": b0, "B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5, "B6": b6}


def run_baseline(name, rec, src):
    t0 = time.perf_counter()
    out = BASELINES[name](rec, src)
    out["latency_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


def _selftest():
    from ..adapters.wikisql import WIKI
    from ..construct.generate import generate
    if not (WIKI / "dev.db").exists():
        print("SKIP baselines selftest: data absent"); return
    recs, counts, _ = generate(split="dev", limit=15)
    src = sqlite3.connect(str(WIKI / "dev.db"))
    # B5 (full LEDGER) should recover gold_support well; B0 should be weak
    import collections
    agree = collections.Counter()
    for r in recs:
        for name in ("B0", "B5"):
            pred = run_baseline(name, r, src)["label"]
            gold = r["gold_support"]
            if gold in ("supported", "unsupported"):
                agree[(name, pred == gold)] += 1
    src.close()
    b5_acc = agree[("B5", True)] / max(1, agree[("B5", True)] + agree[("B5", False)])
    b0_acc = agree[("B0", True)] / max(1, agree[("B0", True)] + agree[("B0", False)])
    print(f"B5 agree={b5_acc:.3f}  B0 agree={b0_acc:.3f}  (n={len(recs)})")
    assert b5_acc >= 0.85, f"B5 should recover gold well, got {b5_acc}"
    assert b5_acc > b0_acc, "full LEDGER (B5) should beat citation-only (B0)"
    print("baselines selftest OK")


if __name__ == "__main__":
    _selftest()
