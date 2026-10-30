"""Construction generator for SQLLedger controlled cases (WikiSQL).

For each parent (a WikiSQL query that executes deterministically) it constructs up to seven cases,
each with a machine-checkable ``construction_witness``. Gold support labels are produced ENTIRELY by
program execution + declared-citation logic — no LEDGER checker, no LLM.

Classes:
  s1   supported + correct         : gold trace, declared = read tables, stated = trace output
  s2   supported + task-incorrect  : faithful compute over a cited schema-compatible DISTRACTOR table
  u_a1 stated/computed mismatch     : gold trace, stated = output of a non-equivalent executable mutant
  u_b  answer after exec failure    : deterministically broken trace, non-empty stated answer
  u_c1 undeclared dependency        : declared sources omit the read table (single-table -> empty)
  u_c2 unresolvable declaration     : declared source is an invalid identifier absent from the source
  u_a3 vacuous computation          : reads the cited table but value-invariant under intervention

Every case is verified by re-execution/restricted-replay before it is emitted; otherwise it is
rejected and counted (no silent labels).

Run self-test: python3 -m experiments.sqlledger.construct.generate
"""
from __future__ import annotations
import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from ..adapters.wikisql import WikiSQLDataset, WIKI, table_name
from ..runtime.canonicalize import canonicalize, equal
from ..runtime.restricted_replay import restricted_replay
from ..runtime.sqlite_reads import capture_reads
from ..schema import new_record, validate
from . import mutations as M
from . import distractors as D

SEED = 20260711
INVALID_SOURCE = "__unresolvable_source__"


def _canon_str(rows, ordered=False):
    return repr(canonicalize(rows, ordered=ordered))


def _stated(rows):
    return {"rows": rows, "canonical": _canon_str(rows)}


def _exists_src(con, name):
    name = str(name).strip().strip('"').strip("'").strip("`")
    return con.execute("SELECT 1 FROM sqlite_master WHERE name=? COLLATE NOCASE", (name,)).fetchone() is not None


def _ordered_query(sql):
    """True if the query's result order is semantically meaningful (top-level ORDER BY / LIMIT)."""
    try:
        import sqlglot
        from sqlglot import exp
        t = sqlglot.parse_one(sql, read="sqlite")
        return t.find(exp.Order) is not None or t.find(exp.Limit) is not None
    except Exception:  # noqa: BLE001
        return False


def _copy_all(src, tables):
    from ..runtime.restricted_replay import materialize
    return materialize(src, list(tables))       # cached serialize/deserialize


def _data_required(src, sql, omit, all_read, gold_rows, ordered):
    """Is `omit`'s DATA required for the answer? Emptying `omit` in a full-read copy must change the
    result (or make it error). Distinguishes a genuine join dependency from a dead syntactic
    reference (codex spider #1). Returns (required: bool, detail)."""
    dst = _copy_all(src, all_read)
    try:
        try:
            dst.execute(f'DELETE FROM "{omit}"')
            dst.commit()
        except sqlite3.Error as e:
            return True, {"probe": "delete_failed_dependency", "error": str(e)[:80]}
        rows, _cap, err = capture_reads(dst, sql)
        if err is not None:
            return True, {"probe": "errors_without_data", "error": err[:80]}
        changed = not equal(rows, gold_rows, ordered=ordered)
        return changed, {"probe": "empty_omit_table", "changed": changed}
    finally:
        dst.close()


def _base(parent, cls, support, engine, revision, eval_split):
    return dict(
        case_id=f"{parent.parent_id}::{cls}", parent_id=parent.parent_id, dataset=parent.dataset,
        source_split=parent.split, evaluation_split=eval_split, db_id=parent.db_id,
        question=parent.question, original_gold_sql=parent.sql,
        gold_support=support, gold_class=cls, generation_seed=SEED,
        dataset_revision=revision, engine=engine)


def _a3_is_vacuous(con, table, a3_sql):
    """Verify the a3 query reads the table but is invariant under a targeted row-deletion
    intervention (delete one qualifying row in a fresh copy). Returns (is_vacuous, witness)."""
    r0 = restricted_replay(con, a3_sql, f_claim=[table])
    if r0.status != "success_comparable" or table not in r0.read_tables:
        return False, {"reason": "did_not_read_table_or_failed", "status": r0.status}
    # intervention: fresh copy (cached), delete one row, re-run a3
    dst = _copy_all(con, [table])
    try:
        if con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] < 1:
            return False, {"reason": "empty_table"}
        dst.execute(f'DELETE FROM "{table}" WHERE rowid = (SELECT MIN(rowid) FROM "{table}")')
        dst.commit()
        rows_pert, _cap, err = capture_reads(dst, a3_sql)
        if err is not None:
            return False, {"reason": "intervention_error", "error": err}
        invariant = equal(r0.rows, rows_pert)
        return invariant, {"base": _canon_str(r0.rows), "perturbed": _canon_str(rows_pert),
                           "invariant": invariant, "intervention": "delete_one_row"}
    finally:
        dst.close()


def generate_for_parent(con, index, dataset, parent, engine, revision, eval_split, reasons):
    """Yield validated records for one parent. Rejections increment ``reasons``."""
    # NUL bytes cannot be embedded in a literal SQL string (codex #4); skip such parents.
    if "\x00" in parent.sql:
        reasons["nul_in_sql"] += 1; return
    # gold execution + determinism + runtime reads
    try:
        gold_rows, cap, err = capture_reads(con, parent.sql)
    except Exception as e:  # noqa: BLE001
        reasons["gold_exec_exception"] += 1; return
    if err is not None:
        reasons["gold_exec_timeout" if err.startswith("timeout:") else "gold_exec_error"] += 1; return
    if cap.catalog_tables:
        reasons["reads_catalog"] += 1; return
    f_read = cap.read_tables()
    if not f_read:
        reasons["no_read_tables"] += 1; return
    ordered = _ordered_query(parent.sql)
    # determinism: same-connection rerun AND a fresh-layout replay (catches ORDER BY/LIMIT tie
    # instability that a same-connection double-run misses — codex spider #3/#4).
    rows2, _c2, _e2 = capture_reads(con, parent.sql)
    if not equal(gold_rows, rows2, ordered=ordered) or canonicalize(gold_rows, ordered) != canonicalize(rows2, ordered):
        reasons["nondeterministic"] += 1; return
    r_layout = restricted_replay(con, parent.sql, f_claim=f_read, ordered=ordered, strategy="copy")
    if r_layout.status != "success_comparable" or not equal(r_layout.rows, gold_rows, ordered=ordered):
        reasons["layout_sensitive"] += 1; return
    primary = getattr(parent, "table", "") or f_read[0]     # a table name for u_b / a3
    gold_canon = _canon_str(gold_rows)
    supports_distractor = getattr(dataset, "supports_distractor", False) and index is not None

    # ---- s1 ---- (reuse the fresh-layout replay r_layout already validated above)
    rec = new_record(**_base(parent, "s1", "supported", engine, revision, eval_split),
                     trace_program=parent.sql, trace_status="ok",
                     claimed_sources=f_read, runtime_read_tables=f_read,
                     runtime_read_columns=cap.read_columns(),
                     stated_answer_canonical=gold_canon, trace_output_canonical=gold_canon,
                     task_correctness=True, mutation_type="none",
                     construction_witness={"replay_status": r_layout.status, "match": True, "ordered": ordered})
    validate(rec); yield rec

    # ---- s2 (supported but task-incorrect) -- WikiSQL single-table only ----
    d = D.find_distractor(con, index, dataset, parent, gold_rows) if supports_distractor else None
    if d:
        dtid, dsql, drows = d
        dtable = table_name(dtid)
        rr = restricted_replay(con, dsql, f_claim=[dtable])
        if rr.status == "success_comparable" and equal(rr.rows, drows) and not equal(drows, gold_rows):
            dcanon = _canon_str(drows)
            rec = new_record(**_base(parent, "s2", "supported", engine, revision, eval_split),
                             trace_program=dsql, trace_status="ok",
                             claimed_sources=[dtable], runtime_read_tables=rr.read_tables,
                             runtime_read_columns=rr.read_columns,
                             stated_answer_canonical=dcanon, trace_output_canonical=dcanon,
                             task_correctness=False, mutation_type="distractor_table",
                             construction_witness={"distractor": dtid, "distractor_result": dcanon,
                                                   "original_result": gold_canon, "differs": True})
            validate(rec); yield rec
        else:
            reasons["s2_reject"] += 1
    elif supports_distractor:
        reasons["s2_no_distractor"] += 1

    # ---- u_a1 (stated/computed mismatch) ----
    if parent.sql_struct:                       # WikiSQL: structured mutant
        mm = M.find_a1_mutant(con, parent.sql_struct, parent.db_id, len(parent.header), gold_rows)
        m = (mm[0], mm[2]) if mm else None      # (mutant_sql, mutant_rows)
    else:                                       # Spider: raw-SQL AST mutant
        m = M.find_sql_a1_mutant(con, parent.sql, gold_rows, ordered=ordered)
    if m:
        msql, mrows = m
        rec = new_record(**_base(parent, "u_a1", "unsupported", engine, revision, eval_split),
                         trace_program=parent.sql, trace_status="ok",
                         claimed_sources=f_read, runtime_read_tables=f_read,
                         runtime_read_columns=cap.read_columns(),
                         stated_answer_canonical=_canon_str(mrows), trace_output_canonical=gold_canon,
                         task_correctness=None, mutation_type="a1_nonequiv_mutant",
                         construction_witness={"trace_output": gold_canon,
                                               "stated_output": _canon_str(mrows), "differ": True,
                                               "mutant_sql": msql})
        validate(rec); yield rec
    else:
        reasons["u_a1_no_mutant"] += 1

    # ---- u_b (answer after execution failure) ----
    # precondition: a genuinely non-empty stated answer AFTER a deterministic failure (codex #2)
    if gold_rows:
        broken = f'SELECT __sqlledger_nope__ FROM "{primary}"'
        rb = restricted_replay(con, broken, f_claim=[primary])
        if rb.status == "ordinary_exec_error":
            rec = new_record(**_base(parent, "u_b", "unsupported", engine, revision, eval_split),
                             trace_program=broken, trace_status="exec_error",
                             claimed_sources=[primary], runtime_read_tables=[],
                             runtime_read_columns=[],
                             stated_answer_canonical=gold_canon, trace_output_canonical=None,
                             task_correctness=None, mutation_type="broken_column",
                             construction_witness={"replay_status": rb.status, "error": rb.error,
                                                   "stated_nonempty": bool(gold_rows),
                                                   "stated_rowcount": len(gold_rows)})
            validate(rec); yield rec
        else:
            reasons["u_b_not_error"] += 1
    else:
        reasons["u_b_empty_stated"] += 1

    # ---- u_c1 (undeclared dependency) ----
    # c1 vs c2 (codex #1): c1's declaration RESOLVES yet omits a runtime-read table. Multi-table
    # (Spider): declare all read tables but ONE (genuine partial-omission on a join). Single-table
    # (WikiSQL): declare a distinct REAL other table so the declaration resolves but is incomplete.
    c1_decl, c1_wit, omitted_ok = [], {}, False
    if len(f_read) >= 2:
        # try each read table as the omitted one; require BOTH a name-resolution halt AND genuine
        # DATA dependence (emptying it changes the answer) so a dead syntactic reference is not a
        # false c1 (codex spider #1); trying every candidate avoids order-dependent misses (#2).
        for omit in f_read:
            declared = [t for t in f_read if t != omit]
            if not all(_exists_src(con, t) for t in declared):
                continue
            rc1 = restricted_replay(con, parent.sql, f_claim=declared)
            if not (rc1.status == "missing_declared_dependency" and rc1.missing_table
                    and rc1.missing_table.lower() in {t.lower() for t in f_read}):
                continue
            req, det = _data_required(con, parent.sql, omit, f_read, gold_rows, ordered)
            if not req:
                continue
            c1_decl = declared
            c1_wit = {"declared_read_subset": declared, "declared_resolves": True,
                      "omitted_read_table": rc1.missing_table, "omitted_in_runtime_read": True,
                      "data_required": det, "runtime_read": f_read,
                      "replay_status": rc1.status, "mode": "partial_omission_join"}
            omitted_ok = True
            break
    else:
        row = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name<>? "
                          "AND name NOT LIKE 'sqlite_%' COLLATE NOCASE LIMIT 1", (primary,)).fetchone()
        other = row[0] if row else None
        rc1 = restricted_replay(con, parent.sql, f_claim=[other]) if other else None
        omitted_ok = (other and rc1 and rc1.status == "missing_declared_dependency"
                      and rc1.missing_table and rc1.missing_table.lower() in {t.lower() for t in f_read})
        c1_decl, c1_wit = ([other] if other else []), {"declared_other_table": other,
                          "declared_resolves": True, "omitted_read_table": (rc1.missing_table if rc1 else None),
                          "omitted_in_runtime_read": True, "runtime_read": f_read,
                          "replay_status": (rc1.status if rc1 else None), "mode": "declare_other_table"}
    if omitted_ok:
        rec = new_record(**_base(parent, "u_c1", "unsupported", engine, revision, eval_split),
                         trace_program=parent.sql, trace_status="ok",
                         claimed_sources=c1_decl, runtime_read_tables=f_read,
                         runtime_read_columns=cap.read_columns(),
                         stated_answer_canonical=gold_canon, trace_output_canonical=gold_canon,
                         task_correctness=None, mutation_type="omit_read_table",
                         construction_witness=c1_wit)
        validate(rec); yield rec
    else:
        reasons["u_c1_unexpected"] += 1

    # ---- u_c2 (unresolvable declaration) ----
    declared_resolves = _exists_src(con, INVALID_SOURCE)
    rc2 = restricted_replay(con, parent.sql, f_claim=[INVALID_SOURCE])
    if rc2.status == "missing_declared_dependency" and not declared_resolves and f_read:
        rec = new_record(**_base(parent, "u_c2", "unsupported", engine, revision, eval_split),
                         trace_program=parent.sql, trace_status="ok",
                         claimed_sources=[INVALID_SOURCE], runtime_read_tables=f_read,
                         runtime_read_columns=cap.read_columns(),
                         stated_answer_canonical=gold_canon, trace_output_canonical=gold_canon,
                         task_correctness=None, mutation_type="invalid_declared_identifier",
                         construction_witness={"invalid_identifier": INVALID_SOURCE,
                                               "declared_resolves": False,
                                               "runtime_read_nonempty": bool(f_read),
                                               "runtime_read": f_read, "replay_status": rc2.status})
        validate(rec); yield rec
    else:
        reasons["u_c2_unexpected"] += 1

    # ---- u_a3 (vacuous computation) ----
    a3sql = M.build_a3_sql(primary)
    vac, wit = _a3_is_vacuous(con, primary, a3sql)
    if vac:
        a3_rows, _c, _e = capture_reads(con, a3sql)
        rec = new_record(**_base(parent, "u_a3", "unsupported", engine, revision, eval_split),
                         trace_program=a3sql, trace_status="ok",
                         claimed_sources=[primary], runtime_read_tables=[primary],
                         runtime_read_columns=[],
                         stated_answer_canonical=_canon_str(a3_rows), trace_output_canonical=_canon_str(a3_rows),
                         task_correctness=None, mutation_type="vacuous_query",
                         construction_witness=wit)
        validate(rec); yield rec
    else:
        reasons["u_a3_not_vacuous"] += 1


def _make_dataset(dataset, split):
    if dataset == "wikisql":
        return WikiSQLDataset(split), "wikisql-salesforce-2017"
    if dataset == "spider":
        from ..adapters.spider import SpiderDataset
        return SpiderDataset(split), "spider-1.0-2020"
    if dataset == "bird":
        from ..adapters.bird import BIRDDataset
        return BIRDDataset(split), "bird-minidev-2024"
    raise ValueError(f"unknown dataset {dataset}")


def generate(split="dev", limit=50, eval_split="calibration", out_path=None, dataset="wikisql"):
    ds, revision = _make_dataset(dataset, split)
    index = D.build_index(ds) if getattr(ds, "supports_distractor", False) else None
    engine = f"sqlite-{sqlite3.sqlite_version}"
    reasons = Counter(); counts = Counter(); recs = []
    con_cache = {}

    def get_con(path):
        if path not in con_cache:
            c = sqlite3.connect(path); c.execute("PRAGMA query_only=ON")
            con_cache[path] = c
        return con_cache[path]

    for p in ds.iter_parents(limit=limit):
        con = get_con(p.db_path)
        for rec in generate_for_parent(con, index, ds, p, engine, revision, eval_split, reasons):
            recs.append(rec); counts[rec["gold_class"]] += 1
    for c in con_cache.values():
        c.close()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return recs, counts, reasons


def _selftest():
    if not (WIKI / "dev.db").exists():
        print("SKIP generate selftest: data absent"); return
    recs, counts, reasons = generate(split="dev", limit=20)
    print("class counts:", dict(counts))
    print("rejections:", dict(reasons))
    # every record validates and has a non-empty witness
    for r in recs:
        validate(r)
        assert r["construction_witness"], r["case_id"]
    # expect all 7 classes represented across 20 parents
    for cls in ("s1", "s2", "u_a1", "u_b", "u_c1", "u_c2", "u_a3"):
        assert counts[cls] > 0, f"class {cls} never generated"
    print(f"generate selftest OK ({len(recs)} records across {len(counts)} classes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wikisql", choices=["wikisql", "spider", "bird"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--eval-split", default="calibration")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or len(sys.argv) == 1:
        _selftest()
    else:
        recs, counts, reasons = generate(a.split, a.limit, a.eval_split, a.out, dataset=a.dataset)
        print(json.dumps({"n": len(recs), "classes": dict(counts), "rejections": dict(reasons)}, indent=1))
