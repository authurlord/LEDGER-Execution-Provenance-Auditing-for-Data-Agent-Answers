"""Citation-restricted re-execution for SQLite traces.

Build a fresh database containing ONLY the independently-declared ``f_claim`` objects (tables/views,
copied faithfully from the source: real columns, rowids, indexes) and execute the trace query under
the read-recording, write-denying authorizer. A runtime reference to an object NOT in ``f_claim``
cannot resolve — the mechanical basis for detecting an undeclared dependency (candidate c1). We do
NOT register objects parsed out of the query (that was the old tautology bug).

Structured status (never a bare boolean):
    success_comparable          -> executed; canonical output available
    missing_declared_dependency -> a referenced object is absent from f_claim AND exists in the
                                    source (candidate c1)
    ordinary_exec_error         -> other SQL error, incl. references to objects that do not exist in
                                    the source either (invalid SQL / typo — NOT c1)  [codex #7]
    reads_catalog               -> query read a sqlite_* catalog table (out of data-source scope) [codex #2]
    timeout                     -> query wall-clock limit hit

Faithful copy handles rowid preservation (#3), generated/hidden columns (#4), indexes (#5), and
views (#6). Copy is bounded by ``max_copy_rows`` (#11).

Run self-test: python3 -m experiments.sqlledger.runtime.restricted_replay
"""
from __future__ import annotations
import re
import sqlite3
import threading
from dataclasses import dataclass, field

from .sqlite_reads import capture_reads, _CATALOG_TABLES
from .canonicalize import canonicalize, scalar_of

STATUS = ("success_comparable", "missing_declared_dependency", "ordinary_exec_error",
          "reads_catalog", "timeout")

_NO_SUCH = re.compile(r"no such (?:table|view|index)\s*:\s*(.+?)\s*$", re.IGNORECASE)

# Materialization cache: (source-db path, sorted declared tables) -> serialized bytes of a fresh DB
# holding only those tables. Copying big BIRD/Spider tables from disk once and deserializing the
# snapshot for every subsequent replay/intervention is far cheaper than re-copying each call.
_MAT_CACHE = {}
_MAT_ORDER = []
_MAT_MAX = 64


def _src_key(src):
    try:
        for _seq, name, fpath in src.execute("PRAGMA database_list").fetchall():
            if name == "main":
                return fpath or "mem"
    except sqlite3.Error:
        pass
    return "mem"


def materialize(src, tables, max_copy_rows=5_000_000):
    """Return a NEW in-memory connection holding only ``tables`` copied faithfully from ``src``.
    Cached by (source path, sorted tables) via serialize/deserialize so repeated calls for the same
    table set skip the disk copy. The caller owns (and closes) the returned connection."""
    key = (_src_key(src), tuple(sorted(tables)))
    blob = _MAT_CACHE.get(key)
    if blob is None:
        tmp = sqlite3.connect(":memory:")
        _copy_objects(src, tmp, list(tables), max_copy_rows)
        blob = tmp.serialize(); tmp.close()
        _MAT_CACHE[key] = blob; _MAT_ORDER.append(key)
        if len(_MAT_ORDER) > _MAT_MAX:
            old = _MAT_ORDER.pop(0); _MAT_CACHE.pop(old, None)
    dst = sqlite3.connect(":memory:")
    dst.deserialize(blob)
    return dst


@dataclass
class ReplayResult:
    status: str
    rows: list | None = None
    canonical: tuple | None = None
    scalar: object = None
    read_tables: list = field(default_factory=list)
    read_columns: list = field(default_factory=list)
    catalog_tables: list = field(default_factory=list)
    missing_table: str | None = None
    error: str | None = None


def _object_ddl(src, name):
    if str(name).lower().startswith("sqlite_"):
        return (None, None)          # internal catalog tables auto-exist; never copy them
    row = src.execute(
        "SELECT type, sql FROM sqlite_master WHERE name=? AND type IN ('table','view') COLLATE NOCASE",
        (name,)).fetchone()
    return (row[0], row[1]) if row and row[1] else (None, None)


def _exists_in_source(src, name):
    if not name:
        return False
    name = name.strip().strip('"').strip("'").strip("`").strip("[]")
    row = src.execute(
        "SELECT 1 FROM sqlite_master WHERE name=? COLLATE NOCASE AND type IN ('table','view')",
        (name,)).fetchone()
    return row is not None


def _has_rowid(src, name):
    # WITHOUT ROWID tables reject `SELECT rowid`; detect by probing.
    try:
        src.execute(f'SELECT rowid FROM "{name}" LIMIT 0'); return True
    except sqlite3.Error:
        return False


def _copy_table(src, dst, name, ddl, max_copy_rows, src_alias=None):
    dst.execute(ddl)
    xinfo = src.execute(f'PRAGMA table_xinfo("{name}")').fetchall()
    # xinfo row: cid,name,type,notnull,dflt,pk,hidden ; hidden: 0 normal,1 hidden,2 virtual,3 stored
    real_cols = [c[1] for c in xinfo if c[6] == 0]
    n = src.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
    if n > max_copy_rows:
        raise RuntimeError(f"table {name} has {n} rows > max_copy_rows={max_copy_rows}")
    collist = ",".join(f'"{c}"' for c in real_cols)
    keep_rowid = _has_rowid(src, name)
    sel_cols = (f'rowid,{collist}' if keep_rowid else collist)
    ins_cols = ('"rowid",' + collist) if keep_rowid else collist
    if src_alias:
        # bulk SQL copy across an ATTACHed source — no Python row round-trip (fast for big tables)
        dst.execute(f'INSERT INTO "{name}"({ins_cols}) SELECT {sel_cols} FROM {src_alias}."{name}"')
    else:
        data = src.execute(f'SELECT {sel_cols} FROM "{name}"').fetchall()
        if data:
            ph = ",".join(["?"] * (len(real_cols) + (1 if keep_rowid else 0)))
            dst.executemany(f'INSERT INTO "{name}"({ins_cols}) VALUES ({ph})', data)
    for iname, isql in src.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
            (name,)).fetchall():
        try:
            dst.execute(isql)
        except sqlite3.Error:
            pass


def _copy_objects(src, dst, names, max_copy_rows):
    """Copy declared tables (with rowid/indexes) then views. When the source is a file DB, ATTACH it
    and bulk-copy via INSERT...SELECT (avoids per-row Python round-trips on big tables). Unknown names
    are skipped (they then fail to resolve at replay = the missing-dependency signal)."""
    src_path = _src_key(src)
    src_alias = None
    if src_path not in ("mem", "", None):
        try:
            dst.execute(f"ATTACH DATABASE '{src_path}' AS _src")
            src_alias = "_src"
        except sqlite3.Error:
            src_alias = None
    try:
        views = []
        for name in names:
            typ, ddl = _object_ddl(src, name)
            if typ == "table":
                _copy_table(src, dst, name, ddl, max_copy_rows, src_alias=src_alias)
            elif typ == "view":
                views.append((name, ddl))
        for name, ddl in views:      # views last (may reference tables)
            try:
                dst.execute(ddl)
            except sqlite3.Error:
                pass
        dst.commit()
    finally:
        if src_alias:
            try:
                dst.execute("DETACH DATABASE _src")
            except sqlite3.Error:
                pass


def restricted_replay_authorizer(src, sql, f_claim, ordered=False):
    """Fast citation-restricted replay: run ``sql`` on the real (indexed) source with an authorizer
    that DENIES reads of tables outside ``f_claim`` — no row copy. An undeclared read (of a table
    that exists in source) is the c1 signal. Requires a source Connection. Cannot be used when the
    replay must mutate data (interventions copy separately)."""
    rows, cap, err = capture_reads(src, sql, allowed_tables=set(f_claim))
    common = dict(read_tables=cap.read_tables(), read_columns=cap.read_columns(),
                  catalog_tables=sorted(cap.catalog_tables))
    if cap.denied_tables:
        return ReplayResult(status="missing_declared_dependency", missing_table=cap.denied_tables[0],
                            error=err, **common)
    if err is not None:
        m = _NO_SUCH.search(err)
        missing = m.group(1).strip().strip('"').strip("'").strip("`") if m else None
        if missing and _exists_in_source(src, missing):
            return ReplayResult(status="missing_declared_dependency", missing_table=missing,
                                error=err, **common)
        return ReplayResult(status="ordinary_exec_error", error=err, **common)
    if cap.catalog_tables:
        return ReplayResult(status="reads_catalog", rows=rows,
                            canonical=canonicalize(rows, ordered=ordered), scalar=scalar_of(rows), **common)
    return ReplayResult(status="success_comparable", rows=rows,
                        canonical=canonicalize(rows, ordered=ordered), scalar=scalar_of(rows), **common)


def restricted_replay(src_db_path_or_con, sql, f_claim, ordered=False, timeout_s=10.0,
                      max_copy_rows=5_000_000, strategy="authorizer"):
    close_src = False
    if isinstance(src_db_path_or_con, sqlite3.Connection):
        src = src_db_path_or_con
    else:
        src = sqlite3.connect(src_db_path_or_con); close_src = True
    # fast path: no copy, deny undeclared reads on the real source (needs a live Connection)
    if strategy == "authorizer":
        try:
            return restricted_replay_authorizer(src, sql, f_claim, ordered=ordered)
        finally:
            if close_src:
                src.close()
    timer = None; dst = None
    try:
        try:
            dst = materialize(src, list(f_claim), max_copy_rows)   # cached deserialize
        except RuntimeError as e:  # oversized table
            return ReplayResult(status="ordinary_exec_error", error=str(e))
        timed_out = {"v": False}

        def _interrupt():
            timed_out["v"] = True; dst.interrupt()

        timer = threading.Timer(timeout_s, _interrupt); timer.start()
        rows, cap, err = capture_reads(dst, sql)
        timer.cancel(); timer.join()          # avoid post-execution race (codex #12)

        common = dict(read_tables=cap.read_tables(), read_columns=cap.read_columns(),
                      catalog_tables=sorted(cap.catalog_tables))
        if err is not None:
            if timed_out["v"]:
                return ReplayResult(status="timeout", error=err, **common)
            m = _NO_SUCH.search(err)
            missing = m.group(1).strip().strip('"').strip("'").strip("`") if m else None
            if missing and _exists_in_source(src, missing) and not _exists_in_source(dst, missing):
                return ReplayResult(status="missing_declared_dependency", missing_table=missing,
                                    error=err, **common)
            # object does not exist in source either -> invalid SQL, NOT c1 (codex #7)
            return ReplayResult(status="ordinary_exec_error", error=err, **common)
        if cap.catalog_tables:                # read a catalog table -> out of data-source scope (codex #2)
            return ReplayResult(status="reads_catalog", rows=rows,
                                canonical=canonicalize(rows, ordered=ordered), scalar=scalar_of(rows),
                                **common)
        return ReplayResult(status="success_comparable", rows=rows,
                            canonical=canonicalize(rows, ordered=ordered), scalar=scalar_of(rows),
                            **common)
    finally:
        if timer:
            timer.cancel()
        if dst is not None:
            dst.close()
        if close_src:
            src.close()


def _selftest():
    src = sqlite3.connect(":memory:")
    src.execute("CREATE TABLE a(x INTEGER, y TEXT)")
    src.executemany("INSERT INTO a(rowid,x,y) VALUES(?,?,?)", [(10, 1, "p"), (20, 2, "q"), (30, 3, "p")])
    src.execute("CREATE TABLE b(z INTEGER)")
    src.executemany("INSERT INTO b VALUES(?)", [(10,), (20,)])
    src.execute("CREATE VIEW v AS SELECT x FROM a")
    src.commit()

    r = restricted_replay(src, "SELECT COUNT(*) FROM a WHERE y='p'", f_claim=["a"])
    assert r.status == "success_comparable" and r.scalar == 2, (r.status, r.scalar)

    # rowid preserved (codex #3)
    r = restricted_replay(src, "SELECT rowid FROM a ORDER BY x", f_claim=["a"], ordered=True)
    assert r.rows == [(10,), (20,), (30,)], r.rows

    # undeclared table b that EXISTS in source -> missing dependency (candidate c1)
    r = restricted_replay(src, "SELECT COUNT(*) FROM a JOIN b ON a.x=b.z", f_claim=["a"])
    assert r.status == "missing_declared_dependency" and r.missing_table == "b", (r.status, r.missing_table)

    # empty declaration, query reads a -> missing dependency
    r = restricted_replay(src, "SELECT COUNT(*) FROM a", f_claim=[])
    assert r.status == "missing_declared_dependency", (r.status, r.error)

    # table that does NOT exist in source -> ordinary error, NOT c1 (codex #7)
    r = restricted_replay(src, "SELECT * FROM typo_table", f_claim=[])
    assert r.status == "ordinary_exec_error", (r.status, r.error)

    # view declared -> resolves (codex #6). view needs underlying table a too.
    r = restricted_replay(src, "SELECT COUNT(*) FROM v", f_claim=["a", "v"])
    assert r.status == "success_comparable" and r.scalar == 3, (r.status, r.scalar)

    # catalog-table read flagged, not silent success (codex #2)
    r = restricted_replay(src, "SELECT COUNT(*) FROM sqlite_master", f_claim=[])
    assert r.status == "reads_catalog", (r.status,)

    # bad column -> ordinary error
    r = restricted_replay(src, "SELECT nope FROM a", f_claim=["a"])
    assert r.status == "ordinary_exec_error", (r.status, r.error)

    # copy strategy (used by the tie-detection layout gate): same verdicts on the same cases
    assert restricted_replay(src, "SELECT COUNT(*) FROM a JOIN b ON a.x=b.z", f_claim=["a"],
                             strategy="copy").status == "missing_declared_dependency"
    assert restricted_replay(src, "SELECT rowid FROM a ORDER BY x", f_claim=["a"], ordered=True,
                             strategy="copy").rows == [(10,), (20,), (30,)]

    print("restricted_replay selftest OK")


if __name__ == "__main__":
    _selftest()
