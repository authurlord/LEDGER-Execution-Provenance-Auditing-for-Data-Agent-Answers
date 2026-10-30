"""Authorize-time table/column reference capture for SQLite traces.

Uses ``sqlite3.Connection.set_authorizer`` + ``SQLITE_READ`` to record the (table, column) references
SQLite resolves when preparing+running a query. This is SQLite's OWN name resolution — more faithful
than a third-party parser for aliases/views/CTEs — and is used as an independent reference set for
attribution and as the basis for the c1 (undeclared-dependency) determination via restricted replay.

IMPORTANT semantics (codex review #1): the authorizer fires at STATEMENT-PREPARE / AUTHORIZE time,
not at row-execution time. It therefore reports references in branches that may not execute at
runtime (e.g. ``CASE WHEN 0 THEN (SELECT x FROM t) ELSE 7 END``, ``... WHERE 0``) and for ``EXPLAIN``.
It is thus a SOUND OVER-APPROXIMATION of the executed read set, NOT "reads as executed". For c1 this
is the correct set (a table referenced anywhere must be resolvable to prepare the statement); for
attribution-fidelity metrics we report it as the authorize-time reference set and note the
over-approximation.

Safety: writes, ATTACH/DETACH, all PRAGMAs, (v)table/temp/trigger/view creation/drop, and a denylist
of dangerous functions (load_extension, readfile/writefile, ...) are denied. Function handling uses a
DENYLIST (codex #10) so ordinary read-only built-ins (date, strftime, json_*, window fns, ...) run.

Catalog tables (``sqlite_master``/``sqlite_schema``/``sqlite_temp_master``/``sqlite_sequence``) are
recorded separately (codex #2) because they auto-exist in every database and must not silently
satisfy a declared-source dependency.

Run self-test: python3 -m experiments.sqlledger.runtime.sqlite_reads
"""
from __future__ import annotations
import sqlite3
import threading
from dataclasses import dataclass, field

DEFAULT_QUERY_TIMEOUT_S = 10.0   # abort any single query that runs longer (pathological BIRD joins)

# Actions denied in a read-only replay. We name every write/DDL/attach action explicitly (codex #14)
# rather than relying on incidental sqlite_master-write callbacks.
_DENY_ACTIONS = frozenset(a for a in (
    getattr(sqlite3, n, None) for n in (
        "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE",
        "SQLITE_ATTACH", "SQLITE_DETACH",
        "SQLITE_CREATE_TABLE", "SQLITE_DROP_TABLE",
        "SQLITE_CREATE_TEMP_TABLE", "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_CREATE_INDEX", "SQLITE_DROP_INDEX",
        "SQLITE_CREATE_TEMP_INDEX", "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_CREATE_TRIGGER", "SQLITE_DROP_TRIGGER",
        "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_CREATE_VIEW", "SQLITE_DROP_VIEW",
        "SQLITE_CREATE_TEMP_VIEW", "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_CREATE_VTABLE", "SQLITE_DROP_VTABLE",
        "SQLITE_ALTER_TABLE", "SQLITE_REINDEX",
    )) if a is not None)

# Dangerous functions denied by name; everything else is allowed (denylist, not allowlist).
_DENY_FUNCTIONS = frozenset({
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
    "zipfile", "sqlite_compileoption_used", "sqlite_compileoption_get",
})

_CATALOG_TABLES = frozenset({
    "sqlite_master", "sqlite_schema", "sqlite_temp_master", "sqlite_temp_schema", "sqlite_sequence",
})


@dataclass
class ReadCapture:
    tables: set = field(default_factory=set)          # data tables referenced
    columns: set = field(default_factory=set)         # (table, column) pairs; column '' for COUNT(*)
    order: list = field(default_factory=list)         # first-seen (table, column) order
    catalog_tables: set = field(default_factory=set)  # sqlite_* catalog tables referenced (codex #2)
    denied: list = field(default_factory=list)        # (action, arg1, arg2) that were denied
    denied_tables: list = field(default_factory=list)  # undeclared data tables denied (c1 signal)

    def note(self, table, column):
        if table in _CATALOG_TABLES:
            self.catalog_tables.add(table)
            return
        self.tables.add(table)
        pair = (table, column or "")
        if pair not in self.columns:
            self.columns.add(pair)
            self.order.append(pair)

    def read_tables(self):
        return sorted(self.tables)

    def read_columns(self):
        return sorted(self.columns)


def make_authorizer(cap: ReadCapture, deny_writes: bool = True, allowed_tables=None):
    """If ``allowed_tables`` is given, DENY reads of any data table not in it (records the denied
    table). This is the citation-restricted fast path: run against the real (indexed) source while
    forbidding undeclared reads — no row copy, and an undeclared read becomes a c1 signal."""
    allow = None if allowed_tables is None else {str(t).lower() for t in allowed_tables}

    def _auth(action, arg1, arg2, db_name, trigger):
        if action == sqlite3.SQLITE_READ:
            if allow is not None and arg1 not in _CATALOG_TABLES and str(arg1).lower() not in allow:
                if arg1 not in cap.denied_tables:
                    cap.denied_tables.append(arg1)
                return sqlite3.SQLITE_DENY
            cap.note(arg1, arg2)
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            fn = (arg2 or "").lower()
            if fn in _DENY_FUNCTIONS:
                cap.denied.append((action, arg1, arg2)); return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_PRAGMA:
            cap.denied.append((action, arg1, arg2)); return sqlite3.SQLITE_DENY
        if deny_writes and action in _DENY_ACTIONS:
            cap.denied.append((action, arg1, arg2)); return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return _auth


def capture_reads(con: sqlite3.Connection, sql: str, params=(), allowed_tables=None,
                  timeout_s=DEFAULT_QUERY_TIMEOUT_S):
    """Execute ``sql`` under a read-recording, write-denying authorizer.

    If ``allowed_tables`` is given, reads of undeclared data tables are DENIED and recorded in
    ``cap.denied_tables`` (citation-restricted fast path). A wall-clock ``timeout_s`` aborts a
    pathologically slow query via ``con.interrupt()`` (error prefixed ``timeout:``). Returns
    ``(rows, ReadCapture, error)``.
    """
    cap = ReadCapture()
    con.set_authorizer(make_authorizer(cap, deny_writes=True, allowed_tables=allowed_tables))
    timed_out = {"v": False}
    timer = None
    if timeout_s:
        def _intr():
            timed_out["v"] = True; con.interrupt()
        timer = threading.Timer(timeout_s, _intr); timer.start()
    try:
        cur = con.execute(sql, params)
        rows = cur.fetchall()
        return rows, cap, None
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        return None, cap, ("timeout: " + msg if timed_out["v"] else msg)
    finally:
        if timer:
            timer.cancel(); timer.join()
        con.set_authorizer(None)


def _selftest():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t(a INTEGER, b TEXT)")
    con.executemany("INSERT INTO t VALUES(?,?)", [(1, "x"), (2, "y"), (3, "z")])
    rows, cap, err = capture_reads(con, "SELECT b FROM t WHERE a > 1")
    assert err is None and rows == [("y",), ("z",)], (rows, err)
    assert cap.read_tables() == ["t"] and ("t", "a") in cap.columns and ("t", "b") in cap.columns
    # write denied
    _, cap, err = capture_reads(con, "DELETE FROM t")
    assert err is not None and cap.denied, (err, cap.denied)
    # date/strftime built-ins now allowed (codex #10)
    rows, cap, err = capture_reads(con, "SELECT strftime('%Y','2020-01-02'), abs(-3), upper(b) FROM t LIMIT 1")
    assert err is None, err
    # load_extension denied
    _, cap, err = capture_reads(con, "SELECT load_extension('x')")
    assert err is not None, "load_extension must be denied"
    # catalog table recorded separately, not as a data table (codex #2)
    rows, cap, err = capture_reads(con, "SELECT count(*) FROM sqlite_master")
    assert err is None and cap.read_tables() == [] and "sqlite_master" in cap.catalog_tables, (cap.read_tables(), cap.catalog_tables)
    # missing table errors
    _, _, err = capture_reads(con, "SELECT * FROM nope")
    assert err is not None
    print("sqlite_reads selftest OK")


if __name__ == "__main__":
    _selftest()
