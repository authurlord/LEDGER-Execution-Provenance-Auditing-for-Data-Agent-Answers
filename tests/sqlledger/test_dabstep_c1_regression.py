"""Regression test for the c1 tautology bug in dabstep_sql_trace.audit().

Bug (docs/sqlledger/IMPLEMENTATION_AUDIT.md §4): audit() ignored its declared-citation argument and
rebuilt the restricted catalog from the SQL itself, so a query could reference any table and never
trigger c1. The fix makes the restricted catalog = the INDEPENDENTLY-declared f_claim; a runtime
reference to a table outside f_claim is a catalog miss = c1.

These assertions FAIL on the buggy version (which returned 'supported' for empty/wrong declarations)
and PASS on the fixed version. Requires the DABStep context CSVs to be present.

Run: python3 -m pytest tests/sqlledger/test_dabstep_c1_regression.py -q
 or: python3 tests/sqlledger/test_dabstep_c1_regression.py
"""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kramabench_integration import dabstep_sql_trace as T

SQL = "SELECT count(*) AS answer FROM payments"


def _ctx_present():
    return (T.CTX / T.TABLES["payments"]).exists()


def test_empty_declaration_is_c1():
    """Declaring NO sources while the SQL reads `payments` must be c1, not supported."""
    if not _ctx_present():
        pytest.skip("DABStep context CSVs absent")
    au = T.audit(SQL, f_claim=[])
    assert au["label"] == "unsupported" and au["subtype"] == "c1", au


def test_wrong_declaration_is_c1():
    """Declaring only an unrelated table while the SQL reads `payments` must be c1."""
    if not _ctx_present():
        pytest.skip("DABStep context CSVs absent")
    au = T.audit(SQL, f_claim=["merchant_category_codes"])
    assert au["label"] == "unsupported" and au["subtype"] == "c1", au


def test_correct_declaration_not_c1():
    """Declaring the table the SQL actually reads must NOT be c1."""
    if not _ctx_present():
        pytest.skip("DABStep context CSVs absent")
    au = T.audit(SQL, f_claim=["payments"])
    assert au["subtype"] != "c1", au
    assert au["label"] in ("supported", "unknown", "unsupported"), au


def test_declared_arg_is_actually_used():
    """Static guard: audit()'s body must reference f_claim (not silently ignore it)."""
    import inspect
    body = inspect.getsource(T.audit).split("):", 1)[1]
    assert "f_claim" in body, "audit() ignores its declared-citation argument (tautology bug)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
