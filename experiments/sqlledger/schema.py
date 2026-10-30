"""Immutable, versioned JSON Schema for SQLLedger construction records.

Every constructed case is one record. Gold fields (``gold_support``, ``gold_class``,
``construction_witness``) are produced by the deterministic generator and MUST NOT be modified by any
audit/baseline. Audit fields are filled later by baselines and never overwrite gold.

Version the format via SCHEMA_VERSION; migrations add fields rather than rewriting.

Run self-test: python3 -m experiments.sqlledger.schema
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0.0"

GOLD_SUPPORT = ("supported", "unsupported", "unknown")
GOLD_CLASS = ("s1", "s2", "u_a1", "u_b", "u_c1", "u_c2", "u_a3", "unknown")

RECORD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SQLLedger construction record",
    "type": "object",
    "additionalProperties": True,  # allow forward-compatible extra fields
    "required": [
        "schema_version", "case_id", "parent_id", "dataset", "source_split", "evaluation_split",
        "db_id", "question", "trace_program", "trace_status",
        "claimed_sources", "runtime_read_tables", "runtime_read_columns",
        "stated_answer_canonical", "trace_output_canonical",
        "gold_support", "gold_class", "construction_witness",
        "mutation_type", "generation_seed", "dataset_revision", "engine",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "case_id": {"type": "string"},
        "parent_id": {"type": "string"},
        "dataset": {"type": "string", "enum": ["wikisql", "spider", "bird"]},
        "source_split": {"type": "string"},
        "evaluation_split": {"type": "string", "enum": ["dev", "test", "calibration"]},
        "db_id": {"type": "string"},
        "question": {"type": "string"},
        "original_gold_sql": {"type": ["string", "null"]},
        "trace_program": {"type": "string"},
        "trace_status": {"type": "string", "enum": ["ok", "exec_error", "timeout"]},
        "claimed_sources": {"type": "array", "items": {"type": "string"}},
        "runtime_read_tables": {"type": "array", "items": {"type": "string"}},
        "runtime_read_columns": {"type": "array"},
        "stated_answer_canonical": {"type": ["string", "null"]},
        "trace_output_canonical": {"type": ["string", "null"]},
        "task_correctness": {"type": ["boolean", "null"]},
        "gold_support": {"type": "string", "enum": list(GOLD_SUPPORT)},
        "gold_class": {"type": "string", "enum": list(GOLD_CLASS)},
        "construction_witness": {"type": "object"},
        "mutation_type": {"type": "string"},
        "generation_seed": {"type": "integer"},
        "dataset_revision": {"type": "string"},
        "engine": {"type": "string"},
        # audit fields — filled by baselines, never by the generator
        "audit_label": {"type": ["string", "null"]},
        "audit_subtype": {"type": ["string", "null"]},
        "audit_evidence": {"type": ["object", "null"]},
        "audit_latency_ms": {"type": ["number", "null"]},
    },
}

_AUDIT_FIELDS = ("audit_label", "audit_subtype", "audit_evidence", "audit_latency_ms")
_GOLD_FIELDS = ("gold_support", "gold_class", "construction_witness", "task_correctness")


def new_record(**kw):
    """Create a record dict with schema_version stamped and audit fields defaulted to None."""
    rec = {"schema_version": SCHEMA_VERSION}
    for f in _AUDIT_FIELDS:
        rec[f] = None
    rec.update(kw)
    return rec


def validate(rec, _validator=[None]):
    """Validate one record against RECORD_SCHEMA. Raises jsonschema.ValidationError on failure."""
    if _validator[0] is None:
        from jsonschema import Draft202012Validator
        _validator[0] = Draft202012Validator(RECORD_SCHEMA)
    _validator[0].validate(rec)
    # gold classes must be consistent with support
    supp = rec["gold_support"]; cls = rec["gold_class"]
    if cls in ("s1", "s2") and supp != "supported":
        raise ValueError(f"class {cls} must be supported, got {supp} ({rec['case_id']})")
    if cls.startswith("u_") and supp != "unsupported":
        raise ValueError(f"class {cls} must be unsupported, got {supp} ({rec['case_id']})")
    if not rec["construction_witness"]:
        raise ValueError(f"empty construction_witness ({rec['case_id']})")
    return True


def apply_audit(rec, label, subtype=None, evidence=None, latency_ms=None):
    """Write audit fields onto a copy of ``rec`` WITHOUT touching gold fields."""
    out = dict(rec)
    for f in _GOLD_FIELDS:
        if f in rec:
            out[f] = rec[f]  # preserved verbatim
    out["audit_label"] = label
    out["audit_subtype"] = subtype
    out["audit_evidence"] = evidence
    out["audit_latency_ms"] = latency_ms
    return out


def _selftest():
    rec = new_record(
        case_id="wikisql/train/1::s1", parent_id="wikisql/train/1", dataset="wikisql",
        source_split="train", evaluation_split="calibration", db_id="1-x", question="q?",
        original_gold_sql="SELECT 1", trace_program="SELECT 1", trace_status="ok",
        claimed_sources=["t"], runtime_read_tables=["t"], runtime_read_columns=[["t", "c"]],
        stated_answer_canonical="('bag', ((('int', 1),), 1))",
        trace_output_canonical="('bag', ((('int', 1),), 1))",
        task_correctness=True, gold_support="supported", gold_class="s1",
        construction_witness={"replay": "success", "match": True},
        mutation_type="none", generation_seed=20260711, dataset_revision="abc", engine="sqlite-3.46.0")
    assert validate(rec)
    audited = apply_audit(rec, "supported", evidence={"b3": "match"})
    assert audited["gold_support"] == "supported" and audited["audit_label"] == "supported"
    assert rec["audit_label"] is None  # original untouched
    # inconsistent gold class rejected
    bad = dict(rec); bad["gold_class"] = "u_c1"
    try:
        validate(bad); assert False, "should have rejected"
    except ValueError:
        pass
    print("schema selftest OK")


if __name__ == "__main__":
    _selftest()
