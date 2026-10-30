"""LEDGER E3 - same-trace replay of the old O1 locatability gate.

This is deterministic and LLM-free. It replays the old correctness/containment
gate over the same KramaBench stated-answer episodes emitted by E12/N2 records.

Old O1 gate:
  - numeric answer: the numeric value must appear as a clean numeric token in
    cited/retrieved evidence text;
  - text/entity answer: boundary-complete containment in cited/retrieved evidence;
  - otherwise the gate blocks/refuses.

Important: this is locatability, not recomputability. LEDGER's provenance checker
can support derived answers that are not verbatim-locatable in source files.

Run after N2 records are stable:
  python3 -m kramabench_integration.replay_old_gate_same_trace
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import glob
import json
import math
import os
import re
from typing import Any

from .kb_trace_provenance import DATA, REPO, boundary_contains

RECORD_GLOB = os.path.join(REPO, "records", "records_ledger_n2_kb_*_*.jsonl")
OUT_JSONL = os.path.join(REPO, "records", "old_gate_replay_kb.jsonl")
OUT_DOC = os.path.join(REPO, "docs", "OLD_GATE_REPLAY.md")

_NUM_TOKEN = re.compile(r"(?<![0-9A-Za-z_.-])-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![0-9A-Za-z_.-])")


def _as_decimal(x: Any) -> Decimal | None:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return Decimal(str(x))
    s = str(x).strip()
    if not s:
        return None
    if s.endswith("%"):
        s = s[:-1].strip()
    s = s.replace(",", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _numeric_close(a: Decimal, b: Decimal) -> bool:
    tol = max(abs(a) * Decimal("1e-9"), Decimal("1e-9"))
    return abs(a - b) <= tol


def _score_positive(score: Any) -> bool:
    try:
        x = float(score)
        return math.isfinite(x) and x > 0.0
    except Exception:
        return False


def _norm_scalar(x: Any) -> str:
    return str(x).strip().casefold()


def _pred_equals_gold(pred: Any, gold: Any) -> bool:
    pn, gn = _as_decimal(pred), _as_decimal(gold)
    if pn is not None and gn is not None:
        return _numeric_close(pn, gn)
    return _norm_scalar(pred) == _norm_scalar(gold)


def _setting_from_path(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"records_ledger_n2_kb_[^_]+_(oracle|discovery)\.jsonl$", base)
    return m.group(1) if m else ""


def _domain_index(domain: str) -> dict[str, str]:
    droot = os.path.join(DATA, domain)
    idx = {}
    for dirpath, _dirs, files in os.walk(droot):
        for f in files:
            idx.setdefault(f, os.path.relpath(os.path.join(dirpath, f), droot))
    return idx


_IDX: dict[str, dict[str, str]] = {}


def _resolve_file(domain: str, path: str) -> str | None:
    if not path:
        return None
    droot = os.path.realpath(os.path.join(DATA, domain))
    s = str(path).strip()
    cands = []
    if os.path.isabs(s):
        cands.append(s)
    else:
        cands.append(os.path.join(droot, s))
        if not s.startswith("input/"):
            cands.append(os.path.join(droot, "input", s))
    if domain not in _IDX:
        _IDX[domain] = _domain_index(domain)
    base = os.path.basename(s)
    if base in _IDX[domain]:
        cands.append(os.path.join(droot, _IDX[domain][base]))
    for cand in cands:
        if os.path.exists(cand):
            real = os.path.realpath(cand)
            if real.startswith(droot):
                return real
    return None


def _evidence_files(record: dict) -> list[str]:
    vals = []
    for p in record.get("provenance") or []:
        if p.get("source_file"):
            vals.append(p["source_file"])
    vals.extend(record.get("sources_selected") or [])
    vals.extend(record.get("offered_files") or [])
    out, seen = [], set()
    for v in vals:
        s = str(v).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _read_evidence_text(path: str, max_bytes: int = 4_000_000) -> str:
    with open(path, "rb") as f:
        raw = f.read(max_bytes)
    return raw.decode("utf-8", errors="ignore")


def _locate_numeric(target: Decimal, text: str) -> bool:
    for m in _NUM_TOKEN.finditer(text):
        try:
            val = Decimal(m.group(0).replace(",", ""))
        except InvalidOperation:
            continue
        if _numeric_close(target, val):
            return True
    return False


def old_gate_locate(record: dict) -> tuple[bool, list[str], list[str]]:
    """Return (located, located_files, resolved_files)."""
    domain = record.get("domain") or ""
    pred = record.get("pred")
    pred_num = _as_decimal(pred)
    located, resolved = [], []
    for rel in _evidence_files(record):
        path = _resolve_file(domain, rel)
        if not path:
            continue
        resolved.append(os.path.relpath(path, os.path.join(DATA, domain)))
        try:
            text = _read_evidence_text(path)
        except Exception:
            continue
        ok = _locate_numeric(pred_num, text) if pred_num is not None else boundary_contains(text, str(pred))
        if ok:
            located.append(os.path.relpath(path, os.path.join(DATA, domain)))
    return bool(located), sorted(set(located)), sorted(set(resolved))


def _iter_records():
    for path in sorted(glob.glob(RECORD_GLOB)):
        setting = _setting_from_path(path)
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                pred = r.get("pred")
                if pred is None or str(pred).strip() == "":
                    continue
                yield path, setting, r


def _summarize(rows: list[dict]) -> dict:
    n = len(rows)
    blocked = [r for r in rows if r["old_gate_blocked"]]
    score_pos_blocked = [r for r in blocked if r["official_score_positive"]]
    pred_gold_blocked = [r for r in blocked if r["pred_equals_gold"]]
    ledger_unsupported = [r for r in rows if r["ledger_grounding_label"] == "unsupported"]
    overlap = [r for r in blocked if r["ledger_grounding_label"] == "unsupported"]
    by_domain = {}
    for domain in sorted(set(r["domain"] for r in rows)):
        sub = [r for r in rows if r["domain"] == domain]
        b = [r for r in sub if r["old_gate_blocked"]]
        by_domain[domain] = {
            "n": len(sub),
            "blocked": len(b),
            "score_positive_blocked": sum(1 for r in b if r["official_score_positive"]),
            "pred_equals_gold_blocked": sum(1 for r in b if r["pred_equals_gold"]),
            "ledger_unsupported": sum(1 for r in sub if r["ledger_grounding_label"] == "unsupported"),
            "overlap_with_ledger_unsupported": sum(1 for r in b if r["ledger_grounding_label"] == "unsupported"),
        }
    return {
        "n": n,
        "blocked": len(blocked),
        "blocked_rate": round(len(blocked) / n, 4) if n else None,
        "official_score_positive_blocked": len(score_pos_blocked),
        "pred_equals_gold_blocked": len(pred_gold_blocked),
        "ledger_unsupported": len(ledger_unsupported),
        "old_gate_blocked_and_ledger_unsupported": len(overlap),
        "old_gate_blocked_not_ledger_unsupported": len(blocked) - len(overlap),
        "ledger_unsupported_not_old_gate_blocked": len(ledger_unsupported) - len(overlap),
        "by_domain": by_domain,
        "grounding_label_distribution": dict(Counter(r["ledger_grounding_label"] for r in rows)),
        "G3_gate": {
            "criterion": "old-gate false rejection materially exceeds LEDGER unsupported flags",
            "ledger_flags_reference": "LEDGER flags only grounding_label == unsupported; expected historical count is 3",
            "pass": len(score_pos_blocked) > len(ledger_unsupported),
        },
    }


def _write_doc(summary: dict):
    lines = [
        "# LEDGER E3 - Same-trace old O1 gate replay",
        "",
        "This replay evaluates the old locatability gate on the same KramaBench stated-answer episodes.",
        "It checks whether the stated answer is verbatim-locatable in cited/retrieved evidence, not whether",
        "the answer is recomputable from cited evidence.",
        "",
        "## Headline",
        "",
        f"- Episodes: {summary['n']}",
        f"- Old gate blocked: {summary['blocked']} ({summary['blocked_rate']})",
        f"- Official-score-positive blocked: {summary['official_score_positive_blocked']}",
        f"- Pred==gold blocked: {summary['pred_equals_gold_blocked']}",
        f"- LEDGER unsupported flags: {summary['ledger_unsupported']}",
        f"- Old gate blocked and LEDGER unsupported overlap: {summary['old_gate_blocked_and_ledger_unsupported']}",
        f"- Old gate blocked but not LEDGER unsupported: {summary['old_gate_blocked_not_ledger_unsupported']}",
        "",
        "## By Domain",
        "",
        "| domain | n | blocked | score-positive blocked | pred==gold blocked | LEDGER unsupported | overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, row in summary["by_domain"].items():
        lines.append(
            f"| {domain} | {row['n']} | {row['blocked']} | {row['score_positive_blocked']} | "
            f"{row['pred_equals_gold_blocked']} | {row['ledger_unsupported']} | "
            f"{row['overlap_with_ledger_unsupported']} |"
        )
    lines += [
        "",
        "## G3",
        "",
        f"Criterion: {summary['G3_gate']['criterion']}.",
        f"G3: {'PASS' if summary['G3_gate']['pass'] else 'FAIL'}",
        "",
        "Records: `records/old_gate_replay_kb.jsonl`.",
    ]
    with open(OUT_DOC, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_DOC), exist_ok=True)
    rows = []
    with open(OUT_JSONL, "w") as out:
        for path, setting, r in _iter_records():
            located, located_files, resolved_files = old_gate_locate(r)
            row = {
                "taskid": r.get("taskid"),
                "domain": r.get("domain"),
                "setting": setting,
                "record_source": os.path.relpath(path, REPO),
                "pred": r.get("pred"),
                "gold": r.get("gold"),
                "score": r.get("score"),
                "old_gate_blocked": not located,
                "located_files": located_files,
                "resolved_evidence_files": resolved_files,
                "ledger_grounding_label": r.get("grounding_label"),
                "official_score_positive": _score_positive(r.get("score")),
                "pred_equals_gold": _pred_equals_gold(r.get("pred"), r.get("gold")),
            }
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    summary = _summarize(rows)
    _write_doc(summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
