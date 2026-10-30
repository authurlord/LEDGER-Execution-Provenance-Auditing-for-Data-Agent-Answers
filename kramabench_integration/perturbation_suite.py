"""LEDGER E9 - controlled perturbation suite for support-taxonomy detection.

Builds synthetic perturbations from live supported KramaBench traces, then runs
the refined deterministic support checker on each perturbation. Construction
labels come only from the perturbation operator, never from checker output.

Do not run while N2 records are being rewritten. Intended command:
  python3 -m kramabench_integration.perturbation_suite
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
import os
import re
from typing import Any

from . import kb_trace_provenance as P

SOURCE = os.path.join(P.REPO, "records", "kb_labels_rederived.jsonl")
OUT_JSONL = os.path.join(P.REPO, "records", "kb_perturbation_suite.jsonl")
OUT_DOC = os.path.join(P.REPO, "docs", "PERTURBATION_SUITE.md")
TARGET_PER_CLASS = 30
CLASSES = ("a1", "a2", "b", "c1", "c2", "d", "unknown")


def _setting_from_system(system: str) -> str:
    s = system or ""
    if "discovery" in s.lower():
        return "discovery"
    if "oracle" in s.lower():
        return "oracle"
    return ""


def _sut_from_system(system: str) -> str:
    return (system or "").split("|", 1)[0]


def _load_supported(path=SOURCE):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("grounding_label") != "supported":
                continue
            if r.get("pred") is None or str(r.get("pred")).strip() == "":
                continue
            system = r.get("system") or ""
            sut = _sut_from_system(system)
            setting = _setting_from_system(system)
            if not sut or not setting:
                continue
            rows.append({**r, "_sut": sut, "_setting": setting})
    return rows


def _capture_for(row):
    qdir = os.path.join(P.SCRATCH, row["_sut"], row["taskid"])
    trace = P.capture_trace(qdir, row["domain"])
    if trace is None:
        return None
    return trace


def _as_float(x: Any):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


def _perturb_answer(pred: Any):
    x = _as_float(pred)
    if x is not None:
        return round((x * 1.7) + 10000019.0, 6)
    return "__LEDGER_WRONG_ENTITY_TOKEN_DOES_NOT_MATCH_BASE_ANSWER__"


def _json_print_code(value: str) -> str:
    return (
        "import json\n"
        f"print(json.dumps({{'main-task': {value}}}, ensure_ascii=False))\n"
    )


def _reader_for(rel: str) -> str:
    ext = os.path.splitext(rel)[1].lower()
    if ext in (".xlsx", ".xls"):
        return "pd.read_excel"
    if ext == ".json":
        return "pd.read_json"
    if ext in (".tsv",):
        return "pd.read_csv"
    return "pd.read_csv"


def _missing_column_code(rel: str) -> str:
    reader = _reader_for(rel)
    sep = ", sep='\\t'" if rel.lower().endswith(".tsv") and reader == "pd.read_csv" else ""
    return (
        "import json\n"
        "import pandas as pd\n"
        f"df = {reader}({rel!r}{sep})\n"
        "value = df['__LEDGER_MISSING_OPERAND_COLUMN__'].sum()\n"
        "print(json.dumps({'main-task': value}, default=str))\n"
    )


def _copy_trace(trace):
    t = copy.deepcopy(trace)
    t.setdefault("files_cited", list(t.get("cited") or []))
    t.setdefault("files_read", list(t.get("files_read") or []))
    return t


def _mutate(row, trace, klass):
    stated = row.get("pred")
    t = _copy_trace(trace)
    reason = ""
    if klass == "a1":
        stated = _perturb_answer(stated)
        reason = "changed stated answer while keeping original recomputation trace"
    elif klass == "a2":
        t["code"] = _json_print_code("None")
        t["stderr"] = ""
        reason = "restricted re-exec returns main-task None"
    elif klass == "b":
        t["stderr"] = "LEDGER_PERTURBATION_FORCED_EXEC_FAILED"
        reason = "marked final trace as exec-failed while preserving stated answer"
    elif klass == "c1":
        cited = list(t.get("cited") or [])
        files_read = set(t.get("files_read") or [])
        read_cited = [c for c in cited if c in files_read]
        if len(cited) <= 1 or not read_cited:
            return None
        removed = read_cited[0]
        t["cited"] = [c for c in cited if c != removed]
        t["files_cited"] = list(t["cited"])
        t["files_read"] = sorted(files_read)
        reason = f"removed actually-read cited file from restricted view: {removed}"
    elif klass == "c2":
        t["cited"] = []
        t["files_cited"] = []
        reason = "erased all resolvable citation references"
    elif klass == "d":
        cited = list(t.get("cited") or [])
        if not cited:
            return None
        rel = cited[0]
        t["code"] = _missing_column_code(rel)
        t["stderr"] = ""
        t["files_read"] = [rel]
        t["files_cited"] = list(cited)
        reason = "re-exec accesses a missing operand column in a cited file"
    elif klass == "unknown":
        t["code"] = "pass\n"
        t["stderr"] = ""
        reason = "re-exec produces no output, yielding deterministic unknown"
    else:
        raise ValueError(f"unknown perturbation class: {klass}")
    return stated, t, reason


def _expected(klass):
    if klass == "unknown":
        return "unknown", None
    return "unsupported", klass


def _is_correct(klass, label, subtype):
    exp_label, exp_subtype = _expected(klass)
    if label != exp_label:
        return False
    if exp_subtype is None:
        return True
    return subtype == exp_subtype


def _build_suite(rows, target_per_class):
    samples = []
    bases = []
    for row in rows:
        trace = _capture_for(row)
        if trace is None:
            continue
        bases.append((row, trace))
    used = {k: 0 for k in CLASSES}
    for klass in CLASSES:
        for row, trace in bases:
            if used[klass] >= target_per_class:
                break
            mutated = _mutate(row, trace, klass)
            if mutated is None:
                continue
            stated, ptrace, construction_reason = mutated
            label, coarse, subtype, detail = P.support_label(stated, ptrace, row["domain"])
            correct = _is_correct(klass, label, subtype)
            samples.append({
                "base_taskid": row["taskid"],
                "domain": row["domain"],
                "setting": row["_setting"],
                "system": row.get("system"),
                "perturb_class": klass,
                "construction_label": klass,
                "construction_reason": construction_reason,
                "detected_label": label,
                "detected_class": coarse,
                "detected_subtype": subtype,
                "correct": bool(correct),
                "support_detail": detail[:500],
                "base_pred": row.get("pred"),
                "perturbed_pred": stated,
            })
            used[klass] += 1
    return samples, {"supported_sources": len(rows), "trace_sources": len(bases), "planned_per_class": target_per_class, "constructed": used}


def _summarize(samples, build_stats):
    by_class = {}
    misses = []
    for klass in CLASSES:
        rows = [s for s in samples if s["perturb_class"] == klass]
        ok = sum(1 for s in rows if s["correct"])
        by_class[klass] = {
            "n": len(rows),
            "correct": ok,
            "detection_rate": round(ok / len(rows), 4) if rows else None,
            "detected": dict(Counter(f'{s["detected_label"]}/{s.get("detected_subtype")}' for s in rows)),
        }
        for s in rows:
            if not s["correct"]:
                exp_label, exp_subtype = _expected(klass)
                misses.append({
                    "base_taskid": s["base_taskid"],
                    "domain": s["domain"],
                    "setting": s["setting"],
                    "perturb_class": klass,
                    "expected": {"label": exp_label, "subtype": exp_subtype},
                    "detected": {"label": s["detected_label"], "subtype": s.get("detected_subtype")},
                    "mechanical_explanation": s.get("support_detail"),
                })
    gate = {
        "criterion": "each perturbation class detection_rate >= 0.90; misses must be mechanically explained",
        "pass": all((row["detection_rate"] is not None and row["detection_rate"] >= 0.90) for row in by_class.values()),
        "note": "c1 is expected to be covered only by perturbations because live unsupported c1 base rate is zero/near-zero.",
    }
    return {"build": build_stats, "by_class": by_class, "misses": misses, "G9_gate": gate}


def _write_doc(summary):
    lines = [
        "# LEDGER E9 - Perturbation suite",
        "",
        "Construction labels come from deterministic perturbation operators, not from checker output.",
        "The checker output is used only as the measured detection result.",
        "",
        f"Supported source episodes: {summary['build']['supported_sources']}",
        f"Trace sources usable: {summary['build']['trace_sources']}",
        f"Planned per class: {summary['build']['planned_per_class']}",
        "",
        "| class | n | correct | detection rate |",
        "|---|---:|---:|---:|",
    ]
    for klass in CLASSES:
        row = summary["by_class"][klass]
        lines.append(f"| {klass} | {row['n']} | {row['correct']} | {row['detection_rate']} |")
    lines += [
        "",
        "## G9",
        "",
        f"Criterion: {summary['G9_gate']['criterion']}",
        f"G9: {'PASS' if summary['G9_gate']['pass'] else 'FAIL'}",
        "",
        "c1 coverage note: c1 is covered by perturbation because live traces have no reliable c1 base rate.",
        "",
        "## Per-miss mechanical explanations",
        "",
    ]
    if not summary["misses"]:
        lines.append("No misses.")
    else:
        for m in summary["misses"]:
            lines.append(
                f"- {m['perturb_class']} {m['domain']}/{m['setting']}/{m['base_taskid']}: "
                f"expected={m['expected']} detected={m['detected']} detail={m['mechanical_explanation']}"
            )
    lines.append("")
    lines.append("Records: `records/kb_perturbation_suite.jsonl`.")
    with open(OUT_DOC, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(target_per_class=TARGET_PER_CLASS):
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_DOC), exist_ok=True)
    rows = _load_supported()
    samples, build_stats = _build_suite(rows, target_per_class)
    with open(OUT_JSONL, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    summary = _summarize(samples, build_stats)
    _write_doc(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


if __name__ == "__main__":
    main()
