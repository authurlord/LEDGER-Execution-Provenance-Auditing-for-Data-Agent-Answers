"""LEDGER E2 - LLM judge baseline smoke/full runner.

Default is smoke only:
  python3 -m kramabench_integration.run_llm_judge_baseline

Full run is explicit:
  python3 -m kramabench_integration.run_llm_judge_baseline --mode full --repeats 3

This script never writes `grounding_label`; judge verdicts are isolated under
`judge_label` and compared to `reference_label`.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import glob
import json
import os
import re
import time
from typing import Any

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import convfinqa_gold as G
from .convfinqa_constrained_calib import swap_negative

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(REPO, ".env")
OUT_RECORDS = os.path.join(REPO, "records", "judge_baseline_records.jsonl")
OUT_SUMMARY = os.path.join(REPO, "records", "judge_baseline_summary.json")
OUT_DOC = os.path.join(REPO, "docs", "LLM_JUDGE_BASELINE.md")

DEFAULT_MODEL = "deepseek-v4-flash"
LABELS = ("supported", "unsupported", "unknown")

# Placeholder constants only. Fill with the exact contracted DeepSeek prices before
# using dollar numbers in the paper.
# DeepSeek official pricing (per 1M tokens; deepseek-v4-flash cache-miss; verify at
# https://api-docs.deepseek.com/quick_start/pricing — subject to change).
PRICE_PER_1M_INPUT_USD = 0.14
PRICE_PER_1M_OUTPUT_USD = 0.28


def read_env(path=ENV_PATH):
    vals = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing env file: {path}")
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        vals[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY") if not vals.get(k)]
    if missing:
        raise RuntimeError(f"missing required .env keys: {', '.join(missing)}")
    return vals


def _as_text(value):
    if value is None:
        return "None"
    return str(value)


def _truncate(value, n=1200):
    s = _as_text(value)
    return s if len(s) <= n else s[:n] + "...[truncated]"


def _normalize_label(value):
    s = _as_text(value).strip().lower()
    return s if s in LABELS else "unknown"


def _json_extract(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


def _task_questions():
    path = os.path.join(REPO, "datasets", "normalized", "kramabench_tasks.jsonl")
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        row = json.loads(line)
        out[row["taskid"]] = row.get("query") or ""
    return out


def _cited_from_provenance(provenance):
    files, columns, ops, operands = [], [], [], []
    for item in provenance or []:
        if item.get("source_file"):
            files.append(item.get("source_file"))
        columns.extend(item.get("columns") or [])
        ops.extend(item.get("ops") or [])
        operands.extend(item.get("cited_operands") or [])
    return {
        "files": sorted(set(map(str, files))),
        "columns": sorted(set(map(str, columns))),
        "ops": sorted(set(map(str, ops))),
        "operands": sorted(set(map(str, operands))),
    }


def load_convfinqa_items(limit_examples=100):
    qas = [q for q in G.load_qas() if q.get("ok") and q.get("exe_ans") is not None]
    qas = qas[:limit_examples]
    items = []
    for i, q in enumerate(qas):
        cited = {"operands": q["operands"], "consts": q["consts"], "program": q["program"]}
        base = {
            "dataset": "convfinqa_constructed",
            "source_id": q["uid"],
            "question": q["question"],
            "cited": cited,
            "program_summary": q["program"],
        }
        items.append({**base, "item_id": f"convfinqa::{q['uid']}::positive",
                      "stated_answer": q["exe_ans"], "reference_label": "supported"})
        items.append({**base, "item_id": f"convfinqa::{q['uid']}::negative",
                      "stated_answer": swap_negative(qas, i), "reference_label": "unsupported"})
    return items


def load_kramabench_items():
    questions = _task_questions()
    items = []
    for path in sorted(glob.glob(os.path.join(REPO, "records", "records_ledger_n2_kb_*.jsonl"))):
        source_name = os.path.basename(path)
        for line in open(path):
            row = json.loads(line)
            pred = row.get("pred")
            if pred is None or str(pred).strip() == "":
                continue
            reference_label = _normalize_label(row.get("grounding_label"))
            if reference_label == "unknown":
                continue
            cited = _cited_from_provenance(row.get("provenance") or [])
            offered = row.get("offered_files") or row.get("sources_selected") or []
            cited["offered_files"] = list(map(str, offered))
            items.append({
                "dataset": "kramabench_stated_answer",
                "item_id": f"kramabench::{source_name}::{row['taskid']}",
                "source_id": row["taskid"],
                "question": questions.get(row["taskid"], ""),
                "stated_answer": pred,
                "reference_label": reference_label,
                "cited": cited,
                "program_summary": "; ".join(cited.get("ops") or []),
                "record_source": os.path.relpath(path, REPO),
            })
    return items


def load_annotation_items():
    path = os.path.join(REPO, "docs", "annotation", "KB_SUPPORT_ANNOTATION.csv")
    items = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cited = {
                "files": [x for x in (row.get("cited_files") or "").split(";") if x],
                "operands": [x for x in (row.get("cited_operands") or "").split(";") if x],
                "snippet_paths": [x for x in (row.get("snippet_paths") or "").split(";") if x],
            }
            taskid = row.get("taskid") or f"row_{len(items)}"
            items.append({
                "dataset": "kb_support_annotation",
                "item_id": f"annotation::{taskid}",
                "source_id": taskid,
                "question": row.get("question") or "",
                "stated_answer": row.get("stated_answer"),
                "reference_label": _normalize_label(row.get("checker_label")),
                "cited": cited,
                "program_summary": "",
                "domain": row.get("domain"),
                "annotation_note": "Compared to checker_label only; human columns are not read or written.",
            })
    return items


def build_prompt(item):
    cited = item.get("cited") or {}
    user = {
        "question": item.get("question") or "",
        "stated_answer": item.get("stated_answer"),
        "cited_files": cited.get("files") or cited.get("offered_files") or [],
        "cited_columns": cited.get("columns") or [],
        "cited_operands": cited.get("operands") or [],
        "cited_consts": cited.get("consts") or [],
        "program_or_ops_summary": item.get("program_summary") or cited.get("ops") or "",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict evidence-support judge. Decide whether the stated answer is "
                "supported by the cited files, columns, operands, and operation/program summary. "
                "Do not judge against any hidden gold answer. Return only JSON with keys "
                "support_label, rationale, confidence. support_label must be one of "
                "supported, unsupported, unknown. confidence must be a number from 0 to 1."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate support for this stated answer. No gold answer is provided.\n\n"
                f"{json.dumps(user, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def call_judge(env, model, item, timeout_s=60):
    base = env["DEEPSEEK_BASE_URL"].rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload = {
        "model": model,
        "messages": build_prompt(item),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {env['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    latency = round(time.perf_counter() - t0, 3)
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    parsed = _json_extract(content)
    label = _normalize_label(parsed.get("support_label"))
    conf = parsed.get("confidence")
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except Exception:
        conf = None
    usage = body.get("usage") or {}
    tokens_in = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    tokens_out = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    cost = (tokens_in / 1_000_000) * PRICE_PER_1M_INPUT_USD + (tokens_out / 1_000_000) * PRICE_PER_1M_OUTPUT_USD
    return {
        "judge_label": label,
        "rationale": _truncate(parsed.get("rationale", ""), 1000),
        "confidence": conf,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_s": latency,
        "cost_usd": round(cost, 8),
        "raw_finish_reason": body["choices"][0].get("finish_reason"),
        "parse_ok": True,
    }


def _safe_call(env, model, item, timeout_s):
    try:
        return call_judge(env, model, item, timeout_s)
    except Exception as e:
        return {
            "judge_label": "unknown",
            "rationale": "call_or_parse_error",
            "confidence": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_s": 0.0,
            "cost_usd": 0.0,
            "error": str(e)[:300],
            "parse_ok": False,
        }


def _metrics(records):
    use = [r for r in records if r.get("reference_label") in LABELS]
    total = len(use)
    correct = sum(1 for r in use if r.get("judge_label") == r.get("reference_label"))
    per_label = {}
    f1s = []
    for lab in LABELS:
        tp = sum(1 for r in use if r["judge_label"] == lab and r["reference_label"] == lab)
        fp = sum(1 for r in use if r["judge_label"] == lab and r["reference_label"] != lab)
        fn = sum(1 for r in use if r["judge_label"] != lab and r["reference_label"] == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per_label[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return {
        "n": total,
        "accuracy": round(correct / total, 4) if total else None,
        "correct": correct,
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "per_label": per_label,
        "confusion": {f"{a}->{b}": c for (a, b), c in sorted(Counter((r["reference_label"], r["judge_label"]) for r in use).items())},
    }


def _flip_rate(records):
    by_item = defaultdict(list)
    for r in records:
        by_item[r["item_id"]].append(r["judge_label"])
    eligible = [labs for labs in by_item.values() if len(labs) > 1]
    flips = sum(1 for labs in eligible if len(set(labs)) > 1)
    return {"eligible_items": len(eligible), "flipped_items": flips,
            "flip_rate": round(flips / len(eligible), 4) if eligible else 0.0}


def _select_items(mode, per_group):
    groups = {
        "convfinqa_constructed": load_convfinqa_items(),
        "kramabench_stated_answer": load_kramabench_items(),
        "kb_support_annotation": load_annotation_items(),
    }
    if mode == "smoke":
        groups = {k: v[:per_group] for k, v in groups.items()}
    return groups


def _write_doc(summary):
    lines = [
        "# LEDGER E2 - LLM judge baseline",
        "",
        f"Mode: `{summary['mode']}`. Model: `{summary['model']}`. Temperature: 0.",
        "",
        "The judge is evaluated as an external baseline only. Its output is written as `judge_label`",
        "and never as `grounding_label`; deterministic LEDGER verdict records are not modified.",
        "",
        "## Smoke summary" if summary["mode"] == "smoke" else "## Summary",
        "",
        f"- Calls: {summary['calls']}",
        f"- Parse errors: {summary['parse_errors']}",
        f"- Total tokens in/out: {summary['tokens_in']} / {summary['tokens_out']}",
        f"- Cost per 1000 items: {summary['cost_per_1000_usd']} USD (placeholder pricing constants)",
        f"- Flip rate: {summary['flip_rate']['flip_rate']} ({summary['flip_rate']['flipped_items']}/{summary['flip_rate']['eligible_items']})",
        f"- LEDGER comparator: {summary['ledger_comparator']['calls']} calls, "
        f"{summary['ledger_comparator']['latency_ms']} ms, {summary['ledger_comparator']['cost_usd']} USD",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(summary["metrics_by_group"], indent=1),
        "```",
        "",
        "Records: `records/judge_baseline_records.jsonl`.",
        "Summary: `records/judge_baseline_summary.json`.",
    ]
    with open(OUT_DOC, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--per-group", type=int, default=5)
    ap.add_argument("--timeout-s", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    repeats = args.repeats if args.repeats is not None else (1 if args.mode == "smoke" else 3)

    env = read_env()
    groups = _select_items(args.mode, args.per_group)
    os.makedirs(os.path.join(REPO, "records"), exist_ok=True)
    os.makedirs(os.path.join(REPO, "docs"), exist_ok=True)
    open(OUT_RECORDS, "w").close()

    tasks = [(g, item, rep) for g, items in groups.items()
             for item in items for rep in range(repeats)]

    def _run_one(task):
        group_name, item, rep = task
        judged = _safe_call(env, args.model, item, args.timeout_s)
        record = {
            "run_mode": args.mode,
            "dataset": group_name,
            "item_id": item["item_id"],
            "source_id": item.get("source_id"),
            "repeat": rep,
            "model": args.model,
            "temperature": 0,
            "question": item.get("question"),
            "stated_answer": item.get("stated_answer"),
            "cited": item.get("cited"),
            "program_summary": item.get("program_summary"),
            "reference_label": item.get("reference_label"),
            **judged,
        }
        # Hard guard: this baseline must not write into the deterministic verdict field.
        assert "grounding_label" not in record
        return record

    # Concurrent judge calls (deepseek); preserve deterministic order by task index.
    records = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut2idx = {ex.submit(_run_one, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(fut2idx):
            records[fut2idx[fut]] = fut.result()
    with open(OUT_RECORDS, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    calls = len(records)
    total_cost = round(sum(r.get("cost_usd", 0.0) for r in records), 8)
    by_group = {g: [r for r in records if r["dataset"] == g] for g in groups}
    summary = {
        "mode": args.mode,
        "model": args.model,
        "temperature": 0,
        "repeats": repeats,
        "items_by_group": {k: len(v) for k, v in groups.items()},
        "calls": calls,
        "parse_errors": sum(1 for r in records if not r.get("parse_ok")),
        "tokens_in": sum(r.get("tokens_in", 0) for r in records),
        "tokens_out": sum(r.get("tokens_out", 0) for r in records),
        "total_cost_usd": total_cost,
        "cost_per_1000_usd": round(total_cost / calls * 1000, 8) if calls else 0.0,
        "pricing_note": {
            "input_per_1m_usd": PRICE_PER_1M_INPUT_USD,
            "output_per_1m_usd": PRICE_PER_1M_OUTPUT_USD,
            "status": "placeholder constants",
        },
        "flip_rate": _flip_rate(records),
        "metrics_overall": _metrics(records),
        "metrics_by_group": {g: _metrics(rs) for g, rs in by_group.items()},
        "ledger_comparator": {"calls": 0, "latency_ms": "<10", "cost_usd": 0.0},
        "outputs": {
            "records": os.path.relpath(OUT_RECORDS, REPO),
            "summary": os.path.relpath(OUT_SUMMARY, REPO),
            "doc": os.path.relpath(OUT_DOC, REPO),
        },
        "safety": {
            "does_not_write_grounding_label": all("grounding_label" not in r for r in records),
            "human_columns_written": False,
            "gold_answer_provided_to_judge": False,
        },
    }
    json.dump(summary, open(OUT_SUMMARY, "w"), indent=1)
    _write_doc(summary)

    print(json.dumps({
        "mode": summary["mode"],
        "model": summary["model"],
        "items_by_group": summary["items_by_group"],
        "calls": summary["calls"],
        "parse_errors": summary["parse_errors"],
        "metrics_overall": summary["metrics_overall"],
        "flip_rate": summary["flip_rate"],
        "tokens_in": summary["tokens_in"],
        "tokens_out": summary["tokens_out"],
        "cost_per_1000_usd": summary["cost_per_1000_usd"],
        "outputs": summary["outputs"],
    }, indent=1))
    return summary


if __name__ == "__main__":
    main()
