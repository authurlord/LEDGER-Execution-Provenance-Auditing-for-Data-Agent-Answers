#!/usr/bin/env python3
"""Run a local evidence-aware judge at temperature zero with three repeats."""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from common import RECORDS, load_jsonl, write_jsonl


SYSTEM = """You are an evidence-aware auditor of completed data-agent answers.
Decide whether the stated answer is supported by the supplied evidence and whether
the recorded program/recomputation is faithful to the operation and scope in the
question. Do not assume missing evidence. Return JSON only:
{"label":"supported|unsupported|unknown","confidence":0.0,"rationale":"brief reason"}
Use unknown when the supplied bundle is insufficient. A successful execution alone
does not establish semantic faithfulness."""


def payload(item):
    return {
        key: item.get(key)
        for key in (
            "question",
            "stated_answer",
            "declared_files",
            "evidence_snippets",
            "actual_file_reads",
            "program_summary",
            "execution_status",
            "recomputed_output",
            "cited_operands",
            "cited_constants",
        )
    }


def parse_json(text: str):
    candidate = text.strip()
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        candidate = match.group(0)
    try:
        value = json.loads(candidate)
        label = str(value.get("label", "")).strip().lower()
        if label not in {"supported", "unsupported", "unknown"}:
            raise ValueError("invalid label")
        return label, float(value.get("confidence", 0.0)), str(value.get("rationale", "")), True
    except Exception as exc:
        return "unknown", 0.0, f"parse failure: {type(exc).__name__}", False


def one_call(client, item, repeat, model, max_tokens):
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(payload(item), ensure_ascii=False, sort_keys=True),
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "seed": 20260730 + repeat,
            },
        )
        content = response.choices[0].message.content or ""
        label, confidence, rationale, parse_ok = parse_json(content)
        usage = response.usage
        return {
            "dataset": item["dataset"],
            "item_id": item["item_id"],
            "repeat": repeat,
            "model": model,
            "temperature": 0,
            "reference_label": item["reference_label"],
            "judge_label": label,
            "confidence": confidence,
            "rationale": rationale,
            "parse_ok": parse_ok,
            "tokens_in": int(getattr(usage, "prompt_tokens", 0) or 0),
            "tokens_out": int(getattr(usage, "completion_tokens", 0) or 0),
            "latency_s": time.perf_counter() - started,
            "cost_usd": 0.0,
        }
    except Exception as exc:
        return {
            "dataset": item["dataset"],
            "item_id": item["item_id"],
            "repeat": repeat,
            "model": model,
            "temperature": 0,
            "reference_label": item["reference_label"],
            "judge_label": "unknown",
            "confidence": 0.0,
            "rationale": f"request failure: {type(exc).__name__}: {str(exc)[:120]}",
            "parse_ok": False,
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_s": time.perf_counter() - started,
            "cost_usd": 0.0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="qwen3.6-35b")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--output",
        type=Path,
        default=RECORDS / "evidence_aware_judge_records.jsonl",
    )
    args = parser.parse_args()
    items = load_jsonl(RECORDS / "evidence_aware_judge_inputs.jsonl")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(output) if output.exists() else []
    done = {(row["item_id"], int(row["repeat"])) for row in existing}
    jobs = [
        (item, repeat)
        for item in items
        for repeat in range(args.repeats)
        if (item["item_id"], repeat) not in done
    ]
    client = OpenAI(base_url=args.base_url, api_key="local-vllm", timeout=90.0)
    new_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                one_call, client, item, repeat, args.model, args.max_tokens
            ): (item, repeat)
            for item, repeat in jobs
        }
        for completed, future in enumerate(as_completed(futures), 1):
            new_rows.append(future.result())
            if completed % 40 == 0:
                write_jsonl(
                    output,
                    sorted(
                        existing + new_rows,
                        key=lambda row: (row["dataset"], row["item_id"], int(row["repeat"])),
                    ),
                )
                print(f"completed {completed}/{len(jobs)}", flush=True)
    write_jsonl(
        output,
        sorted(
            existing + new_rows,
            key=lambda row: (row["dataset"], row["item_id"], int(row["repeat"])),
        ),
    )
    print(f"wrote {len(existing) + len(new_rows)} calls")


if __name__ == "__main__":
    main()
