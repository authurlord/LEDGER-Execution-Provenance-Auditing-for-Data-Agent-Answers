"""LEDGER STEP 0 — cost-accounting instrumentation (prerequisite for every run).

Emits ONE structured JSONL record per task with the mandated schema (see LEDGER guidance):
    {taskid, domain, system, sources_selected, gold_sources, source_precision, source_recall,
     pred, gold, score, tokens_in, tokens_out, tool_calls, turns, latency_s,
     trace_path, provenance, grounding_label}

Design notes
- This module is STANDALONE and GPU-free; it is unit-tested in __main__ with a fake usage object.
- `CostMeter` is the single accumulator threaded through an agent run. `local_qwen.py`
  (LocalQwenGenerator) already sums usage.{prompt,completion,total}_tokens — feed those in via
  `meter.add_llm(usage)`. `fulllake_agent.py` should create one meter per task and call
  `meter.tool()` / `meter.turn()` in its ReAct loop, then `write_record(...)` at submit.

INTEGRATION (do in the GPU session, keep it thin — do not restructure the agents):
  - fulllake_agent.run_one(tf):  meter = CostMeter(); ... meter.add_llm(resp.usage) after each call;
      meter.tool() per tool dispatch; meter.turn() per loop iter; record = build_record(...); writer.write(record)
  - local_qwen SUT: wrap serve_query so the generator's (input_tokens, output_tokens) land in a meter,
      and emit a judged record with cost fields (KramaBench previously wrote only logs — fix that).
A run without populated cost fields is useless for this paper (guidance §STEP 0 gate).
"""
from __future__ import annotations
import json, time, os
from dataclasses import dataclass, field, asdict
from typing import Any

RECORD_FIELDS = [
    "taskid", "domain", "system", "sources_selected", "gold_sources",
    "source_precision", "source_recall", "pred", "gold", "score",
    "tokens_in", "tokens_out", "tool_calls", "turns", "latency_s",
    "trace_path", "provenance", "grounding_label",
]


@dataclass
class CostMeter:
    """One per task. Accumulates cost across the whole ReAct/pipeline run."""
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    turns: int = 0
    _t0: float = field(default_factory=time.perf_counter)

    def add_llm(self, usage: Any) -> "CostMeter":
        """Accept an OpenAI-style usage object OR a dict OR (in,out) tuple."""
        if usage is None:
            return self
        if isinstance(usage, dict):
            self.tokens_in += int(usage.get("prompt_tokens", usage.get("input_tokens", usage.get("tokens_in", 0))) or 0)
            self.tokens_out += int(usage.get("completion_tokens", usage.get("output_tokens", usage.get("tokens_out", 0))) or 0)
        elif isinstance(usage, (tuple, list)) and len(usage) == 2:
            self.tokens_in += int(usage[0] or 0); self.tokens_out += int(usage[1] or 0)
        else:  # object with attributes (accept prompt_tokens/input_tokens naming)
            self.tokens_in += int(getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0)
            self.tokens_out += int(getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0)
        return self

    def tool(self, n: int = 1) -> "CostMeter":
        self.tool_calls += n; return self

    def turn(self, n: int = 1) -> "CostMeter":
        self.turns += n; return self

    @property
    def latency_s(self) -> float:
        return round(time.perf_counter() - self._t0, 3)

    def as_cost(self) -> dict:
        return {"tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "tool_calls": self.tool_calls, "turns": self.turns, "latency_s": self.latency_s}


def source_metrics(selected: list[str], gold: list[str]) -> tuple[float, float]:
    """Precision/recall of the selected source set vs gold data_sources (basename-insensitive)."""
    def base(x): return os.path.basename(str(x)).strip().lower()
    S = {base(x) for x in (selected or [])}
    G = {base(x) for x in (gold or [])}
    if not S and not G:
        return 1.0, 1.0
    hit = len(S & G)
    prec = hit / len(S) if S else 0.0
    rec = hit / len(G) if G else 0.0
    return round(prec, 4), round(rec, 4)


def build_record(*, taskid, domain, system, pred, gold, score,
                 sources_selected=None, gold_sources=None, meter: CostMeter | None = None,
                 trace_path="", provenance=None, grounding_label="unknown",
                 **extra) -> dict:
    """Assemble a schema-complete record. Missing cost -> zeros (flagged by the smoke gate)."""
    sel = sources_selected or []
    gs = gold_sources or []
    prec, rec = source_metrics(sel, gs)
    cost = meter.as_cost() if meter else {"tokens_in": 0, "tokens_out": 0, "tool_calls": 0, "turns": 0, "latency_s": 0.0}
    rec_d = {
        "taskid": taskid, "domain": domain, "system": system,
        "sources_selected": list(sel), "gold_sources": list(gs),
        "source_precision": prec, "source_recall": rec,
        "pred": pred, "gold": gold, "score": score,
        **cost,
        "trace_path": trace_path, "provenance": provenance or [],
        "grounding_label": grounding_label,
    }
    rec_d.update(extra)
    return rec_d


class RecordWriter:
    """Append-only JSONL writer; truncates on first open per run."""
    def __init__(self, path: str, fresh: bool = True):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if fresh:
            open(path, "w").close()

    def write(self, record: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def cost_fields_ok(record: dict) -> bool:
    """STEP-0 smoke gate: cost fields present and plausible (turns>0 and some tokens)."""
    for k in ("tokens_in", "tokens_out", "tool_calls", "turns", "latency_s"):
        if k not in record:
            return False
    return record["turns"] > 0 and (record["tokens_in"] + record["tokens_out"]) > 0


if __name__ == "__main__":
    # GPU-free self-test — proves cost fields populate and the schema is complete.
    class _Usage:
        prompt_tokens = 1200; completion_tokens = 300
    m = CostMeter()
    for _ in range(3):
        m.add_llm(_Usage()); m.turn()
    m.tool(4); m.add_llm({"prompt_tokens": 50, "completion_tokens": 10}); m.add_llm((100, 20))
    r = build_record(taskid="environment-easy-1", domain="environment", system="B3",
                     pred=4.796, gold=4.796, score=1.0,
                     sources_selected=["water-body-testing-2013.csv", "distractor.csv"],
                     gold_sources=["water-body-testing-2013.csv"], meter=m,
                     grounding_label="supported")
    assert set(RECORD_FIELDS).issubset(r), "schema incomplete"
    assert r["source_precision"] == 0.5 and r["source_recall"] == 1.0, r
    assert r["tokens_in"] == 1200*3+50+100 and r["tokens_out"] == 300*3+10+20, r
    assert cost_fields_ok(r), "cost gate should pass"
    assert not cost_fields_ok({**r, "turns": 0, "tokens_in": 0, "tokens_out": 0})
    print("cost_accounting self-test OK:", json.dumps(r, ensure_ascii=False))
