#!/usr/bin/env python3
"""Calibration-only closure grid plus frozen-point transfer results."""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict

import numpy as np

from common import RECORDS, ROOT, finite_float, load_jsonl, write_json
from kramabench_integration.provenance import ConstrainedPool


TOLERANCES = (1e-6, 1e-5, 1e-4, 1e-3)
DEPTHS = (1, 2, 3)
OPSETS = (
    ("sum", "diff", "ratio"),
    ("sum", "diff", "ratio", "pctchg"),
    ("sum", "diff", "ratio", "pctchg", "prod"),
)
SOURCES = ("reached_operands", "all_numbers_in_cited_files", "all_context_numbers")
RESTRICTIONS = ("citation_restricted", "unrestricted")
NUMERIC = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")


def numbers(value):
    out = []
    for token in NUMERIC.findall(str(value)):
        number = finite_float(token)
        if number is not None and abs(number) < 1e15:
            out.append(number)
    return sorted(set(out))


def raw_index():
    items = json.load(
        (ROOT / "datasets" / "convfinqa_original" / "data" / "dev.json").open(
            encoding="utf-8"
        )
    )
    return {item["id"]: item for item in items}


class TargetClosure:
    """Target-directed exact closure, avoiding all-context D2 materialization."""

    def __init__(self, operands, consts, ops):
        self.ops = tuple(ops)
        self.d0 = np.asarray(sorted(set(float(x) for x in [*operands, *consts])), dtype=float)
        self.d1 = self._d1()
        self.right = np.unique(np.concatenate((self.d0, self.d1)))

    def _d1(self):
        if not len(self.d0):
            return self.d0
        a, b = self.d0[:, None], self.d0[None, :]
        values = []
        if "sum" in self.ops:
            values.append((a + b).ravel())
        if "diff" in self.ops:
            values.extend(((a - b).ravel(), (b - a).ravel()))
        with np.errstate(divide="ignore", invalid="ignore"):
            if "ratio" in self.ops:
                values.extend((
                    np.where(b != 0, a / b, np.nan).ravel(),
                    np.where(a != 0, b / a, np.nan).ravel(),
                ))
            if "pctchg" in self.ops:
                values.extend((
                    np.where(b != 0, (a - b) / b, np.nan).ravel(),
                    np.where(a != 0, (b - a) / a, np.nan).ravel(),
                ))
        if "prod" in self.ops:
            values.append((a * b).ravel())
        merged = np.concatenate(values) if values else np.asarray([], dtype=float)
        return np.unique(merged[np.isfinite(merged) & (np.abs(merged) < 1e15)])

    @staticmethod
    def _member(values, target, tol):
        if not len(values):
            return False
        margin = max(tol * abs(target), 1e-9)
        index = np.searchsorted(values, target)
        lo = max(0, index - 1)
        hi = min(len(values), index + 2)
        return bool(np.any(np.abs(values[lo:hi] - target) <= margin))

    def _any_members(self, candidates, values, tol):
        candidates = np.asarray(candidates, dtype=float)
        candidates = candidates[np.isfinite(candidates) & (np.abs(candidates) < 1e15)]
        if not len(candidates) or not len(values):
            return False
        indices = np.searchsorted(values, candidates)
        margin = np.maximum(tol * np.abs(candidates), 1e-9)
        for offset in (-1, 0):
            positions = np.clip(indices + offset, 0, len(values) - 1)
            if np.any(np.abs(values[positions] - candidates) <= margin):
                return True
        return False

    def _depth2(self, target, tol):
        a = self.d1
        right = self.right
        if not len(a):
            return False
        if "sum" in self.ops and self._any_members(target - a, right, tol):
            return True
        if "diff" in self.ops and (
            self._any_members(a - target, right, tol)
            or self._any_members(target + a, right, tol)
        ):
            return True
        with np.errstate(divide="ignore", invalid="ignore"):
            if "ratio" in self.ops and (
                (target != 0 and self._any_members(a / target, right, tol))
                or self._any_members(target * a, right, tol)
            ):
                return True
            if "pctchg" in self.ops and (
                (1 + target != 0 and self._any_members(a / (1 + target), right, tol))
                or self._any_members(a * (1 + target), right, tol)
            ):
                return True
            if "prod" in self.ops and self._any_members(
                np.where(a != 0, target / a, np.nan), right, tol
            ):
                return True
        return False

    def check(self, value, tol, percent_ctx, max_depth):
        number = finite_float(value)
        if number is None or not len(self.d0):
            return None
        targets = [number, number / 100.0, number * 100.0] if percent_ctx and number else [number]
        if any(self._member(self.d0, target, tol) for target in targets):
            return "depth0"
        if max_depth >= 1 and any(self._member(self.d1, target, tol) for target in targets):
            return "depth1"
        if max_depth >= 2 and any(self._depth2(target, tol) for target in targets):
            return "depth2"
        if max_depth >= 3:
            for target in targets:
                for d0 in self.d0:
                    candidates = []
                    if "sum" in self.ops:
                        candidates.append(target - d0)
                    if "diff" in self.ops:
                        candidates.extend((target + d0, d0 - target))
                    if "ratio" in self.ops:
                        if d0:
                            candidates.append(target * d0)
                        if target:
                            candidates.append(d0 / target)
                    if "pctchg" in self.ops:
                        candidates.append(d0 * (1 + target))
                        if 1 + target:
                            candidates.append(d0 / (1 + target))
                    if "prod" in self.ops and d0:
                        candidates.append(target / d0)
                    if any(self._depth2(candidate, tol) for candidate in candidates):
                        return "depth3"
        return None

    def expression_count_upper_bound(self, max_depth):
        n0, n1 = len(self.d0), len(self.d1)
        factor = (
            int("sum" in self.ops)
            + 2 * int("diff" in self.ops)
            + 2 * int("ratio" in self.ops)
            + 2 * int("pctchg" in self.ops)
            + int("prod" in self.ops)
        )
        count = n0 + n1
        if max_depth >= 2:
            count += factor * n1 * (n0 + n1)
        if max_depth >= 3:
            count += factor * n0 * max(1, factor * n1 * (n0 + n1))
        return int(count)


class MaterializedClosure:
    """Adapter around the frozen implementation for reached-operand settings."""

    def __init__(self, operands, consts, ops):
        self.pool = ConstrainedPool(operands, consts, ops, chain="full")

    def check(self, value, tol, percent_ctx, max_depth):
        return self.pool.check(
            value,
            tol_rel=tol,
            percent=True,
            percent_ctx=percent_ctx,
            max_depth=max_depth,
        )

    def expression_count_upper_bound(self, max_depth):
        count = len(self.pool.d0) + len(self.pool.d1)
        if max_depth >= 2:
            count += len(self.pool.d2)
        return int(count)


def source_pools(source_id, provenance, raw):
    base_id = source_id.split("::qa", 1)[0]
    item = raw[base_id]
    reached = [
        *(provenance.get("cited_operands") or []),
        *(provenance.get("cited_consts") or []),
    ]
    cited = numbers(item.get("table", []))
    context = numbers({
        "table": item.get("table", []),
        "pre_text": item.get("pre_text", []),
        "post_text": item.get("post_text", []),
    })
    return {
        "reached_operands": reached,
        "all_numbers_in_cited_files": cited,
        "all_context_numbers": context,
    }


def pair_rows(path):
    groups = defaultdict(dict)
    for row in load_jsonl(path):
        taskid = row["taskid"]
        if taskid.endswith("::positive"):
            groups[taskid.removesuffix("::positive")]["positive"] = row
        elif taskid.endswith("::negative"):
            groups[taskid.removesuffix("::negative")]["negative"] = row
    return [
        (source_id, rows["positive"], rows["negative"])
        for source_id, rows in groups.items()
        if set(rows) == {"positive", "negative"}
    ]


def seed_pairs():
    rows = json.load((RECORDS / "convfinqa_constrained_seed_detail.json").open())
    pairs = []
    for row in rows:
        provenance = {
            "cited_operands": row["operands"],
            "cited_consts": row["consts"],
        }
        shared = {
            "provenance": [provenance],
            "percent_ctx": row["percent_ctx"],
        }
        pairs.append((
            row["uid"],
            {**shared, "pred": row["gold"]},
            {**shared, "pred": row["swap"]},
        ))
    return pairs


def main():
    raw = raw_index()
    calibration = seed_pairs()
    wrong = load_jsonl(RECORDS / "convfinqa_wrong_operation_negatives.jsonl")
    cache = {}

    def get_pool(case_id, source, restriction, ops, provenance):
        effective = "all_context_numbers" if restriction == "unrestricted" else source
        key = (case_id, effective, ops)
        if key not in cache:
            pools = source_pools(case_id, provenance, raw)
            if effective == "reached_operands":
                cache[key] = MaterializedClosure(
                    provenance.get("cited_operands") or [],
                    provenance.get("cited_consts") or [],
                    ops,
                )
            else:
                cache[key] = TargetClosure(pools[effective], (), ops)
        return cache[key]

    rows = []
    for source in SOURCES:
        for depth in DEPTHS:
            for tolerance in TOLERANCES:
                for ops in OPSETS:
                    for restriction in RESTRICTIONS:
                        started = time.perf_counter()
                        support, detected, expressions = 0, 0, []
                        for case_id, positive, negative in calibration:
                            provenance = (positive.get("provenance") or [{}])[0]
                            pool = get_pool(case_id, source, restriction, ops, provenance)
                            support += bool(pool.check(
                                positive.get("pred"), tolerance,
                                bool(positive.get("percent_ctx")), depth,
                            ))
                            detected += not bool(pool.check(
                                negative.get("pred"), tolerance,
                                bool(negative.get("percent_ctx")), depth,
                            ))
                            expressions.append(pool.expression_count_upper_bound(depth))
                        wrong_supported = 0
                        for negative in wrong:
                            case_id = negative["taskid"].removesuffix("::wrong_operation_negative")
                            provenance = (negative.get("provenance") or [{}])[0]
                            pool = get_pool(case_id, source, restriction, ops, provenance)
                            wrong_supported += bool(pool.check(
                                negative.get("pred"), tolerance,
                                bool(negative.get("percent_ctx")), depth,
                            ))
                        rows.append({
                            "selection_split": "convfinqa_seed_60_pairs",
                            "operand_source": source,
                            "restriction": restriction,
                            "max_depth": depth,
                            "tol_rel": tolerance,
                            "operations": list(ops),
                            "support_recall": support / len(calibration),
                            "support_recall_counts": f"{support}/{len(calibration)}",
                            "unsupported_detection": detected / len(calibration),
                            "unsupported_detection_counts": f"{detected}/{len(calibration)}",
                            "wrong_operation_false_support_rate": wrong_supported / len(wrong),
                            "wrong_operation_false_support_counts": f"{wrong_supported}/{len(wrong)}",
                            "median_candidate_expression_upper_bound": int(np.median(expressions)),
                            "runtime_s": time.perf_counter() - started,
                        })
    payload = {
        "schema_version": 1,
        "grid": rows,
        "frozen_selection": {
            "operand_source": "reached_operands",
            "restriction": "citation_restricted",
            "max_depth": 2,
            "tol_rel": 1e-4,
            "operations": ["sum", "diff", "ratio", "pctchg", "prod"],
            "selection_rule": "pre-existing frozen point selected on ConvFinQA seed only",
        },
        "frozen_transfer": {
            "convfinqa_remainder": {
                "support_recall_counts": "360/386",
                "unsupported_detection_counts": "371/386",
            },
            "multihiertt": {
                "support_recall_counts": "5648/6301",
                "unsupported_detection_counts": "6125/6301",
            },
            "same_operand_wrong_operation": {
                "detected_counts": "9/382",
                "within_frozen_ops_and_depth": "0/347",
            },
        },
        "candidate_size_note": (
            "For target-directed context closure, candidate size is reported as an "
            "expression-count upper bound; D2 is not materialized."
        ),
    }
    write_json(RECORDS / "closure_ablation_grid_rev2.json", payload)
    print(f"wrote {len(rows)} grid cells; cached {len(cache)} case/source/op pools")


if __name__ == "__main__":
    main()
