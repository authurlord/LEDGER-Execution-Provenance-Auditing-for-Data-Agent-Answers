"""Shared, deterministic helpers for the LEDGER IP&MC revision experiments."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
ARTIFACTS = ROOT / "artifacts" / "ipmc_rev2"
DOCS = ROOT / "docs"
LOGS = ROOT / "logs" / "ipmc_rev2"


def load_json(path: Path | str) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def write_json(path: Path | str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def relative_distance(value: Any, candidates: Iterable[Any]) -> float:
    number = finite_float(value)
    parsed = [candidate for candidate in (finite_float(x) for x in candidates) if candidate is not None]
    if number is None or not parsed:
        return 1.0
    denominator = max(abs(number), 1.0)
    return min(abs(number - candidate) / denominator for candidate in parsed)


def normalized_text(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    text = re.sub(r"[\s,]+", " ", text)
    return text.strip(" '\"[]()")


def values_equal(left: Any, right: Any, tol_rel: float = 1e-4) -> bool:
    lnum, rnum = finite_float(left), finite_float(right)
    if lnum is not None and rnum is not None:
        return abs(lnum - rnum) <= max(tol_rel * max(abs(lnum), abs(rnum)), 1e-8)
    return normalized_text(left) == normalized_text(right) and normalized_text(left) != ""


def binary_metrics(y_true: list[int], accepted: list[bool]) -> dict[str, Any]:
    """Metrics for support acceptance.

    ``y_true=1`` means supported, while ``accepted=True`` means that a method
    would issue a support witness or deliver the answer.
    """

    if len(y_true) != len(accepted):
        raise ValueError("length mismatch")
    support_n = sum(y_true)
    unsupported_n = len(y_true) - support_n
    support_accept = sum(label == 1 and keep for label, keep in zip(y_true, accepted))
    unsupported_reject = sum(label == 0 and not keep for label, keep in zip(y_true, accepted))
    return {
        "n": len(y_true),
        "support_n": support_n,
        "unsupported_n": unsupported_n,
        "support_recall": support_accept / support_n if support_n else None,
        "support_recall_counts": f"{support_accept}/{support_n}",
        "unsupported_detection": unsupported_reject / unsupported_n if unsupported_n else None,
        "unsupported_detection_counts": f"{unsupported_reject}/{unsupported_n}",
        "accepted": sum(bool(x) for x in accepted),
    }


_RECOMPUTED = re.compile(r"\brecomputed=(?:'([^']*)'|\"([^\"]*)\"|([^;]+))")


def extract_recomputed(detail: Any) -> str | None:
    """Extract the replayed value from a canonical support-detail string."""

    match = _RECOMPUTED.search(str(detail or ""))
    if not match:
        return None
    value = next((group for group in match.groups() if group is not None), "").strip()
    return value if value.lower() not in {"", "none", "null", "nan"} else None


def has_comparable_recomputed(detail: Any) -> bool:
    return extract_recomputed(detail) is not None


def outcome_counts(rows: Iterable[dict[str, Any]], field: str = "grounding_label") -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))
