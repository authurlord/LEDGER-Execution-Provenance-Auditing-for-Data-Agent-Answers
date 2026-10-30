#!/usr/bin/env python3
"""Hermetically reproduce the identity-theft running example from its bundle.

Run this file from any checkout or extracted release.  It uses only evidence
stored beside the script; the original smart01 paths and full KramaBench clone
are not required.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
ORIGINAL_CSV_BASE = Path(
    "/mnt/data/yanmy/LakeOrch/KramaBench/data/legal/input/"
    "csn-data-book-2024-csv/CSVs"
)
HOST_CSV_BASE = (
    ROOT
    / "evidence"
    / "host_read"
    / "input"
    / "csn-data-book-2024-csv"
    / "CSVs"
)
DECLARED_CSV_BASE = (
    ROOT
    / "evidence"
    / "declared"
    / "input"
    / "csn-data-book-2024-csv"
    / "CSVs"
)
REFERENCE_ROOT = ROOT / "evidence" / "reference" / "input"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes() -> int:
    checked = 0
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, rel = line.split("  ", 1)
        path = ROOT / rel
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"SHA-256 mismatch for {rel}: {actual} != {expected}")
        checked += 1
    return checked


def scrubbed_env() -> dict[str, str]:
    safe = {}
    for key, value in os.environ.items():
        if any(
            marker in key.upper()
            for marker in (
                "KEY",
                "TOKEN",
                "SECRET",
                "PASSWORD",
                "OPENAI",
                "DEEPSEEK",
                "ANTHROPIC",
            )
        ):
            continue
        safe[key] = value
    return safe


def execute_final_program(csv_base: Path) -> dict:
    code = (ROOT / "final_program.py").read_text(encoding="utf-8")
    code = code.replace(str(ORIGINAL_CSV_BASE), str(csv_base.resolve()))
    with tempfile.TemporaryDirectory(prefix="identity_theft_reproduce_") as tmp:
        program = Path(tmp) / "final_program.py"
        program.write_text(code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(program)],
            cwd=tmp,
            env=scrubbed_env(),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    if result.returncode != 0 or result.stderr:
        raise AssertionError(
            f"program failed: exit={result.returncode} stderr={result.stderr!r}"
        )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": json.loads(result.stdout),
    }


def convert_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value:
            return None
        if value.startswith("$"):
            value = value[1:]
        match = re.match(r"^([\d.]+)([MK]?)$", value, re.IGNORECASE)
        if match:
            number, suffix = match.groups()
            number = float(number)
            if suffix.upper() == "M":
                number *= 1_000_000
            elif suffix.upper() == "K":
                number *= 1_000
            return int(number) if number.is_integer() else number
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value
    return value


def read_clean_numeric_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    header_idx = None
    for idx, line in enumerate(lines):
        parts = [part.strip() for part in line.split(",")]
        if sum(bool(part) for part in parts) >= 2:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"no suitable header row in {path}")
    data_lines = []
    for line in lines[header_idx:]:
        parts = [part.strip() for part in line.split(",")]
        if sum(bool(part) for part in parts) < 2:
            break
        data_lines.append(line)
    frame = pd.read_csv(io.StringIO("\n".join(data_lines)))
    for column in frame.columns:
        frame[column] = frame[column].apply(convert_value)
    return frame


def reference_recompute() -> dict:
    html_path = REFERENCE_ROOT / "metropolitan_statistics.html"
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    target = None
    for table in soup.find_all("table"):
        caption = table.find("caption")
        if caption and "the 387 metropolitan statistical areas" in caption.get_text().lower():
            target = table
            break
    if target is None:
        raise AssertionError("reference population table not found")
    html_rows = target.find_all("tr")
    headers = [cell.get_text(" ", strip=True) for cell in html_rows[0].find_all(["th", "td"])]
    values = []
    for row in html_rows[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if len(cells) == len(headers):
            values.append(cells)
    population = pd.DataFrame(values, columns=headers)
    for column in ("2024 estimate", "2020 census"):
        population[column] = pd.to_numeric(
            population[column].str.replace(",", "", regex=False), errors="coerce"
        )
    population["2023 interpolated"] = (
        population["2020 census"]
        + (3 / 4) * (population["2024 estimate"] - population["2020 census"])
    ).astype(int)

    state_dir = (
        REFERENCE_ROOT
        / "csn-data-book-2024-csv"
        / "CSVs"
        / "State MSA Identity Theft data"
    )
    reports = pd.concat(
        [read_clean_numeric_csv(path) for path in state_dir.glob("*.csv")],
        ignore_index=True,
    )

    def normalize_name(name: str) -> str:
        name = name.lower()
        name = re.sub(r"\s*(metropolitan statistical area|msa)$", "", name)
        return re.sub(r"[^a-z0-9]+", "", name)

    reports["msa_key"] = reports["Metropolitan Area"].apply(normalize_name)
    population["msa_key"] = population["Metropolitan statistical area"].apply(normalize_name)
    reports.drop_duplicates(inplace=True)
    population.drop_duplicates(inplace=True)
    merged = pd.merge(reports, population, on="msa_key", how="inner")
    large = merged[merged["2023 interpolated"] > 1_000_000]
    mean_value = float(large["# of Reports"].mean())
    return {
        "matched_rows": int(len(merged)),
        "population_gt_1m_rows": int(len(large)),
        "sum_reports": float(large["# of Reports"].sum()),
        "mean_reports": mean_value,
        "rounded_4dp": round(mean_value, 4),
    }


def main() -> None:
    episode = json.loads((ROOT / "episode.json").read_text(encoding="utf-8"))
    checked_files = verify_hashes()
    host = execute_final_program(HOST_CSV_BASE)
    replay = execute_final_program(DECLARED_CSV_BASE)
    reference = reference_recompute()

    host_value = host["output"].get("main-task")
    replay_value = replay["output"].get("main-task")
    if host_value != episode["host_value"] or host_value != episode["stated_answer"]:
        raise AssertionError("host output does not match the preserved stated answer")
    if replay_value is not None:
        raise AssertionError(f"restricted replay unexpectedly produced {replay_value!r}")
    if reference["rounded_4dp"] != episode["official_answer"]:
        raise AssertionError("reference recomputation does not match the official answer")

    result = {
        "status": "PASS",
        "bundle": str(ROOT),
        "sha256_files_verified": checked_files,
        "host": {
            "exit_code": host["exit_code"],
            "value": host_value,
            "equals_stated_answer": True,
        },
        "restricted_replay": {
            "exit_code": replay["exit_code"],
            "value": replay_value,
            "comparable_to_stated_answer": False,
        },
        "benchmark_reference": reference,
        "audit": {
            "certificate_issued": False,
            "failure_code": episode["failure_code"],
            "host_subset_of_resolved_declared": episode["evidence_counts"][
                "host_subset_of_resolved_declared"
            ],
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
