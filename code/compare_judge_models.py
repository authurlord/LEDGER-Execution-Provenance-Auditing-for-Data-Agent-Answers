#!/usr/bin/env python3
"""Compare fixed-design evidence-aware judge runs without model cherry-picking."""

from __future__ import annotations

from common import ARTIFACTS, DOCS, RECORDS, load_json, write_csv, write_json


def aggregate(summary):
    datasets = summary["datasets"]
    return {
        "majority_correct": sum(row["majority_correct"] for row in datasets.values()),
        "n_clear": sum(row["n_clear"] for row in datasets.values()),
        "repeat_flip_items": sum(row["repeat_flip_items"] for row in datasets.values()),
        "n_items": sum(row["n_items"] for row in datasets.values()),
        "parse_failures": summary["parse_failures"],
        "calls": summary["calls"],
    }


def main():
    summaries = [
        load_json(RECORDS / "evidence_aware_judge_summary.json"),
        load_json(RECORDS / "evidence_aware_judge_summary_qwen36_27b_nvfp4.json"),
    ]
    models = {summary["model"]: summary for summary in summaries}
    baseline = models["qwen3.6-35b"]
    diagnostic = models["qwen3.6-27b-nvfp4"]
    dataset_rows = []
    for dataset in sorted(baseline["datasets"]):
        left = baseline["datasets"][dataset]
        right = diagnostic["datasets"][dataset]
        dataset_rows.append({
            "dataset": dataset,
            "n_clear": left["n_clear"],
            "qwen3.6-35b_correct": left["majority_correct"],
            "qwen3.6-35b_accuracy": left["majority_accuracy"],
            "qwen3.6-35b_repeat_flip_items": left["repeat_flip_items"],
            "qwen3.6-27b-nvfp4_correct": right["majority_correct"],
            "qwen3.6-27b-nvfp4_accuracy": right["majority_accuracy"],
            "qwen3.6-27b-nvfp4_repeat_flip_items": right["repeat_flip_items"],
            "accuracy_delta_27b_minus_35b": (
                right["majority_accuracy"] - left["majority_accuracy"]
            ),
        })
    payload = {
        "schema_version": 1,
        "design": {
            "inputs": "same frozen 290 items",
            "temperature": 0,
            "repeats": 3,
            "max_tokens": 384,
            "response_format": "json_object",
            "changed_factor": "model",
        },
        "models": {
            summary["model"]: {
                **aggregate(summary),
                "tokens_in": summary["tokens_in"],
                "tokens_out": summary["tokens_out"],
            }
            for summary in summaries
        },
        "datasets": dataset_rows,
        "primary_model_retained": "qwen3.6-35b",
        "conclusion": (
            "The 27B NVFP4 diagnostic is marginally better on ConvFinQA but worse "
            "on the human boundary and WikiSQL, with more repeat flips and parse "
            "failures. The observed trend is not explained by a simple 35B model "
            "failure, so the 35B run remains primary and both complete runs are retained."
        ),
    }
    write_json(RECORDS / "evidence_aware_judge_model_comparison.json", payload)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_csv(ARTIFACTS / "table_judge_model_comparison.csv", dataset_rows)
    latex = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & $n$ & 35B correct & 27B correct & 35B flips & 27B flips \\",
        r"\midrule",
    ]
    for row in dataset_rows:
        name = row["dataset"].replace("_", r"\_")
        latex.append(
            f"{name} & {row['n_clear']} & {row['qwen3.6-35b_correct']} & "
            f"{row['qwen3.6-27b-nvfp4_correct']} & "
            f"{row['qwen3.6-35b_repeat_flip_items']} & "
            f"{row['qwen3.6-27b-nvfp4_repeat_flip_items']} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}", ""])
    (ARTIFACTS / "table_judge_model_comparison.tex").write_text(
        "\n".join(latex), encoding="utf-8"
    )

    lines = [
        "# Fixed-design local-judge model diagnostic",
        "",
        "The two runs use the same 290 items, prompt, temperature 0, three repeats, "
        "384-token cap, and JSON response format. Only the model changed.",
        "",
        "| dataset | clear n | 35B correct | 27B NVFP4 correct | 35B flips | 27B flips |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['n_clear']} | "
            f"{row['qwen3.6-35b_correct']} | {row['qwen3.6-27b-nvfp4_correct']} | "
            f"{row['qwen3.6-35b_repeat_flip_items']} | "
            f"{row['qwen3.6-27b-nvfp4_repeat_flip_items']} |"
        )
    lines.extend([
        "",
        f"- 35B parse failures: {baseline['parse_failures']}/{baseline['calls']}.",
        f"- 27B NVFP4 parse failures: {diagnostic['parse_failures']}/{diagnostic['calls']}.",
        "",
        payload["conclusion"],
        "",
    ])
    (DOCS / "EVIDENCE_AWARE_JUDGE_MODEL_COMPARISON.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
