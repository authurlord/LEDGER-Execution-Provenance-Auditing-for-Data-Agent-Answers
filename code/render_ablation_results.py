#!/usr/bin/env python3
"""Render generated LaTeX and a compact three-panel closure sensitivity figure."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

from common import ARTIFACTS, RECORDS


def main():
    payload = json.load((RECORDS / "closure_ablation_grid_rev2.json").open())
    rows = payload["grid"]
    selected = [
        row for row in rows
        if row["tol_rel"] == 1e-4
        and row["operations"] == ["sum", "diff", "ratio", "pctchg", "prod"]
        and row["restriction"] == "citation_restricted"
    ]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Operand source / depth & Support recall & Unsupported detection & Wrong-op false support & Runtime (s) \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['operand_source'].replace('_', ' ')}, d={row['max_depth']} & "
            f"{row['support_recall']:.3f} ({row['support_recall_counts']}) & "
            f"{row['unsupported_detection']:.3f} ({row['unsupported_detection_counts']}) & "
            f"{row['wrong_operation_false_support_rate']:.3f} & "
            f"{row['runtime_s']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ARTIFACTS / "table_ablation_rev2.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    colors = {
        "reached_operands": "#167D9A",
        "all_numbers_in_cited_files": "#E07A3F",
        "all_context_numbers": "#7A6BB7",
    }
    for source, color in colors.items():
        subset = sorted(
            (row for row in selected if row["operand_source"] == source),
            key=lambda row: row["max_depth"],
        )
        depth = [row["max_depth"] for row in subset]
        axes[0].plot(depth, [row["support_recall"] for row in subset], "-o", color=color, label=source)
        axes[1].plot(depth, [row["unsupported_detection"] for row in subset], "-o", color=color)
        axes[2].plot(depth, [row["wrong_operation_false_support_rate"] for row in subset], "-o", color=color)
    axes[0].set_title("Support recall")
    axes[1].set_title("Unsupported detection")
    axes[2].set_title("Wrong-operation false support")
    for axis in axes:
        axis.set_xlabel("Maximum depth")
        axis.set_xticks([1, 2, 3])
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Rate")
    fig.legend(loc="upper center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(ARTIFACTS / "fig_ablation_3panel.pdf", bbox_inches="tight")
    fig.savefig(ARTIFACTS / "fig_ablation_3panel.png", dpi=180, bbox_inches="tight")
    print(f"rendered {len(selected)} selected sensitivity cells")


if __name__ == "__main__":
    main()
