#!/usr/bin/env python3
"""Generate all revision tables and compact three-panel figures from records."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import ARTIFACTS, RECORDS


BLUE = "#167D9A"
ORANGE = "#E07A3F"
PURPLE = "#7A6BB7"
GRAY = "#68737D"


def read(path):
    return json.load(Path(path).open())


def write(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_generated(source, target):
    if Path(source).exists():
        shutil.copy2(source, target)


def table1(out):
    assertions = read(RECORDS / "submission_assertions_rev2.json")
    wiki = read(RECORDS / "wikisql_semantic_validation_summary.json")
    rows = [
        ("ConvFinQA remainder", assertions["observed"]["convfinqa"]["total_remainder"], "numeric closure"),
        ("MultiHiertt", assertions["observed"]["multihiertt"]["total"], "numeric closure"),
        ("KramaBench audit", assertions["observed"]["kramabench"]["total"], "Python/pandas replay"),
        ("DABstep audit", assertions["observed"]["dabstep"]["total"], "fixed-file replay"),
        ("WikiSQL dev pilot", wiki["executable_base_examples"], "SQL replay + semantics"),
    ]
    lines = [
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"Dataset/testbed & N & Audit surface \\",
        r"\midrule",
        *[f"{name} & {n} & {surface} \\\\" for name, n, surface in rows],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write(out / "table_1_datasets.tex", lines)


def table2(out):
    numeric = read(RECORDS / "numeric_support_baselines_convfinqa.json")["results"]
    anomaly = read(RECORDS / "anomaly_baseline_results.json")["results"]
    selected = []
    for row in numeric:
        if row["method"] in {
            "exact_operand", "single_operation", "all_numbers_in_cited_file",
            "operand_constrained_closure",
        }:
            selected.append((
                "numeric", row["method"], row["support_recall_counts"],
                row["unsupported_detection_counts"],
            ))
    for row in anomaly:
        if row["dataset"] == "convfinqa_remainder" and row["method"] in {
            "isolation_forest", "logistic_regression_supervised_upper",
        }:
            selected.append((
                "anomaly", row["method"], row["support_recall_counts"],
                row["unsupported_detection_counts"],
            ))
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Family & Method & Support recall & Unsupported detection \\",
        r"\midrule",
        *[
            f"{family} & {method.replace('_', ' ')} & {recall} & {detection} \\\\"
            for family, method, recall, detection in selected
        ],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write(out / "table_2_baselines.tex", lines)


def table6(out):
    ablation = read(RECORDS / "closure_ablation_grid_rev2.json")["grid"]
    runtime = read(RECORDS / "audit_runtime_summary.json")["components"]["total_audit"]
    selected = [
        row for row in ablation
        if row["operand_source"] == "reached_operands"
        and row["restriction"] == "citation_restricted"
        and row["tol_rel"] == 1e-4
        and row["operations"] == ["sum", "diff", "ratio", "pctchg", "prod"]
    ]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Setting & Support recall & Unsupported detection & Wrong-op false support \\",
        r"\midrule",
        *[
            f"Depth {row['max_depth']} & {row['support_recall_counts']} & "
            f"{row['unsupported_detection_counts']} & "
            f"{row['wrong_operation_false_support_counts']} \\\\"
            for row in selected
        ],
        r"\midrule",
        f"Total audit median/p95 (ms) & {runtime['median_ms']:.3f} & "
        f"{runtime['p95_ms']:.3f} & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write(out / "table_6_ablation_runtime.tex", lines)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def baseline_figure(out):
    numeric = read(RECORDS / "numeric_support_baselines_convfinqa.json")["results"]
    anomaly = [
        row for row in read(RECORDS / "anomaly_baseline_results.json")["results"]
        if row["dataset"] == "convfinqa_remainder"
    ]
    with (ARTIFACTS / "table_trace_rule_baselines.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        trace = [
            row for row in csv.DictReader(handle)
            if row["dataset"] == "kramabench_human"
        ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    for row in numeric:
        if row["method"] == "gold_program_execution_oracle":
            continue
        axes[0].scatter(row["support_recall"], row["unsupported_detection"], color=BLUE)
        axes[0].annotate(row["method"].replace("_", " "), (
            row["support_recall"], row["unsupported_detection"]
        ), fontsize=6)
    for row in anomaly:
        axes[1].scatter(row["support_recall"], row["unsupported_detection"], color=ORANGE)
    for row in trace:
        recall = row["support_recall"]
        detection = row["unsupported_detection"]
        if recall and detection:
            axes[2].scatter(float(recall), float(detection), color=PURPLE)
    for axis, title in zip(
        axes, ("Numeric support", "Anomaly transfer", "Human trace rules")
    ):
        axis.set_title(title)
        axis.set_xlabel("Support recall")
        axis.set_ylabel("Unsupported detection")
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, out, "fig_3_baseline_results")


def governance_figure(out):
    policies = read(RECORDS / "delivery_policy_results.json")["results"]
    human = [row for row in policies if row["dataset"] == "kramabench_human_clear"]
    live = read(RECORDS / "live_six_domain_summary.json")["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for row in human:
        unsafe = (
            row["human_unsupported_delivered"] / row["n_delivered"]
            if row["n_delivered"] else 0
        )
        axes[0].scatter(row["coverage"], unsafe, color=BLUE)
        if row["policy"] in {"deliver_all", "deliver_if_replay_certificate"}:
            axes[0].annotate(row["policy"].replace("deliver_if_", ""), (
                row["coverage"], unsafe
            ), fontsize=6)
        if row["official_score_on_delivered"] is not None:
            axes[1].scatter(
                row["coverage"], row["official_score_on_delivered"], color=ORANGE
            )
    domains = [row["domain"] for row in live if row["replay_applicable_n"]]
    certificate = [row["certificate"] for row in live if row["replay_applicable_n"]]
    failure = [row["explicit_failure"] for row in live if row["replay_applicable_n"]]
    unknown = [row["unknown"] for row in live if row["replay_applicable_n"]]
    axes[2].bar(domains, certificate, color=BLUE, label="certificate")
    axes[2].bar(domains, failure, bottom=certificate, color=ORANGE, label="failure")
    bottom = [a + b for a, b in zip(certificate, failure)]
    axes[2].bar(domains, unknown, bottom=bottom, color=GRAY, label="unknown")
    axes[0].set(title="Unsafe delivery", xlabel="Coverage", ylabel="Human-unsupported fraction")
    axes[1].set(title="Delivered utility", xlabel="Coverage", ylabel="Official score")
    axes[2].set(title="Live audit distribution", ylabel="Episodes")
    axes[2].tick_params(axis="x", labelrotation=30, labelsize=7)
    axes[2].legend(fontsize=6, frameon=False)
    for axis in axes[:2]:
        axis.set_xlim(-0.03, 1.03)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, out, "fig_4_live_governance")
    shutil.copy2(out / "fig_4_live_governance.pdf", out / "fig_delivery_tradeoff.pdf")


def semantic_figure(out):
    wiki = read(RECORDS / "wikisql_semantic_validation_summary.json")
    same = [
        row for row in wiki["results"]
        if row["perturbation_class"] == "same_original_output_counterfactual"
    ]
    ablation = read(RECORDS / "closure_ablation_grid_rev2.json")["grid"]
    depth = sorted([
        row for row in ablation
        if row["operand_source"] == "reached_operands"
        and row["restriction"] == "citation_restricted"
        and row["tol_rel"] == 1e-4
        and row["operations"] == ["sum", "diff", "ratio", "pctchg", "prod"]
    ], key=lambda row: row["max_depth"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    labels = [row["validator"].replace("_", " ") for row in same]
    rates = [row["detection_rate"] for row in same]
    axes[0].barh(labels, rates, color=[GRAY, PURPLE, PURPLE, PURPLE, BLUE, ORANGE])
    axes[0].set(title="Same-output semantic drift", xlabel="Detection rate", xlim=(0, 1.03))
    axes[1].plot(
        [row["max_depth"] for row in depth],
        [row["support_recall"] for row in depth],
        "-o", color=BLUE, label="support recall",
    )
    axes[1].plot(
        [row["max_depth"] for row in depth],
        [row["unsupported_detection"] for row in depth],
        "-o", color=ORANGE, label="unsupported detection",
    )
    axes[1].set(title="Depth tradeoff", xlabel="Maximum depth", ylim=(-0.03, 1.03))
    axes[1].legend(fontsize=6, frameon=False)
    axes[2].plot(
        [row["max_depth"] for row in depth],
        [row["wrong_operation_false_support_rate"] for row in depth],
        "-o", color=PURPLE,
    )
    axes[2].set(
        title="Semantic boundary",
        xlabel="Maximum depth",
        ylabel="Wrong-operation false support",
        ylim=(-0.03, 1.03),
    )
    for axis in axes[1:]:
        axis.set_xticks([1, 2, 3])
        axis.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, out, "fig_5_semantic_ablation")


def architecture_figure(out):
    fig, axis = plt.subplots(figsize=(8.0, 2.2))
    axis.axis("off")
    boxes = [
        (0.02, "Completed\nagent episode", GRAY),
        (0.27, "Trace + declared\nevidence", BLUE),
        (0.52, "Restricted replay /\noperand closure", PURPLE),
        (0.77, "Deliver · abstain ·\nescalate", ORANGE),
    ]
    for x, label, color in boxes:
        patch = FancyBboxPatch(
            (x, 0.3), 0.19, 0.38, boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="none", alpha=0.9,
        )
        axis.add_patch(patch)
        axis.text(x + 0.095, 0.49, label, ha="center", va="center", color="white", fontsize=9)
    for x in (0.21, 0.46, 0.71):
        axis.annotate("", xy=(x + 0.05, 0.49), xytext=(x, 0.49),
                      arrowprops={"arrowstyle": "->", "color": "#303840", "lw": 1.5})
    axis.text(
        0.615, 0.12, "Optional semantic layer: SQL structure + differential execution",
        ha="center", fontsize=8, color=PURPLE,
    )
    fig.tight_layout()
    save(fig, out, "fig_2_three_phase_architecture")


def teaser_figure(out):
    numeric = read(RECORDS / "numeric_support_baselines_convfinqa.json")["results"]
    constrained = next(row for row in numeric if row["method"] == "operand_constrained_closure")
    free = next(row for row in numeric if row["method"] == "all_numbers_in_cited_file")
    policies = read(RECORDS / "delivery_policy_results.json")["results"]
    human = {
        row["policy"]: row for row in policies
        if row["dataset"] == "kramabench_human_clear"
    }
    wiki = read(RECORDS / "wikisql_semantic_validation_summary.json")["results"]
    same = {
        row["validator"]: row for row in wiki
        if row["perturbation_class"] == "same_original_output_counterfactual"
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    axes[0].bar(
        ["free", "restricted"],
        [free["unsupported_detection"], constrained["unsupported_detection"]],
        color=[GRAY, BLUE],
    )
    axes[0].set(title="Accidental-collision control", ylabel="Unsupported detection", ylim=(0, 1))
    axes[1].bar(
        ["deliver all", "replay policy"],
        [
            human["deliver_all"]["human_unsupported_delivered"],
            human["deliver_if_replay_certificate"]["human_unsupported_delivered"],
        ],
        color=[GRAY, ORANGE],
    )
    axes[1].set(title="Human escalation", ylabel="Unsupported delivered")
    axes[2].bar(
        ["output\nequality", "SQL\nstructure"],
        [
            same["original_output_equality"]["detection_rate"],
            same["normalized_sql_exact_match"]["detection_rate"],
        ],
        color=[GRAY, PURPLE],
    )
    axes[2].set(title="Same-output semantic drift", ylabel="Detection rate", ylim=(0, 1))
    fig.tight_layout()
    save(fig, out, "fig_1_three_panel_teaser")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(RECORDS))
    parser.add_argument("--out", default=str(ARTIFACTS))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table1(out)
    table2(out)
    copy_generated(ARTIFACTS / "table_live_six_domain.tex", out / "table_3_live_agent_audit.tex")
    copy_generated(ARTIFACTS / "table_delivery_policies.tex", out / "table_4_delivery_policies.tex")
    copy_generated(
        ARTIFACTS / "table_wikisql_semantic_validation.tex",
        out / "table_5_wikisql_semantic_validation.tex",
    )
    table6(out)
    teaser_figure(out)
    architecture_figure(out)
    baseline_figure(out)
    governance_figure(out)
    semantic_figure(out)
    print(json.dumps({
        "out": str(out),
        "tables": len(list(out.glob("table_*.tex"))),
        "pdf_figures": len(list(out.glob("fig_*.pdf"))),
        "png_figures": len(list(out.glob("fig_*.png"))),
    }, indent=2))


if __name__ == "__main__":
    main()
