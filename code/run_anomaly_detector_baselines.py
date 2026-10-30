#!/usr/bin/env python3
"""Evaluate ordinary anomaly detectors at a calibration-frozen 90% recall point."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from build_audit_feature_matrix import FEATURES
from common import ARTIFACTS, RECORDS, binary_metrics, load_jsonl, write_json


SEED = 20260730


def matrix(rows):
    return np.asarray(
        [[float(row["features"][feature]) for feature in FEATURES] for row in rows],
        dtype=float,
    )


def threshold_at_recall(scores, labels, recall=0.90):
    """Accept scores <= threshold; freeze the smallest threshold meeting recall."""

    positives = np.sort(scores[np.asarray(labels) == 1])
    index = min(len(positives) - 1, max(0, int(np.ceil(recall * len(positives))) - 1))
    return float(positives[index])


def metric_row(dataset, method, labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    accepted = scores <= threshold
    metrics = binary_metrics(labels.tolist(), accepted.tolist())
    anomaly_labels = 1 - labels
    try:
        auroc = float(roc_auc_score(anomaly_labels, scores))
        auprc = float(average_precision_score(anomaly_labels, scores))
    except ValueError:
        auroc = auprc = None
    return {
        "dataset": dataset,
        "method": method,
        "threshold": threshold,
        **metrics,
        "auroc_unsupported": auroc,
        "auprc_unsupported": auprc,
    }


def main():
    rows = load_jsonl(RECORDS / "audit_feature_matrix.jsonl")
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    calibration = by_dataset["convfinqa_seed"]
    x_cal = matrix(calibration)
    y_cal = np.asarray([row["reference_label"] for row in calibration], dtype=int)
    x_supported = x_cal[y_cal == 1]

    median = float(np.median(x_supported[:, FEATURES.index("min_relative_distance_to_operand")]))
    mad = float(np.median(np.abs(
        x_supported[:, FEATURES.index("min_relative_distance_to_operand")] - median
    ))) or 1e-12

    models = {
        "robust_mad_distance": None,
        "isolation_forest": make_pipeline(
            StandardScaler(),
            IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=1),
        ),
        "local_outlier_factor": make_pipeline(
            StandardScaler(),
            LocalOutlierFactor(n_neighbors=20, novelty=True),
        ),
        "one_class_svm": make_pipeline(StandardScaler(), OneClassSVM(nu=0.10, gamma="scale")),
        "logistic_regression_supervised_upper": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        ),
    }
    for name, model in models.items():
        if model is None:
            continue
        if name == "logistic_regression_supervised_upper":
            model.fit(x_cal, y_cal)
        else:
            model.fit(x_supported)

    def score(name, values):
        if name == "robust_mad_distance":
            scalar = values[:, FEATURES.index("min_relative_distance_to_operand")]
            return np.abs(scalar - median) / (1.4826 * mad)
        model = models[name]
        if name == "logistic_regression_supervised_upper":
            return 1.0 - model.predict_proba(values)[:, 1]
        return -model.decision_function(values)

    thresholds = {
        name: threshold_at_recall(score(name, x_cal), y_cal)
        for name in models
    }
    results = []
    for dataset in (
        "convfinqa_seed",
        "convfinqa_remainder",
        "multihiertt",
        "convfinqa_wrong_operation",
        "kramabench_human",
    ):
        subset = by_dataset[dataset]
        values = matrix(subset)
        labels = [row["reference_label"] for row in subset]
        for name in models:
            results.append(metric_row(
                dataset, name, labels, score(name, values), thresholds[name]
            ))

    payload = {
        "schema_version": 1,
        "seed": SEED,
        "features": list(FEATURES),
        "forbidden_features_used": [],
        "calibration": (
            "All models and thresholds use only the 60-pair ConvFinQA seed; "
            "thresholds are frozen at >=0.90 calibration support recall."
        ),
        "thresholds": thresholds,
        "results": results,
    }
    write_json(RECORDS / "anomaly_baseline_results.json", payload)

    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Dataset & Method & Support recall & Unsupported detection \\",
        r"\midrule",
    ]
    for row in results:
        recall = "--" if row["support_recall"] is None else f"{row['support_recall']:.3f}"
        detection = (
            "--"
            if row["unsupported_detection"] is None
            else f"{row['unsupported_detection']:.3f}"
        )
        lines.append(
            f"{row['dataset']} & {row['method'].replace('_', ' ')} & "
            f"{recall} ({row['support_recall_counts']}) & "
            f"{detection} "
            f"({row['unsupported_detection_counts']}) \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ARTIFACTS / "table_anomaly_baselines.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
