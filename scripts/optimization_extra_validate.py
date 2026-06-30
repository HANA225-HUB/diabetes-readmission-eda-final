"""Validate the focused extra ensemble against the previous final ensemble."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diabetes_readmission_project import choose_f2_threshold, patient_level_split, prepare_data  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"


def metric_row(name, y, probability, threshold=None):
    if threshold is None:
        threshold, _ = choose_f2_threshold(y, probability)
    prediction = probability >= threshold
    return {
        "model": name,
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "log_loss": log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "f2": fbeta_score(y, prediction, beta=2, zero_division=0),
    }


def topk_rows(name, y, probability):
    order = np.argsort(-probability)
    rows = []
    for fraction in [0.03, 0.05, 0.10, 0.20]:
        k = int(np.ceil(len(y) * fraction))
        selected = order[:k]
        tp = y[selected].sum()
        rows.append({
            "model": name,
            "capacity": fraction,
            "selected_n": k,
            "precision": tp / k,
            "recall_capture": tp / y.sum(),
            "lift": (tp / k) / y.mean(),
        })
    return rows


def operating_points(name, y_val, p_val, y_test, p_test):
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    rows = []
    for target_recall in [0.50, 0.60, 0.70, 0.80]:
        valid = np.flatnonzero(recall[:-1] >= target_recall)
        threshold = thresholds[valid[-1]]
        prediction = p_test >= threshold
        rows.append({
            "model": name,
            "validation_target_recall": target_recall,
            "threshold": threshold,
            "test_precision": precision_score(y_test, prediction, zero_division=0),
            "test_recall": recall_score(y_test, prediction, zero_division=0),
            "test_accuracy": accuracy_score(y_test, prediction),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, prediction),
            "selected_fraction": prediction.mean(),
        })
    return rows


def patient_bootstrap(y, groups, probabilities, repeats=1000):
    rng = np.random.default_rng(20260625)
    patients = np.unique(groups)
    locations = {patient: np.flatnonzero(groups == patient) for patient in patients}
    rows = []
    for repeat in range(repeats):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([locations[patient] for patient in sampled])
        if np.unique(y[idx]).size < 2:
            continue
        row = {"repeat": repeat + 1}
        order_cache = {}
        for name, probability in probabilities.items():
            row[f"pr_{name}"] = average_precision_score(y[idx], probability[idx])
            row[f"roc_{name}"] = roc_auc_score(y[idx], probability[idx])
            order_cache[name] = np.argsort(-probability[idx])
            k = int(np.ceil(len(idx) * 0.05))
            top = order_cache[name][:k]
            row[f"top5_precision_{name}"] = y[idx][top].sum() / k
        rows.append(row)
    return pd.DataFrame(rows)


def load_candidates():
    ensemble = np.load(RES / "ensemble_probabilities.npz")
    focused = np.load(RES / "focused_extra_probabilities.npz")
    candidates = {
        "previous_final": (ensemble["blend_val"], ensemble["blend_test"]),
        "final_tree_ensemble": (ensemble["blend_val"], ensemble["blend_test"]),
    }
    for key in focused.files:
        if key.startswith("val_"):
            name = key.removeprefix("val_")
            candidates[name] = (focused[key], focused[f"test_{name}"])
    return candidates, ensemble["y_val"], ensemble["y_test"]


def main():
    candidates, y_val, y_test = load_candidates()
    blend_table = pd.read_csv(TAB / "24_focused_extra_blends.csv")
    best = blend_table.sort_values("validation_pr_auc", ascending=False).iloc[0].to_dict()
    best["source_table"] = "24_focused_extra_blends.csv"
    local_path = TAB / "29_extra_local_blend_refinement.csv"
    if local_path.exists():
        local_best = pd.read_csv(local_path).sort_values("validation_pr_auc", ascending=False).iloc[0].to_dict()
        local_best["source_table"] = local_path.name
        if local_best["validation_pr_auc"] > best["validation_pr_auc"]:
            best = local_best
    weights = json.loads(best["weights"])

    val_blend = sum(weight * candidates[name][0] for name, weight in weights.items())
    test_blend = sum(weight * candidates[name][1] for name, weight in weights.items())
    candidates["focused_extra_ensemble"] = (val_blend, test_blend)

    metric_rows = []
    for name in ["previous_final", "focused_prefix_regularized_depth5", "focused_extra_ensemble"]:
        threshold, _ = choose_f2_threshold(y_val, candidates[name][0])
        metric_rows.append(metric_row(name, y_test, candidates[name][1], threshold))
    metrics = pd.DataFrame(metric_rows).sort_values("pr_auc", ascending=False)
    metrics.to_csv(TAB / "25_extra_final_metrics.csv", index=False, encoding="utf-8-sig")

    capacity_rows = []
    op_rows = []
    for name in ["previous_final", "focused_prefix_regularized_depth5", "focused_extra_ensemble"]:
        capacity_rows.extend(topk_rows(name, y_test, candidates[name][1]))
        op_rows.extend(operating_points(name, y_val, candidates[name][0], y_test, candidates[name][1]))
    pd.DataFrame(capacity_rows).to_csv(TAB / "26_extra_topk_capacity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(op_rows).to_csv(TAB / "27_extra_operating_points.csv", index=False, encoding="utf-8-sig")

    frame, _ = prepare_data()
    _, _, test_idx, _ = patient_level_split(frame)
    groups = frame.loc[test_idx, "patient_nbr"].to_numpy()
    bootstrap = patient_bootstrap(
        y_test, groups,
        {
            "previous_final": candidates["previous_final"][1],
            "focused_extra_ensemble": candidates["focused_extra_ensemble"][1],
        },
    )
    bootstrap.to_csv(TAB / "28_extra_bootstrap.csv", index=False, encoding="utf-8-sig")
    deltas = {}
    for metric in ["pr", "roc", "top5_precision"]:
        delta = bootstrap[f"{metric}_focused_extra_ensemble"] - bootstrap[f"{metric}_previous_final"]
        deltas[metric] = {
            "mean": float(delta.mean()),
            "ci95": [float(delta.quantile(0.025)), float(delta.quantile(0.975))],
            "positive_probability": float((delta > 0).mean()),
        }

    summary = {
        "selected_blend": best,
        "weights": weights,
        "metrics": metrics.to_dict(orient="records"),
        "bootstrap_delta_vs_previous_final": deltas,
    }
    (RES / "extra_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
