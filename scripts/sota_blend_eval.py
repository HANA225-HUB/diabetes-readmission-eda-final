"""Validation-selected blending and patient bootstrap for SOTA predictions."""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diabetes_readmission_project import patient_level_split, prepare_data  # noqa: E402


RES = HERE / "optimization" / "results"
TAB = HERE / "optimization" / "tables"


def score(y, probability):
    return {
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, probability),
    }


def patient_bootstrap(y, groups, probabilities, repeats=1000):
    rng = np.random.default_rng(20260625)
    patients = np.unique(groups)
    locations = {patient: np.flatnonzero(groups == patient) for patient in patients}
    rows = []
    for repeat in range(repeats):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([locations[patient] for patient in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        row = {"repeat": repeat + 1}
        for name, probability in probabilities.items():
            row[f"pr_{name}"] = average_precision_score(y[idx], probability[idx])
            row[f"roc_{name}"] = roc_auc_score(y[idx], probability[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    sota = np.load(RES / "sota_probabilities.npz")
    baseline = np.load(RES / "ensemble_probabilities.npz")
    y_val, y_test = sota["y_val"], sota["y_test"]
    base_val, base_test = baseline["blend_val"], baseline["blend_test"]
    models = sorted(key.removeprefix("val_") for key in sota.files if key.startswith("val_"))

    rows = []
    candidates = {"tree_ensemble": (base_val, base_test)}
    for model in models:
        candidates[model] = (sota[f"val_{model}"], sota[f"test_{model}"])
    for name, (val_probability, test_probability) in candidates.items():
        rows.append({"candidate": name, "split": "validation", **score(y_val, val_probability)})
        rows.append({"candidate": name, "split": "test", **score(y_test, test_probability)})

    blend_rows = []
    for model in models:
        val_new, test_new = candidates[model]
        for weight in np.arange(0.0, 0.501, 0.01):
            val_probability = (1 - weight) * base_val + weight * val_new
            test_probability = (1 - weight) * base_test + weight * test_new
            blend_rows.append({"model": model, "new_model_weight": weight,
                               "validation_pr_auc": average_precision_score(y_val, val_probability),
                               "validation_roc_auc": roc_auc_score(y_val, val_probability),
                               "test_pr_auc": average_precision_score(y_test, test_probability),
                               "test_roc_auc": roc_auc_score(y_test, test_probability),
                               "test_brier": brier_score_loss(y_test, test_probability)})

    blend_table = pd.DataFrame(blend_rows)
    selected_rows = []
    selected_probabilities = {}
    for model in models:
        selected = blend_table[blend_table["model"] == model].sort_values("validation_pr_auc", ascending=False).iloc[0]
        selected_rows.append(selected.to_dict())
        weight = selected["new_model_weight"]
        selected_probabilities[f"tree_plus_{model}"] = (1 - weight) * base_test + weight * candidates[model][1]

    # Coarse simplex search over the tree model and all completed neural models.
    all_names = ["tree_ensemble"] + models
    best = None
    if len(all_names) <= 4:
        grid = np.arange(0.0, 1.001, 0.05)
        for weights in product(grid, repeat=len(all_names)):
            if not np.isclose(sum(weights), 1.0):
                continue
            probability = sum(weight * candidates[name][0] for name, weight in zip(all_names, weights))
            value = average_precision_score(y_val, probability)
            if best is None or value > best["validation_pr_auc"]:
                best = {"validation_pr_auc": value, "weights": dict(zip(all_names, weights))}
        best_val = sum(best["weights"][name] * candidates[name][0] for name in all_names)
        best_test = sum(best["weights"][name] * candidates[name][1] for name in all_names)
        best.update({"validation": score(y_val, best_val), "test": score(y_test, best_test)})
        selected_probabilities["sota_simplex"] = best_test

    frame, _ = prepare_data()
    _, _, test_idx, _ = patient_level_split(frame)
    groups = frame.loc[test_idx, "patient_nbr"].to_numpy()
    bootstrap_probabilities = {"tree_ensemble": base_test, **selected_probabilities}
    bootstrap = patient_bootstrap(y_test, groups, bootstrap_probabilities)
    bootstrap.to_csv(TAB / "20_sota_bootstrap.csv", index=False, encoding="utf-8-sig")
    deltas = {}
    for name in selected_probabilities:
        for metric in ["pr", "roc"]:
            delta = bootstrap[f"{metric}_{name}"] - bootstrap[f"{metric}_tree_ensemble"]
            deltas[f"{metric}_{name}_vs_tree"] = {
                "mean": float(delta.mean()),
                "ci95": [float(delta.quantile(0.025)), float(delta.quantile(0.975))],
                "positive_probability": float((delta > 0).mean()),
            }

    pd.DataFrame(rows).to_csv(TAB / "18_sota_direct_comparison.csv", index=False, encoding="utf-8-sig")
    blend_table.to_csv(TAB / "19_sota_blend_search.csv", index=False, encoding="utf-8-sig")
    summary = {"models": models, "selected_pairwise_blends": selected_rows, "simplex_blend": best,
               "bootstrap_deltas": deltas}
    (RES / "sota_blend_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
