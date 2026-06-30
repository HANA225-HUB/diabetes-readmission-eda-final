"""Ablate encounter-id gap features from the strict longitudinal-history model.

The final strict-history model uses encounter_id only to order repeated
encounters. This script checks whether the numeric gap between consecutive
encounter IDs is responsible for the observed gain.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diabetes_readmission_project import choose_f2_threshold, patient_level_split, prepare_data  # noqa: E402
from optimization_extra_search import add_ultra_features  # noqa: E402
from optimization_history_refinement import blend_grid, fit_history_models, reconstruct_focused_extra  # noqa: E402
from optimization_research_pass import add_longitudinal_features  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"

GAP_COLUMNS = {
    "encounter_id_gap_from_previous",
    "log1p_encounter_id_gap_from_previous",
}


def metric_row(model: str, y_val, p_val, y_test, p_test):
    threshold, _ = choose_f2_threshold(y_val, p_val)
    predicted = p_test >= threshold
    return {
        "model": model,
        "roc_auc": roc_auc_score(y_test, p_test),
        "pr_auc": average_precision_score(y_test, p_test),
        "brier": brier_score_loss(y_test, p_test),
        "log_loss": log_loss(y_test, np.clip(p_test, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "accuracy": float((predicted == y_test).mean()),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "f2": fbeta_score(y_test, predicted, beta=2, zero_division=0),
    }


def topk_rows(model: str, y, p):
    rows = []
    prevalence = float(y.mean())
    order = np.argsort(-p)
    n = len(y)
    for fraction in [0.03, 0.05, 0.10, 0.20]:
        k = max(1, int(np.ceil(n * fraction)))
        chosen = order[:k]
        true_positive = int(y[chosen].sum())
        rows.append({
            "model": model,
            "capacity": fraction,
            "selected_n": k,
            "true_positive": true_positive,
            "false_positive": int(k - true_positive),
            "precision": true_positive / k,
            "recall_capture": true_positive / int(y.sum()),
            "lift": (true_positive / k) / prevalence,
        })
    return rows


def main():
    started = time.time()
    base_frame, _ = prepare_data()
    ultra_frame, ultra_num, ultra_cat = add_ultra_features(base_frame)
    train_idx, val_idx, test_idx, _ = patient_level_split(ultra_frame)
    y_val = ultra_frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = ultra_frame.loc[test_idx, "target_30d"].to_numpy()

    history_frame, history_num, history_cat = add_longitudinal_features(
        ultra_frame, include_target_history=False
    )
    no_gap_num = [column for column in history_num if column not in GAP_COLUMNS]
    removed = sorted(set(history_num) - set(no_gap_num))
    print(f"Removed history numeric features: {removed}", flush=True)

    model_rows, predictions = fit_history_models(
        history_frame,
        ultra_num,
        ultra_cat,
        no_gap_num,
        history_cat,
        train_idx,
        val_idx,
        test_idx,
        "strict_history_no_gap",
    )
    pd.DataFrame(model_rows).to_csv(
        TAB / "53_history_gap_ablation_models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    focused_val, focused_test = reconstruct_focused_extra()
    candidates = {"focused_extra_ensemble": (focused_val, focused_test), **predictions}
    blends = blend_grid(y_val, y_test, candidates, "strict_history_no_gap")
    blends.to_csv(TAB / "54_history_gap_ablation_blends.csv", index=False, encoding="utf-8-sig")
    best = blends.iloc[0]
    weights = json.loads(best["weights"])
    no_gap_val = sum(weights[name] * candidates[name][0] for name in weights)
    no_gap_test = sum(weights[name] * candidates[name][1] for name in weights)

    full = np.load(RES / "history_final_probabilities.npz")
    final_rows = [
        metric_row("focused_extra_ensemble", y_val, full["focused_extra_val"], y_test, full["focused_extra_test"]),
        metric_row(
            "strict_history_ensemble_full",
            y_val,
            full["strict_history_ensemble_val"],
            y_test,
            full["strict_history_ensemble_test"],
        ),
        metric_row("strict_history_no_gap_ensemble", y_val, no_gap_val, y_test, no_gap_test),
    ]
    pd.DataFrame(final_rows).to_csv(
        TAB / "55_history_gap_ablation_final_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    capacity_rows = []
    for name, probability in [
        ("focused_extra_ensemble", full["focused_extra_test"]),
        ("strict_history_ensemble_full", full["strict_history_ensemble_test"]),
        ("strict_history_no_gap_ensemble", no_gap_test),
    ]:
        capacity_rows.extend(topk_rows(name, y_test, probability))
    pd.DataFrame(capacity_rows).to_csv(
        TAB / "56_history_gap_ablation_topk.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        RES / "history_gap_ablation_probabilities.npz",
        y_val=y_val,
        y_test=y_test,
        strict_history_no_gap_val=no_gap_val,
        strict_history_no_gap_test=no_gap_test,
        **{f"val_{name}": value[0] for name, value in predictions.items()},
        **{f"test_{name}": value[1] for name, value in predictions.items()},
    )
    summary = {
        "runtime_seconds": time.time() - started,
        "removed_features": removed,
        "best_no_gap_blend": best.to_dict(),
        "final_metrics": final_rows,
    }
    (RES / "history_gap_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
