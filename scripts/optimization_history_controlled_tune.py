"""Controlled train-internal tuning for the strict history CatBoost model.

This is intentionally smaller and more auditable than the interrupted Optuna run:
each candidate is pre-specified, selected only on a patient-level internal split
inside the training set, and written to disk as soon as it finishes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bonus_experiments import catboost_frame  # noqa: E402
from diabetes_readmission_project import SEED, patient_level_split, prepare_data  # noqa: E402
from optimization_extra_search import BASE_CAT_PARAMS, add_ultra_features, feature_variant  # noqa: E402
from optimization_lab import patient_internal_split  # noqa: E402
from optimization_research_pass import add_longitudinal_features, unique  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"


def score(name, split, y, probability, **extra):
    return {
        "candidate": name,
        "split": split,
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "log_loss": log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7)),
        **extra,
    }


def candidate_configs():
    base = {
        **BASE_CAT_PARAMS,
        "random_seed": 3407,
        "thread_count": -1,
    }
    return [
        ("current_history", {
            **base, "iterations": 620, "depth": 5, "learning_rate": 0.038,
            "l2_leaf_reg": 24.0, "random_strength": 0.45, "bagging_temperature": 0.75,
        }),
        ("stronger_l2", {
            **base, "iterations": 720, "depth": 5, "learning_rate": 0.032,
            "l2_leaf_reg": 38.0, "random_strength": 0.60, "bagging_temperature": 0.95,
        }),
        ("lighter_l2", {
            **base, "iterations": 560, "depth": 5, "learning_rate": 0.045,
            "l2_leaf_reg": 16.0, "random_strength": 0.32, "bagging_temperature": 0.65,
        }),
        ("shallow_regularized", {
            **base, "iterations": 760, "depth": 4, "learning_rate": 0.035,
            "l2_leaf_reg": 30.0, "random_strength": 0.55, "bagging_temperature": 0.85,
        }),
        ("depth6_slow_regularized", {
            **base, "iterations": 560, "depth": 6, "learning_rate": 0.034,
            "l2_leaf_reg": 28.0, "random_strength": 0.45, "bagging_temperature": 0.85,
        }),
        ("depth5_feature_subsample", {
            **base, "iterations": 680, "depth": 5, "learning_rate": 0.036,
            "l2_leaf_reg": 24.0, "random_strength": 0.45, "bagging_temperature": 0.75,
            "rsm": 0.88,
        }),
    ]


def main():
    started = time.time()
    frame, _ = prepare_data()
    frame, ultra_num, ultra_cat = add_ultra_features(frame)
    frame, history_num, history_cat = add_longitudinal_features(frame, include_target_history=False)
    base_num, base_cat = feature_variant(frame, ultra_num, ultra_cat, "ultra_prefix_no_raw_diag")
    numeric = unique(base_num + history_num)
    categorical = unique(base_cat + history_cat)
    train_idx, val_idx, test_idx, _ = patient_level_split(frame)
    fit_idx, tune_idx = patient_internal_split(frame, train_idx)

    features = catboost_frame(frame, numeric, categorical)
    cat_positions = [features.columns.get_loc(col) for col in categorical]
    y_fit = frame.loc[fit_idx, "target_30d"].to_numpy()
    y_tune = frame.loc[tune_idx, "target_30d"].to_numpy()
    y_train = frame.loc[train_idx, "target_30d"].to_numpy()
    y_val = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = frame.loc[test_idx, "target_30d"].to_numpy()

    rows = []
    tune_predictions = {}
    for name, params in candidate_configs():
        model_started = time.time()
        model = CatBoostClassifier(**params)
        model.fit(
            features.loc[fit_idx],
            y_fit,
            cat_features=cat_positions,
            eval_set=(features.loc[tune_idx], y_tune),
            early_stopping_rounds=70,
            verbose=False,
        )
        probability = model.predict_proba(features.loc[tune_idx])[:, 1]
        best_iteration = int(model.get_best_iteration())
        row = score(
            name,
            "internal_tune",
            y_tune,
            probability,
            best_iteration=best_iteration,
            runtime_seconds=time.time() - model_started,
            **{f"param_{key}": value for key, value in params.items() if key not in {"verbose", "thread_count"}},
        )
        rows.append(row)
        tune_predictions[name] = probability
        pd.DataFrame(rows).sort_values("pr_auc", ascending=False).to_csv(
            TAB / "40_history_controlled_tuning_internal.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"{name}: internal AP={row['pr_auc']:.5f}, ROC={row['roc_auc']:.5f}, "
            f"iter={best_iteration}, {row['runtime_seconds']:.1f}s",
            flush=True,
        )

    internal = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    best_name = internal.iloc[0]["candidate"]
    best_params = dict(candidate_configs())[best_name]
    best_iterations = max(160, int(internal.iloc[0]["best_iteration"] * 1.15))
    final_params = {**best_params, "iterations": best_iterations}

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(features.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
    val_probability = final_model.predict_proba(features.loc[val_idx])[:, 1]
    test_probability = final_model.predict_proba(features.loc[test_idx])[:, 1]
    final_rows = [
        score(best_name, "validation", y_val, val_probability, selected_by_internal_tune=True),
        score(best_name, "test", y_test, test_probability, selected_by_internal_tune=True),
    ]
    pd.DataFrame(final_rows).to_csv(
        TAB / "41_history_controlled_tuned_final.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        RES / "history_controlled_tuned_probabilities.npz",
        y_val=y_val,
        y_test=y_test,
        val_controlled_tuned=val_probability,
        test_controlled_tuned=test_probability,
    )
    summary = {
        "runtime_seconds": time.time() - started,
        "selected_candidate": best_name,
        "selected_internal_pr_auc": float(internal.iloc[0]["pr_auc"]),
        "selected_internal_roc_auc": float(internal.iloc[0]["roc_auc"]),
        "selected_internal_best_iteration": int(internal.iloc[0]["best_iteration"]),
        "final_iterations": int(best_iterations),
        "locked_validation": final_rows[0],
        "locked_test": final_rows[1],
    }
    (RES / "history_controlled_tuning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
