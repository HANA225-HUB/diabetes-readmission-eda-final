"""Train-only hyperparameter tuning for the strict longitudinal-history model."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bonus_experiments import catboost_frame  # noqa: E402
from diabetes_readmission_project import SEED, patient_level_split, prepare_data  # noqa: E402
from optimization_extra_search import add_ultra_features, feature_variant  # noqa: E402
from optimization_lab import patient_internal_split  # noqa: E402
from optimization_research_pass import add_longitudinal_features, unique  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def trial_table(study):
    rows = []
    for trial in study.trials:
        rows.append({
            "trial": trial.number,
            "internal_pr_auc": trial.value,
            "internal_roc_auc": trial.user_attrs.get("roc_auc"),
            "best_iteration": trial.user_attrs.get("best_iteration"),
            **trial.params,
        })
    return pd.DataFrame(rows).sort_values("internal_pr_auc", ascending=False)


def main():
    started = time.time()
    base_frame, _ = prepare_data()
    ultra_frame, ultra_num, ultra_cat = add_ultra_features(base_frame)
    history_frame, history_num, history_cat = add_longitudinal_features(
        ultra_frame, include_target_history=False
    )
    base_num, base_cat = feature_variant(history_frame, ultra_num, ultra_cat, "ultra_prefix_no_raw_diag")
    numeric = unique(base_num + history_num)
    categorical = unique(base_cat + history_cat)
    train_idx, val_idx, test_idx, _ = patient_level_split(history_frame)
    fit_idx, tune_idx = patient_internal_split(history_frame, train_idx)
    features = catboost_frame(history_frame, numeric, categorical)
    cat_positions = [features.columns.get_loc(col) for col in categorical]
    y_fit = history_frame.loc[fit_idx, "target_30d"].to_numpy()
    y_tune = history_frame.loc[tune_idx, "target_30d"].to_numpy()
    y_train = history_frame.loc[train_idx, "target_30d"].to_numpy()
    y_val = history_frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = history_frame.loc[test_idx, "target_30d"].to_numpy()

    def objective(trial):
        params = {
            "iterations": 1000,
            "depth": trial.suggest_int("depth", 4, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.020, 0.070, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 8.0, 60.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.08, 1.20, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.20, 1.20),
            "rsm": trial.suggest_float("rsm", 0.75, 1.0),
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "random_seed": 3407,
            "allow_writing_files": False,
            "verbose": False,
            "thread_count": -1,
        }
        model = CatBoostClassifier(**params)
        model.fit(
            features.loc[fit_idx], y_fit,
            cat_features=cat_positions,
            eval_set=(features.loc[tune_idx], y_tune),
            early_stopping_rounds=80,
            verbose=False,
        )
        probability = model.predict_proba(features.loc[tune_idx])[:, 1]
        trial.set_user_attr("best_iteration", int(model.get_best_iteration()))
        trial.set_user_attr("roc_auc", float(roc_auc_score(y_tune, probability)))
        return average_precision_score(y_tune, probability)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=20260626),
    )
    study.optimize(objective, n_trials=14, show_progress_bar=False)
    trials = trial_table(study)
    trials.to_csv(TAB / "40_history_internal_tuning_trials.csv", index=False, encoding="utf-8-sig")

    best_params = dict(study.best_params)
    best_iteration = max(120, int(study.best_trial.user_attrs["best_iteration"] * 1.15))
    final = CatBoostClassifier(
        iterations=best_iteration,
        **best_params,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=3407,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    final.fit(features.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
    val_probability = final.predict_proba(features.loc[val_idx])[:, 1]
    test_probability = final.predict_proba(features.loc[test_idx])[:, 1]
    rows = [
        score("strict_history_tuned_catboost", "validation", y_val, val_probability),
        score("strict_history_tuned_catboost", "test", y_test, test_probability),
    ]
    pd.DataFrame(rows).to_csv(TAB / "41_history_tuned_model.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        RES / "history_tuned_probabilities.npz",
        y_val=y_val,
        y_test=y_test,
        val_strict_history_tuned_catboost=val_probability,
        test_strict_history_tuned_catboost=test_probability,
    )
    summary = {
        "runtime_seconds": time.time() - started,
        "best_internal_pr_auc": float(study.best_value),
        "best_params": best_params,
        "best_internal_iteration": int(study.best_trial.user_attrs["best_iteration"]),
        "final_iterations": int(best_iteration),
        "locked_validation": rows[0],
        "locked_test": rows[1],
    }
    (RES / "history_tuning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
