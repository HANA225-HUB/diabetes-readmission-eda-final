"""Refine longitudinal-history models and separate strict vs sensitivity settings."""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bonus_experiments import catboost_frame  # noqa: E402
from diabetes_readmission_project import patient_level_split, prepare_data  # noqa: E402
from optimization_extra_search import BASE_CAT_PARAMS, add_ultra_features, feature_variant  # noqa: E402
from optimization_research_pass import add_longitudinal_features, unique  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"


def score(name, split, y, probability, **extra):
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "candidate": name,
        "split": split,
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "log_loss": log_loss(y, clipped),
        **extra,
    }


def cat_configs():
    base_no_bagging = {k: v for k, v in BASE_CAT_PARAMS.items() if k != "bagging_temperature"}
    return [
        ("prefix_regularized_history", "ultra_prefix_no_raw_diag", {
            **BASE_CAT_PARAMS, "iterations": 620, "depth": 5, "learning_rate": 0.038,
            "l2_leaf_reg": 24.0, "random_strength": 0.45, "bagging_temperature": 0.75,
            "random_seed": 3407,
        }),
        ("prefix_slow_history", "ultra_prefix_no_raw_diag", {
            **BASE_CAT_PARAMS, "iterations": 520, "depth": 6, "learning_rate": 0.046,
            "l2_leaf_reg": 9.0, "random_strength": 0.25, "bagging_temperature": 0.55,
            "random_seed": 3407,
        }),
        ("full_mvs_history", "ultra_full", {
            **base_no_bagging, "iterations": 520, "depth": 6, "learning_rate": 0.050,
            "l2_leaf_reg": 11.0, "random_strength": 0.20, "bootstrap_type": "MVS",
            "subsample": 0.88, "random_seed": 3407,
        }),
        ("full_bernoulli_history", "ultra_full", {
            **base_no_bagging, "iterations": 520, "depth": 6, "learning_rate": 0.050,
            "l2_leaf_reg": 10.5, "random_strength": 0.18, "bootstrap_type": "Bernoulli",
            "subsample": 0.82, "random_seed": 3407,
        }),
    ]


def fit_history_models(frame, ultra_num, ultra_cat, history_num, history_cat, train_idx, val_idx, test_idx, label):
    y_train = frame.loc[train_idx, "target_30d"].to_numpy()
    y_val = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = frame.loc[test_idx, "target_30d"].to_numpy()
    rows = []
    predictions = {}
    feature_cache = {}
    for config_name, variant, params in cat_configs():
        if variant not in feature_cache:
            base_num, base_cat = feature_variant(frame, ultra_num, ultra_cat, variant)
            numeric = unique(base_num + history_num)
            categorical = unique(base_cat + history_cat)
            features = catboost_frame(frame, numeric, categorical)
            cat_positions = [features.columns.get_loc(col) for col in categorical]
            feature_cache[variant] = (features, cat_positions, len(numeric), len(categorical))
        features, cat_positions, n_num, n_cat = feature_cache[variant]
        name = f"{label}_{config_name}"
        started = time.time()
        model = CatBoostClassifier(**params)
        model.fit(features.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
        runtime = time.time() - started
        val_probability = model.predict_proba(features.loc[val_idx])[:, 1]
        test_probability = model.predict_proba(features.loc[test_idx])[:, 1]
        predictions[name] = (val_probability, test_probability)
        rows.append(score(
            name, "validation", y_val, val_probability,
            history_setting=label, variant=variant, config=config_name,
            runtime_seconds=runtime, numeric_features=n_num, categorical_features=n_cat,
        ))
        rows.append(score(
            name, "test", y_test, test_probability,
            history_setting=label, variant=variant, config=config_name,
            runtime_seconds=runtime, numeric_features=n_num, categorical_features=n_cat,
        ))
        print(f"{name}: val AP={rows[-2]['pr_auc']:.5f}, test AP={rows[-1]['pr_auc']:.5f}", flush=True)
    return rows, predictions


def reconstruct_focused_extra():
    ensemble = np.load(RES / "ensemble_probabilities.npz")
    focused = np.load(RES / "focused_extra_probabilities.npz")
    weights = {
        "focused_prefix_regularized_depth5": 0.34,
        "final_tree_ensemble": 0.21,
        "focused_prefix_slow_depth6": 0.19,
        "focused_full_mvs_depth6": 0.26,
    }
    parts = {
        "focused_prefix_regularized_depth5": (
            focused["val_focused_prefix_regularized_depth5"],
            focused["test_focused_prefix_regularized_depth5"],
        ),
        "final_tree_ensemble": (ensemble["blend_val"], ensemble["blend_test"]),
        "focused_prefix_slow_depth6": (
            focused["val_focused_prefix_slow_depth6"],
            focused["test_focused_prefix_slow_depth6"],
        ),
        "focused_full_mvs_depth6": (
            focused["val_focused_full_mvs_depth6"],
            focused["test_focused_full_mvs_depth6"],
        ),
    }
    val = sum(weights[name] * parts[name][0] for name in weights)
    test = sum(weights[name] * parts[name][1] for name in weights)
    return val, test


def blend_grid(y_val, y_test, candidates, label):
    validation_scores = {
        name: average_precision_score(y_val, val)
        for name, (val, _) in candidates.items()
    }
    top_names = sorted(validation_scores, key=validation_scores.get, reverse=True)[:5]
    rows = []
    step_count = 50

    def compositions(total, size):
        if size == 1:
            yield [total]
            return
        for value in range(total + 1):
            for rest in compositions(total - value, size - 1):
                yield [value] + rest

    for size in [2, 3, 4]:
        active = top_names[:size]
        for units in compositions(step_count, size):
            weights = [unit / step_count for unit in units]
            val_probability = sum(weight * candidates[name][0] for name, weight in zip(active, weights))
            test_probability = sum(weight * candidates[name][1] for name, weight in zip(active, weights))
            rows.append({
                "blend_setting": label,
                "blend_size": size,
                "active_candidates": "+".join(active),
                "weights": json.dumps({name: float(weight) for name, weight in zip(active, weights)}),
                "validation_pr_auc": average_precision_score(y_val, val_probability),
                "validation_roc_auc": roc_auc_score(y_val, val_probability),
                "test_pr_auc": average_precision_score(y_test, test_probability),
                "test_roc_auc": roc_auc_score(y_test, test_probability),
                "test_brier": brier_score_loss(y_test, test_probability),
            })
    return pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False)


def main():
    started_all = time.time()
    base_frame, _ = prepare_data()
    ultra_frame, ultra_num, ultra_cat = add_ultra_features(base_frame)
    train_idx, val_idx, test_idx, _ = patient_level_split(ultra_frame)
    y_val = ultra_frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = ultra_frame.loc[test_idx, "target_30d"].to_numpy()

    print("Fitting strict history models", flush=True)
    strict_frame, strict_hist_num, strict_hist_cat = add_longitudinal_features(
        ultra_frame, include_target_history=False
    )
    strict_rows, strict_predictions = fit_history_models(
        strict_frame, ultra_num, ultra_cat, strict_hist_num, strict_hist_cat,
        train_idx, val_idx, test_idx, "strict_history",
    )

    print("Fitting target-history sensitivity models", flush=True)
    sensitivity_frame, sens_hist_num, sens_hist_cat = add_longitudinal_features(
        ultra_frame, include_target_history=True
    )
    sensitivity_rows, sensitivity_predictions = fit_history_models(
        sensitivity_frame, ultra_num, ultra_cat, sens_hist_num, sens_hist_cat,
        train_idx, val_idx, test_idx, "target_history_sensitivity",
    )

    model_table = pd.DataFrame(strict_rows + sensitivity_rows).sort_values(
        ["split", "pr_auc"], ascending=[True, False]
    )
    model_table.to_csv(TAB / "32_history_refinement_models.csv", index=False, encoding="utf-8-sig")

    focused_extra_val, focused_extra_test = reconstruct_focused_extra()
    research = np.load(RES / "research_hypothesis_probabilities.npz")
    strict_candidates = {
        "focused_extra_ensemble": (focused_extra_val, focused_extra_test),
        **strict_predictions,
    }
    if "val_research_history_without_target_labels" in research.files:
        strict_candidates["research_history_without_target_labels"] = (
            research["val_research_history_without_target_labels"],
            research["test_research_history_without_target_labels"],
        )
    sensitivity_candidates = {
        **strict_candidates,
        **sensitivity_predictions,
    }
    if "val_research_longitudinal_target_history_sensitivity" in research.files:
        sensitivity_candidates["research_longitudinal_target_history_sensitivity"] = (
            research["val_research_longitudinal_target_history_sensitivity"],
            research["test_research_longitudinal_target_history_sensitivity"],
        )
    if "val_research_full_mvs_target_history_sensitivity" in research.files:
        sensitivity_candidates["research_full_mvs_target_history_sensitivity"] = (
            research["val_research_full_mvs_target_history_sensitivity"],
            research["test_research_full_mvs_target_history_sensitivity"],
        )

    np.savez_compressed(
        RES / "history_refinement_probabilities.npz",
        y_val=y_val,
        y_test=y_test,
        focused_extra_val=focused_extra_val,
        focused_extra_test=focused_extra_test,
        **{f"val_{name}": value[0] for name, value in {**strict_predictions, **sensitivity_predictions}.items()},
        **{f"test_{name}": value[1] for name, value in {**strict_predictions, **sensitivity_predictions}.items()},
    )

    strict_blends = blend_grid(y_val, y_test, strict_candidates, "strict_no_target_history")
    sensitivity_blends = blend_grid(y_val, y_test, sensitivity_candidates, "target_history_sensitivity")
    blend_table = pd.concat([strict_blends, sensitivity_blends], ignore_index=True)
    blend_table.to_csv(TAB / "33_history_refinement_blends.csv", index=False, encoding="utf-8-sig")
    summary = {
        "runtime_seconds": time.time() - started_all,
        "best_strict_model": (
            model_table[
                (model_table["split"] == "validation")
                & model_table["candidate"].str.startswith("strict_history")
            ].sort_values("pr_auc", ascending=False).iloc[0].to_dict()
        ),
        "best_sensitivity_model": (
            model_table[
                (model_table["split"] == "validation")
                & model_table["candidate"].str.startswith("target_history")
            ].sort_values("pr_auc", ascending=False).iloc[0].to_dict()
        ),
        "best_strict_blend": strict_blends.iloc[0].to_dict(),
        "best_sensitivity_blend": sensitivity_blends.iloc[0].to_dict(),
    }
    (RES / "history_refinement_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
