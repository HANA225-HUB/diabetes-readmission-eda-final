"""Focused extra search after the broad pass identified promising candidates."""

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


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"


def fast_rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks / len(values)


def score(name: str, split: str, y: np.ndarray, probability: np.ndarray, **extra):
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "candidate": name,
        "split": split,
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, np.clip(probability, 0, 1)),
        "log_loss": log_loss(y, clipped),
        **extra,
    }


def selected_configs():
    base_no_bagging = {k: v for k, v in BASE_CAT_PARAMS.items() if k != "bagging_temperature"}
    return [
        ("prefix_regularized_depth5", "ultra_prefix_no_raw_diag", {
            **BASE_CAT_PARAMS, "iterations": 620, "depth": 5, "learning_rate": 0.038,
            "l2_leaf_reg": 24.0, "random_strength": 0.45, "bagging_temperature": 0.75,
            "random_seed": 3407,
        }),
        ("prefix_slow_depth6", "ultra_prefix_no_raw_diag", {
            **BASE_CAT_PARAMS, "iterations": 520, "depth": 6, "learning_rate": 0.046,
            "l2_leaf_reg": 9.0, "random_strength": 0.25, "bagging_temperature": 0.55,
            "random_seed": 3407,
        }),
        ("full_mvs_depth6", "ultra_full", {
            **base_no_bagging, "iterations": 520, "depth": 6, "learning_rate": 0.050,
            "l2_leaf_reg": 11.0, "random_strength": 0.20, "bootstrap_type": "MVS",
            "subsample": 0.88, "random_seed": 3407,
        }),
        ("full_bernoulli_depth6", "ultra_full", {
            **base_no_bagging, "iterations": 520, "depth": 6, "learning_rate": 0.050,
            "l2_leaf_reg": 10.5, "random_strength": 0.18, "bootstrap_type": "Bernoulli",
            "subsample": 0.82, "random_seed": 3407,
        }),
        ("prefix_bernoulli_depth6", "ultra_prefix_no_raw_diag", {
            **base_no_bagging, "iterations": 520, "depth": 6, "learning_rate": 0.050,
            "l2_leaf_reg": 10.5, "random_strength": 0.18, "bootstrap_type": "Bernoulli",
            "subsample": 0.82, "random_seed": 3407,
        }),
        ("full_positive_weight_125", "ultra_full", {
            **BASE_CAT_PARAMS, "random_seed": 42, "class_weights": [1.0, 1.25],
        }),
    ]


def fit_selected(frame, numeric, categorical, train_idx, val_idx, test_idx):
    y_train = frame.loc[train_idx, "target_30d"].to_numpy()
    y_val = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = frame.loc[test_idx, "target_30d"].to_numpy()
    rows = []
    predictions = {}
    feature_cache = {}

    for short_name, variant, params in selected_configs():
        if variant not in feature_cache:
            variant_num, variant_cat = feature_variant(frame, numeric, categorical, variant)
            features = catboost_frame(frame, variant_num, variant_cat)
            cat_positions = [features.columns.get_loc(col) for col in variant_cat]
            feature_cache[variant] = (features, cat_positions, len(variant_num), len(variant_cat))
        features, cat_positions, n_num, n_cat = feature_cache[variant]
        name = f"focused_{short_name}"
        started = time.time()
        model = CatBoostClassifier(**params)
        model.fit(features.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
        runtime = time.time() - started
        val_probability = model.predict_proba(features.loc[val_idx])[:, 1]
        test_probability = model.predict_proba(features.loc[test_idx])[:, 1]
        predictions[name] = (val_probability, test_probability)
        rows.append(score(
            name, "validation", y_val, val_probability,
            variant=variant, config=short_name, runtime_seconds=runtime,
            numeric_features=n_num, categorical_features=n_cat,
        ))
        rows.append(score(
            name, "test", y_test, test_probability,
            variant=variant, config=short_name, runtime_seconds=runtime,
            numeric_features=n_num, categorical_features=n_cat,
        ))
        print(f"{name}: val AP={rows[-2]['pr_auc']:.5f}, test AP={rows[-1]['pr_auc']:.5f}", flush=True)

    return rows, predictions


def load_existing():
    existing = {}
    ensemble = np.load(RES / "ensemble_probabilities.npz")
    existing["final_tree_ensemble"] = (ensemble["blend_val"], ensemble["blend_test"])
    existing["selected_catboost"] = (ensemble["selected_val"], ensemble["selected_test"])
    optimized = np.load(RES / "optimized_probabilities.npz")
    key_map = {
        "old_catboost_rich": "CatBoost_Rich",
        "old_lightgbm_rich": "LightGBM_Rich",
        "old_xgboost_rich": "XGBoost_Rich",
        "old_optimized_blend": "Optimized_blend",
    }
    for label, key in key_map.items():
        if f"val_{key}" in optimized.files:
            existing[label] = (optimized[f"val_{key}"], optimized[f"test_{key}"])
    return existing, ensemble["y_val"], ensemble["y_test"]


def evaluate_blends(y_val, y_test, candidates):
    ranks = {
        name: (fast_rank01(val), fast_rank01(test))
        for name, (val, test) in candidates.items()
    }
    rows = []
    base_name = "final_tree_ensemble"
    for candidate in candidates:
        if candidate == base_name:
            continue
        for method in ["probability", "rank"]:
            source = candidates if method == "probability" else ranks
            base_val, base_test = source[base_name]
            cand_val, cand_test = source[candidate]
            for weight in np.arange(0, 0.601, 0.005):
                val_blend = (1 - weight) * base_val + weight * cand_val
                test_blend = (1 - weight) * base_test + weight * cand_test
                rows.append({
                    "blend": "pairwise_with_final", "method": method,
                    "weights": json.dumps({base_name: float(1 - weight), candidate: float(weight)}),
                    "validation_pr_auc": average_precision_score(y_val, val_blend),
                    "validation_roc_auc": roc_auc_score(y_val, val_blend),
                    "test_pr_auc": average_precision_score(y_test, test_blend),
                    "test_roc_auc": roc_auc_score(y_test, test_blend),
                    "test_brier": brier_score_loss(y_test, np.clip(test_blend, 0, 1)),
                })

    validation_scores = {
        name: average_precision_score(y_val, val)
        for name, (val, _) in candidates.items()
    }
    top_names = sorted(validation_scores, key=validation_scores.get, reverse=True)[:5]
    grids = {
        3: np.arange(0, 1.001, 0.02),
        4: np.arange(0, 1.001, 0.05),
    }
    for size, grid in grids.items():
        active = top_names[:size]
        for method in ["probability", "rank"]:
            source = candidates if method == "probability" else ranks
            for weights in product(grid, repeat=size):
                if not np.isclose(sum(weights), 1.0):
                    continue
                val_blend = sum(weight * source[name][0] for name, weight in zip(active, weights))
                test_blend = sum(weight * source[name][1] for name, weight in zip(active, weights))
                rows.append({
                    "blend": f"top{size}_grid", "method": method,
                    "weights": json.dumps({name: float(weight) for name, weight in zip(active, weights)}),
                    "validation_pr_auc": average_precision_score(y_val, val_blend),
                    "validation_roc_auc": roc_auc_score(y_val, val_blend),
                    "test_pr_auc": average_precision_score(y_test, test_blend),
                    "test_roc_auc": roc_auc_score(y_test, test_blend),
                    "test_brier": brier_score_loss(y_test, np.clip(test_blend, 0, 1)),
                })
    return pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False)


def main():
    started = time.time()
    frame, _ = prepare_data()
    frame, numeric, categorical = add_ultra_features(frame)
    train_idx, val_idx, test_idx, _ = patient_level_split(frame)
    model_rows, new_predictions = fit_selected(frame, numeric, categorical, train_idx, val_idx, test_idx)
    model_table = pd.DataFrame(model_rows).sort_values(["split", "pr_auc"], ascending=[True, False])
    model_table.to_csv(TAB / "23_focused_extra_models.csv", index=False, encoding="utf-8-sig")

    existing, y_val, y_test = load_existing()
    all_candidates = {**existing, **new_predictions}
    blend_table = evaluate_blends(y_val, y_test, all_candidates)
    blend_table.to_csv(TAB / "24_focused_extra_blends.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        RES / "focused_extra_probabilities.npz",
        y_val=y_val,
        y_test=y_test,
        **{f"val_{name}": value[0] for name, value in new_predictions.items()},
        **{f"test_{name}": value[1] for name, value in new_predictions.items()},
    )
    summary = {
        "runtime_seconds": time.time() - started,
        "best_single_validation": (
            model_table[model_table["split"] == "validation"].sort_values("pr_auc", ascending=False).iloc[0].to_dict()
        ),
        "best_blend_validation": blend_table.iloc[0].to_dict(),
    }
    (RES / "focused_extra_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
