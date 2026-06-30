"""Seed ensembling and class-weight experiments for the optimized CatBoost model."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bonus_experiments import catboost_frame  # noqa: E402
from diabetes_readmission_project import patient_level_split, prepare_data  # noqa: E402
from optimization_lab import add_rich_features  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"

BASE_PARAMS = {
    "iterations": 239,
    "depth": 5,
    "learning_rate": 0.0865970873429558,
    "l2_leaf_reg": 12.456101835909122,
    "random_strength": 0.10294864098843955,
    "bagging_temperature": 0.2727374508106509,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": -1,
}


def score(name, split, y, probability, **extra):
    return {
        "candidate": name,
        "split": split,
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, probability),
        **extra,
    }


def main():
    start = time.time()
    frame, _ = prepare_data()
    frame, numeric, categorical = add_rich_features(frame)
    train_idx, val_idx, test_idx, _ = patient_level_split(frame)
    features = catboost_frame(frame, numeric, categorical)
    cat_positions = [features.columns.get_loc(col) for col in categorical]
    y_train = frame.loc[train_idx, "target_30d"].to_numpy()
    y_val = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = frame.loc[test_idx, "target_30d"].to_numpy()

    rows = []
    seed_val, seed_test = [], []
    seeds = [42, 3407, 2026, 17, 31415]
    for seed in seeds:
        model = CatBoostClassifier(**BASE_PARAMS, random_seed=seed)
        model.fit(features.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
        val_probability = model.predict_proba(features.loc[val_idx])[:, 1]
        test_probability = model.predict_proba(features.loc[test_idx])[:, 1]
        seed_val.append(val_probability)
        seed_test.append(test_probability)
        rows.append(score(f"seed_{seed}", "validation", y_val, val_probability, seed=seed))
        rows.append(score(f"seed_{seed}", "test", y_test, test_probability, seed=seed))
        cumulative_val = np.mean(seed_val, axis=0)
        cumulative_test = np.mean(seed_test, axis=0)
        rows.append(score(f"seed_ensemble_{len(seed_val)}", "validation", y_val, cumulative_val,
                          ensemble_size=len(seed_val)))
        rows.append(score(f"seed_ensemble_{len(seed_val)}", "test", y_test, cumulative_test,
                          ensemble_size=len(seed_val)))
        print(f"seed ensemble {len(seed_val)}: val PR={rows[-2]['pr_auc']:.5f}, test PR={rows[-1]['pr_auc']:.5f}", flush=True)

    weighted_predictions = {}
    for positive_weight in [1.25, 1.5, 2.0]:
        model = CatBoostClassifier(
            **BASE_PARAMS,
            random_seed=42,
            class_weights=[1.0, positive_weight],
        )
        model.fit(features.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
        val_probability = model.predict_proba(features.loc[val_idx])[:, 1]
        test_probability = model.predict_proba(features.loc[test_idx])[:, 1]
        name = f"positive_weight_{positive_weight:g}"
        weighted_predictions[name] = (val_probability, test_probability)
        rows.append(score(name, "validation", y_val, val_probability, positive_weight=positive_weight))
        rows.append(score(name, "test", y_test, test_probability, positive_weight=positive_weight))
        print(f"{name}: val PR={rows[-2]['pr_auc']:.5f}, test PR={rows[-1]['pr_auc']:.5f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(TAB / "12_catboost_ensemble_weights.csv", index=False, encoding="utf-8-sig")

    validation = table[table["split"] == "validation"]
    best_name = validation.sort_values("pr_auc", ascending=False).iloc[0]["candidate"]
    if best_name.startswith("seed_ensemble_"):
        count = int(best_name.rsplit("_", 1)[1])
        best_val = np.mean(seed_val[:count], axis=0)
        best_test = np.mean(seed_test[:count], axis=0)
    elif best_name.startswith("seed_"):
        seed = int(best_name.split("_", 1)[1])
        index = seeds.index(seed)
        best_val, best_test = seed_val[index], seed_test[index]
    else:
        best_val, best_test = weighted_predictions[best_name]

    saved = np.load(RES / "optimized_probabilities.npz")
    lgb_val = saved["val_LightGBM_Rich"]
    lgb_test = saved["test_LightGBM_Rich"]
    blend_rows = []
    for light_weight in np.arange(0.0, 0.301, 0.01):
        val_probability = (1 - light_weight) * best_val + light_weight * lgb_val
        test_probability = (1 - light_weight) * best_test + light_weight * lgb_test
        blend_rows.append(score("selected_catboost_lgb_blend", "validation", y_val, val_probability,
                                catboost_candidate=best_name, lightgbm_weight=light_weight))
        blend_rows.append(score("selected_catboost_lgb_blend", "test", y_test, test_probability,
                                catboost_candidate=best_name, lightgbm_weight=light_weight))
    blend_table = pd.DataFrame(blend_rows)
    blend_table.to_csv(TAB / "13_ensemble_blend_search.csv", index=False, encoding="utf-8-sig")
    best_blend = blend_table[blend_table["split"] == "validation"].sort_values("pr_auc", ascending=False).iloc[0]
    test_blend = blend_table[
        (blend_table["split"] == "test")
        & np.isclose(blend_table["lightgbm_weight"], best_blend["lightgbm_weight"])
    ].iloc[0]
    summary = {
        "runtime_seconds": time.time() - start,
        "selected_catboost_candidate": best_name,
        "selected_validation_pr_auc": float(validation.loc[validation["candidate"] == best_name, "pr_auc"].iloc[0]),
        "selected_test_pr_auc": float(table[(table["candidate"] == best_name) & (table["split"] == "test")]["pr_auc"].iloc[0]),
        "best_lightgbm_weight": float(best_blend["lightgbm_weight"]),
        "blend_validation_pr_auc": float(best_blend["pr_auc"]),
        "blend_validation_roc_auc": float(best_blend["roc_auc"]),
        "blend_test_pr_auc": float(test_blend["pr_auc"]),
        "blend_test_roc_auc": float(test_blend["roc_auc"]),
        "blend_test_brier": float(test_blend["brier"]),
    }
    (RES / "optimization_ensemble.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        RES / "ensemble_probabilities.npz",
        y_val=y_val,
        y_test=y_test,
        selected_val=best_val,
        selected_test=best_test,
        blend_val=(1 - best_blend["lightgbm_weight"]) * best_val + best_blend["lightgbm_weight"] * lgb_val,
        blend_test=(1 - best_blend["lightgbm_weight"]) * best_test + best_blend["lightgbm_weight"] * lgb_test,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
