"""Seed refinement around the best focused CatBoost configuration."""

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
from optimization_extra_search import BASE_CAT_PARAMS, add_ultra_features, feature_variant  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"


BEST_CONFIG = {
    **BASE_CAT_PARAMS,
    "iterations": 620,
    "depth": 5,
    "learning_rate": 0.038,
    "l2_leaf_reg": 24.0,
    "random_strength": 0.45,
    "bagging_temperature": 0.75,
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
    started = time.time()
    frame, _ = prepare_data()
    frame, numeric, categorical = add_ultra_features(frame)
    variant_num, variant_cat = feature_variant(frame, numeric, categorical, "ultra_prefix_no_raw_diag")
    features = catboost_frame(frame, variant_num, variant_cat)
    cat_positions = [features.columns.get_loc(col) for col in variant_cat]
    train_idx, val_idx, test_idx, _ = patient_level_split(frame)
    y_train = frame.loc[train_idx, "target_30d"].to_numpy()
    y_val = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = frame.loc[test_idx, "target_30d"].to_numpy()

    focused = np.load(RES / "focused_extra_probabilities.npz")
    val_predictions = [focused["val_focused_prefix_regularized_depth5"]]
    test_predictions = [focused["test_focused_prefix_regularized_depth5"]]
    rows = [
        score("seed_3407", "validation", y_val, val_predictions[0], seed=3407),
        score("seed_3407", "test", y_test, test_predictions[0], seed=3407),
    ]

    for seed in [42, 2026, 17, 31415, 2718]:
        started_fit = time.time()
        model = CatBoostClassifier(**BEST_CONFIG, random_seed=seed)
        model.fit(features.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
        val_probability = model.predict_proba(features.loc[val_idx])[:, 1]
        test_probability = model.predict_proba(features.loc[test_idx])[:, 1]
        val_predictions.append(val_probability)
        test_predictions.append(test_probability)
        runtime = time.time() - started_fit
        rows.append(score(f"seed_{seed}", "validation", y_val, val_probability, seed=seed, runtime_seconds=runtime))
        rows.append(score(f"seed_{seed}", "test", y_test, test_probability, seed=seed, runtime_seconds=runtime))
        print(
            f"seed_{seed}: val AP={rows[-2]['pr_auc']:.5f}, test AP={rows[-1]['pr_auc']:.5f}",
            flush=True,
        )

    order = np.argsort([
        average_precision_score(y_val, pred) for pred in val_predictions
    ])[::-1]
    ensemble_rows = []
    ensemble_predictions = {}
    for count in range(1, len(order) + 1):
        chosen = order[:count]
        val_ensemble = np.mean([val_predictions[idx] for idx in chosen], axis=0)
        test_ensemble = np.mean([test_predictions[idx] for idx in chosen], axis=0)
        name = f"best_seed_ensemble_{count}"
        ensemble_predictions[name] = (val_ensemble, test_ensemble)
        ensemble_rows.append(score(name, "validation", y_val, val_ensemble, ensemble_size=count))
        ensemble_rows.append(score(name, "test", y_test, test_ensemble, ensemble_size=count))
        print(
            f"{name}: val AP={ensemble_rows[-2]['pr_auc']:.5f}, "
            f"test AP={ensemble_rows[-1]['pr_auc']:.5f}",
            flush=True,
        )

    table = pd.DataFrame(rows + ensemble_rows)
    table.to_csv(TAB / "30_extra_seed_refinement.csv", index=False, encoding="utf-8-sig")
    best = table[table["split"] == "validation"].sort_values("pr_auc", ascending=False).iloc[0].to_dict()
    summary = {
        "runtime_seconds": time.time() - started,
        "best_validation": best,
        "seed_order_by_validation": [int([3407, 42, 2026, 17, 31415, 2718][idx]) for idx in order],
    }
    np.savez_compressed(
        RES / "extra_seed_refinement_probabilities.npz",
        y_val=y_val,
        y_test=y_test,
        **{f"val_{name}": values[0] for name, values in ensemble_predictions.items()},
        **{f"test_{name}": values[1] for name, values in ensemble_predictions.items()},
    )
    (RES / "extra_seed_refinement_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
