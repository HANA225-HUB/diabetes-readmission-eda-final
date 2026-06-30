"""OOF stacking check for strict longitudinal-history CatBoost models.

The meta-learner is trained only on patient-level out-of-fold predictions inside
the training split, then evaluated once on the locked validation and test sets.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bonus_experiments import catboost_frame  # noqa: E402
from diabetes_readmission_project import SEED, patient_level_split, prepare_data  # noqa: E402
from optimization_extra_search import BASE_CAT_PARAMS, add_ultra_features, feature_variant  # noqa: E402
from optimization_research_pass import add_longitudinal_features, unique  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"


def candidate_specs():
    base_no_bagging = {k: v for k, v in BASE_CAT_PARAMS.items() if k != "bagging_temperature"}
    return {
        "strict_history_prefix_regularized_history": ("ultra_prefix_no_raw_diag", {
            **BASE_CAT_PARAMS, "iterations": 620, "depth": 5, "learning_rate": 0.038,
            "l2_leaf_reg": 24.0, "random_strength": 0.45, "bagging_temperature": 0.75,
            "random_seed": 3407, "thread_count": -1,
        }),
        "strict_history_full_mvs_history": ("ultra_full", {
            **base_no_bagging, "iterations": 520, "depth": 6, "learning_rate": 0.050,
            "l2_leaf_reg": 11.0, "random_strength": 0.20, "bootstrap_type": "MVS",
            "subsample": 0.88, "random_seed": 3407, "thread_count": -1,
        }),
        "strict_history_prefix_slow_history": ("ultra_prefix_no_raw_diag", {
            **BASE_CAT_PARAMS, "iterations": 520, "depth": 6, "learning_rate": 0.046,
            "l2_leaf_reg": 9.0, "random_strength": 0.25, "bagging_temperature": 0.55,
            "random_seed": 3407, "thread_count": -1,
        }),
        "strict_history_full_bernoulli_history": ("ultra_full", {
            **base_no_bagging, "iterations": 520, "depth": 6, "learning_rate": 0.050,
            "l2_leaf_reg": 10.5, "random_strength": 0.18, "bootstrap_type": "Bernoulli",
            "subsample": 0.82, "random_seed": 3407, "thread_count": -1,
        }),
    }


def score(name, split, y, probability, **extra):
    return {
        "model": name,
        "split": split,
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, np.clip(probability, 0, 1)),
        "log_loss": log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7)),
        **extra,
    }


def logit_features(matrix):
    clipped = np.clip(matrix, 1e-5, 1 - 1e-5)
    return np.hstack([matrix, np.log(clipped / (1 - clipped))])


def compositions(total, size):
    if size == 1:
        yield [total]
        return
    for value in range(total + 1):
        for rest in compositions(total - value, size - 1):
            yield [value] + rest


def search_oof_weights(y, oof, names):
    rows = []
    step_count = 50
    for size in [2, 3, 4]:
        active = names[:size]
        indices = [names.index(name) for name in active]
        for units in compositions(step_count, size):
            weights = np.array([unit / step_count for unit in units])
            probability = oof[:, indices] @ weights
            rows.append({
                "kind": f"top{size}_simplex",
                "active_candidates": "+".join(active),
                "weights": json.dumps({name: float(weight) for name, weight in zip(active, weights)}),
                "oof_pr_auc": average_precision_score(y, probability),
                "oof_roc_auc": roc_auc_score(y, probability),
            })
    return pd.DataFrame(rows).sort_values("oof_pr_auc", ascending=False)


def main():
    started_all = time.time()
    frame, _ = prepare_data()
    frame, ultra_num, ultra_cat = add_ultra_features(frame)
    frame, history_num, history_cat = add_longitudinal_features(frame, include_target_history=False)
    train_idx, val_idx, test_idx, _ = patient_level_split(frame)
    y_train = frame.loc[train_idx, "target_30d"].to_numpy()
    y_val = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = frame.loc[test_idx, "target_30d"].to_numpy()

    specs = candidate_specs()
    feature_cache = {}
    for variant in sorted({variant for variant, _ in specs.values()}):
        base_num, base_cat = feature_variant(frame, ultra_num, ultra_cat, variant)
        numeric = unique(base_num + history_num)
        categorical = unique(base_cat + history_cat)
        features = catboost_frame(frame, numeric, categorical)
        cat_positions = [features.columns.get_loc(col) for col in categorical]
        feature_cache[variant] = (features, cat_positions)

    patient_target = frame.loc[train_idx].groupby("patient_nbr")["target_30d"].max()
    train_patient_labels = frame.loc[train_idx, "patient_nbr"].map(patient_target).to_numpy()
    train_groups = frame.loc[train_idx, "patient_nbr"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20260626)

    names = list(specs)
    oof = np.zeros((len(train_idx), len(names)), dtype=float)
    fold_rows = []
    for fold, (subtrain_pos, holdout_pos) in enumerate(
        splitter.split(train_idx, train_patient_labels, train_groups),
        start=1,
    ):
        subtrain_idx = train_idx[subtrain_pos]
        holdout_idx = train_idx[holdout_pos]
        y_holdout = frame.loc[holdout_idx, "target_30d"].to_numpy()
        print(f"OOF fold {fold}: train={len(subtrain_idx)}, holdout={len(holdout_idx)}", flush=True)
        for col, name in enumerate(names):
            variant, params = specs[name]
            features, cat_positions = feature_cache[variant]
            started = time.time()
            model = CatBoostClassifier(**params)
            model.fit(
                features.loc[subtrain_idx],
                frame.loc[subtrain_idx, "target_30d"].to_numpy(),
                cat_features=cat_positions,
                verbose=False,
            )
            probability = model.predict_proba(features.loc[holdout_idx])[:, 1]
            oof[holdout_pos, col] = probability
            row = score(name, f"oof_fold_{fold}", y_holdout, probability, fit_seconds=time.time() - started)
            fold_rows.append(row)
            print(f"  {name}: AP={row['pr_auc']:.5f}, ROC={row['roc_auc']:.5f}", flush=True)
            pd.DataFrame(fold_rows).to_csv(TAB / "44_history_oof_base_folds.csv", index=False, encoding="utf-8-sig")

    base_oof_scores = [
        score(name, "oof_train", y_train, oof[:, col])
        for col, name in enumerate(names)
    ]
    base_oof_table = pd.DataFrame(base_oof_scores).sort_values("pr_auc", ascending=False)
    base_oof_table.to_csv(TAB / "45_history_oof_base_summary.csv", index=False, encoding="utf-8-sig")
    ordered_names = base_oof_table["model"].tolist()
    ordered_indices = [names.index(name) for name in ordered_names]
    ordered_oof = oof[:, ordered_indices]

    weight_search = search_oof_weights(y_train, ordered_oof, ordered_names)
    weight_search.to_csv(TAB / "46_history_oof_weight_search.csv", index=False, encoding="utf-8-sig")
    best_weights = json.loads(weight_search.iloc[0]["weights"])

    stacker = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)),
    ])
    stacker.fit(logit_features(ordered_oof), y_train)

    saved = np.load(RES / "history_refinement_probabilities.npz")
    val_matrix = np.column_stack([saved[f"val_{name}"] for name in ordered_names])
    test_matrix = np.column_stack([saved[f"test_{name}"] for name in ordered_names])
    convex_val = sum(best_weights[name] * saved[f"val_{name}"] for name in best_weights)
    convex_test = sum(best_weights[name] * saved[f"test_{name}"] for name in best_weights)
    logistic_val = stacker.predict_proba(logit_features(val_matrix))[:, 1]
    logistic_test = stacker.predict_proba(logit_features(test_matrix))[:, 1]

    # Existing validation-selected strict blend for comparison.
    history_final = np.load(RES / "history_final_probabilities.npz")
    comparison_rows = []
    for name, val_probability, test_probability in [
        ("validation_selected_strict_history", history_final["strict_history_ensemble_val"], history_final["strict_history_ensemble_test"]),
        ("oof_convex_strict_history", convex_val, convex_test),
        ("oof_logistic_strict_history", logistic_val, logistic_test),
    ]:
        comparison_rows.append(score(name, "validation", y_val, val_probability))
        comparison_rows.append(score(name, "test", y_test, test_probability))
    comparison = pd.DataFrame(comparison_rows).sort_values(["split", "pr_auc"], ascending=[True, False])
    comparison.to_csv(TAB / "47_history_oof_stacking_locked.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        RES / "history_oof_stacking_probabilities.npz",
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        oof_predictions=oof,
        base_candidate_names=np.array(names),
        val_oof_convex=convex_val,
        test_oof_convex=convex_test,
        val_oof_logistic=logistic_val,
        test_oof_logistic=logistic_test,
    )
    summary = {
        "runtime_seconds": time.time() - started_all,
        "base_oof": base_oof_table.to_dict(orient="records"),
        "best_oof_convex": weight_search.iloc[0].to_dict(),
        "oof_logistic_coefficients": stacker.named_steps["model"].coef_.ravel().tolist(),
        "ordered_candidate_names": ordered_names,
        "locked_comparison": comparison.to_dict(orient="records"),
    }
    (RES / "history_oof_stacking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
