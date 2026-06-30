"""Patient-level cross-validation for the strict longitudinal-history effect."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bonus_experiments import catboost_frame  # noqa: E402
from diabetes_readmission_project import SEED, prepare_data  # noqa: E402
from optimization_extra_search import BASE_CAT_PARAMS, add_ultra_features, feature_variant  # noqa: E402
from optimization_research_pass import add_longitudinal_features, unique  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"


PARAMS = {
    **BASE_CAT_PARAMS,
    "iterations": 620,
    "depth": 5,
    "learning_rate": 0.038,
    "l2_leaf_reg": 24.0,
    "random_strength": 0.45,
    "bagging_temperature": 0.75,
    "random_seed": 3407,
    "thread_count": -1,
}


def score(model, fold, y, probability, fit_seconds):
    return {
        "model": model,
        "fold": fold,
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "log_loss": log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7)),
        "fit_seconds": fit_seconds,
        "n_validation": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }


def fit_predict(frame, numeric, categorical, train_pos, valid_pos):
    features = catboost_frame(frame, numeric, categorical)
    cat_positions = [features.columns.get_loc(col) for col in categorical]
    train_idx = frame.index.to_numpy()[train_pos]
    valid_idx = frame.index.to_numpy()[valid_pos]
    y_train = frame.loc[train_idx, "target_30d"].to_numpy()
    started = time.time()
    model = CatBoostClassifier(**PARAMS)
    model.fit(features.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
    fit_seconds = time.time() - started
    probability = model.predict_proba(features.loc[valid_idx])[:, 1]
    return probability, fit_seconds


def main():
    started_all = time.time()
    base_frame, _ = prepare_data()
    ultra_frame, ultra_num, ultra_cat = add_ultra_features(base_frame)
    strict_frame, history_num, history_cat = add_longitudinal_features(
        ultra_frame, include_target_history=False
    )
    prefix_num, prefix_cat = feature_variant(ultra_frame, ultra_num, ultra_cat, "ultra_prefix_no_raw_diag")
    history_prefix_num = unique(prefix_num + history_num)
    history_prefix_cat = unique(prefix_cat + history_cat)

    patient_target = strict_frame.groupby("patient_nbr")["target_30d"].max()
    patient_labels = strict_frame["patient_nbr"].map(patient_target).to_numpy()
    groups = strict_frame["patient_nbr"].to_numpy()
    y = strict_frame["target_30d"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (train_pos, valid_pos) in enumerate(splitter.split(strict_frame, patient_labels, groups), start=1):
        y_valid = y[valid_pos]
        print(f"Fold {fold}: train={len(train_pos)}, valid={len(valid_pos)}", flush=True)
        p_no_history, seconds = fit_predict(ultra_frame, prefix_num, prefix_cat, train_pos, valid_pos)
        row = score("prefix_catboost_no_history", fold, y_valid, p_no_history, seconds)
        rows.append(row)
        print(f"  no-history AP={row['pr_auc']:.5f}, ROC={row['roc_auc']:.5f}", flush=True)
        pd.DataFrame(rows).to_csv(TAB / "42_history_group_cv.csv", index=False, encoding="utf-8-sig")

        p_history, seconds = fit_predict(strict_frame, history_prefix_num, history_prefix_cat, train_pos, valid_pos)
        row = score("prefix_catboost_strict_history", fold, y_valid, p_history, seconds)
        rows.append(row)
        print(f"  strict-history AP={row['pr_auc']:.5f}, ROC={row['roc_auc']:.5f}", flush=True)
        pd.DataFrame(rows).to_csv(TAB / "42_history_group_cv.csv", index=False, encoding="utf-8-sig")

    table = pd.DataFrame(rows)
    summary_rows = []
    for model, group in table.groupby("model"):
        summary_rows.append({
            "model": model,
            "mean_pr_auc": group["pr_auc"].mean(),
            "sd_pr_auc": group["pr_auc"].std(ddof=1),
            "mean_roc_auc": group["roc_auc"].mean(),
            "sd_roc_auc": group["roc_auc"].std(ddof=1),
            "mean_brier": group["brier"].mean(),
            "mean_log_loss": group["log_loss"].mean(),
        })
    summary = pd.DataFrame(summary_rows).sort_values("mean_pr_auc", ascending=False)
    summary.to_csv(TAB / "43_history_group_cv_summary.csv", index=False, encoding="utf-8-sig")

    pivot = table.pivot(index="fold", columns="model", values="pr_auc")
    if {"prefix_catboost_strict_history", "prefix_catboost_no_history"}.issubset(pivot.columns):
        deltas = pivot["prefix_catboost_strict_history"] - pivot["prefix_catboost_no_history"]
    else:
        deltas = pd.Series(dtype=float)
    result = {
        "runtime_seconds": time.time() - started_all,
        "summary": summary.to_dict(orient="records"),
        "fold_pr_auc_delta_history_minus_no_history": [float(x) for x in deltas.to_numpy()],
        "mean_pr_auc_delta": float(deltas.mean()) if len(deltas) else None,
        "all_folds_positive_delta": bool((deltas > 0).all()) if len(deltas) else None,
    }
    (RES / "history_group_cv_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
