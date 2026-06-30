"""Grouped cross-validation and feature ablation for the optimized CatBoost model."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bonus_experiments import catboost_frame  # noqa: E402
from diabetes_readmission_project import SEED, patient_level_split, prepare_data  # noqa: E402
from optimization_lab import add_rich_features  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"
TAB.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

PARAMS = {
    "iterations": 239,
    "depth": 5,
    "learning_rate": 0.0865970873429558,
    "l2_leaf_reg": 12.456101835909122,
    "random_strength": 0.10294864098843955,
    "bagging_temperature": 0.2727374508106509,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "random_seed": SEED,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": -1,
}


def fit_predict(frame, numeric, categorical, train_idx, eval_idx, seed_offset=0):
    features = catboost_frame(frame, numeric, categorical)
    cat_positions = [features.columns.get_loc(col) for col in categorical]
    model = CatBoostClassifier(**{**PARAMS, "random_seed": SEED + seed_offset})
    model.fit(
        features.loc[train_idx],
        frame.loc[train_idx, "target_30d"].to_numpy(),
        cat_features=cat_positions,
        verbose=False,
    )
    return model.predict_proba(features.loc[eval_idx])[:, 1]


def metric_row(experiment, split, y, probability, **extra):
    return {
        "experiment": experiment,
        "split": split,
        "n": len(y),
        "positive_rate": float(np.mean(y)),
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "brier": brier_score_loss(y, probability),
        **extra,
    }


def feature_variants(numeric, categorical):
    raw_diagnosis = {"diag_1", "diag_2", "diag_3"}
    raw_admin = {
        "admission_type_raw",
        "discharge_disposition_raw",
        "admission_source_raw",
        "medical_specialty",
        "payer_code",
    }
    variants = {
        "Rich full": (numeric, categorical),
        "Without raw diagnosis codes": (numeric, [c for c in categorical if c not in raw_diagnosis]),
        "Without raw administrative fields": (numeric, [c for c in categorical if c not in raw_admin]),
        "Core engineered only": (
            numeric,
            [c for c in categorical if c not in raw_diagnosis | raw_admin | {"age"}],
        ),
    }
    return variants


def main():
    start = time.time()
    frame, _ = prepare_data()
    frame, numeric, categorical = add_rich_features(frame)
    y = frame["target_30d"].to_numpy()
    groups = frame["patient_nbr"].to_numpy()

    cv_rows = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_pos, eval_pos) in enumerate(splitter.split(frame, y, groups), start=1):
        train_idx = frame.index.to_numpy()[train_pos]
        eval_idx = frame.index.to_numpy()[eval_pos]
        probability = fit_predict(frame, numeric, categorical, train_idx, eval_idx, fold)
        cv_rows.append(
            metric_row(
                "Rich CatBoost",
                f"fold_{fold}",
                y[eval_pos],
                probability,
                fold=fold,
                train_patients=int(pd.Series(groups[train_pos]).nunique()),
                eval_patients=int(pd.Series(groups[eval_pos]).nunique()),
            )
        )
        print(f"fold {fold}: PR={cv_rows[-1]['pr_auc']:.5f}, ROC={cv_rows[-1]['roc_auc']:.5f}", flush=True)

    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(TAB / "10_rich_catboost_group_cv.csv", index=False, encoding="utf-8-sig")

    train_idx, val_idx, test_idx, _ = patient_level_split(frame)
    ablation_rows = []
    for offset, (name, (variant_num, variant_cat)) in enumerate(feature_variants(numeric, categorical).items(), start=20):
        val_probability = fit_predict(frame, variant_num, variant_cat, train_idx, val_idx, offset)
        test_probability = fit_predict(frame, variant_num, variant_cat, train_idx, test_idx, offset)
        y_val = frame.loc[val_idx, "target_30d"].to_numpy()
        y_test = frame.loc[test_idx, "target_30d"].to_numpy()
        ablation_rows.append(metric_row(name, "validation", y_val, val_probability,
                                        numeric_features=len(variant_num), categorical_features=len(variant_cat)))
        ablation_rows.append(metric_row(name, "test", y_test, test_probability,
                                        numeric_features=len(variant_num), categorical_features=len(variant_cat)))
        print(f"{name}: val PR={ablation_rows[-2]['pr_auc']:.5f}, test PR={ablation_rows[-1]['pr_auc']:.5f}", flush=True)

    ablation_table = pd.DataFrame(ablation_rows)
    ablation_table.to_csv(TAB / "11_catboost_feature_ablation.csv", index=False, encoding="utf-8-sig")

    old_cv_path = HERE / "tables" / "09_group_cv.csv"
    old_cv = pd.read_csv(old_cv_path) if old_cv_path.exists() else None
    summary = {
        "runtime_seconds": time.time() - start,
        "catboost_parameters": PARAMS,
        "rich_catboost_cv": {
            "mean_pr_auc": float(cv_table["pr_auc"].mean()),
            "sd_pr_auc": float(cv_table["pr_auc"].std(ddof=1)),
            "mean_roc_auc": float(cv_table["roc_auc"].mean()),
            "sd_roc_auc": float(cv_table["roc_auc"].std(ddof=1)),
            "mean_brier": float(cv_table["brier"].mean()),
        },
        "previous_histgb_cv": None if old_cv is None else {
            "mean_pr_auc": float(old_cv["pr_auc"].mean()),
            "sd_pr_auc": float(old_cv["pr_auc"].std(ddof=1)),
            "mean_roc_auc": float(old_cv["roc_auc"].mean()),
            "sd_roc_auc": float(old_cv["roc_auc"].std(ddof=1)),
            "mean_brier": float(old_cv["brier"].mean()),
        },
        "ablation": ablation_table.to_dict(orient="records"),
    }
    (RES / "optimization_cv_ablation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
