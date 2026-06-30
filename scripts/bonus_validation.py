"""Validate the promising CatBoost branch without changing the report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bonus_experiments import TAB, RES, catboost_frame  # noqa: E402
from diabetes_readmission_project import (  # noqa: E402
    SEED,
    feature_sets,
    make_pipeline,
    patient_level_split,
    prepare_data,
)


def catboost_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=295,
        depth=8,
        learning_rate=0.045,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        random_seed=SEED,
        random_strength=0.5,
        allow_writing_files=False,
        verbose=False,
    )


def patient_bootstrap(y, groups, probabilities, repeats=500):
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    positions = {patient: np.flatnonzero(groups == patient) for patient in unique}
    rows = []
    for repeat in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([positions[patient] for patient in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        metric = {"repeat": repeat + 1}
        for name, probability in probabilities.items():
            metric[f"roc_{name}"] = roc_auc_score(y[idx], probability[idx])
            metric[f"pr_{name}"] = average_precision_score(y[idx], probability[idx])
        rows.append(metric)
    return pd.DataFrame(rows)


def main():
    df, _ = prepare_data()
    train_idx, val_idx, test_idx, _ = patient_level_split(df)
    y = df["target_30d"].to_numpy()
    y_train = df.loc[train_idx, "target_30d"].to_numpy()
    y_test = df.loc[test_idx, "target_30d"].to_numpy()

    engineered_num, engineered_cat = feature_sets()["Engineered"]
    engineered = catboost_frame(df, engineered_num, engineered_cat)
    cat_positions = [engineered.columns.get_loc(col) for col in engineered_cat]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for fold, (fit_pos, holdout_pos) in enumerate(
        cv.split(engineered, y, groups=df["patient_nbr"]), start=1
    ):
        model = catboost_model()
        model.fit(engineered.iloc[fit_pos], y[fit_pos], cat_features=cat_positions, verbose=False)
        probability = model.predict_proba(engineered.iloc[holdout_pos])[:, 1]
        cv_rows.append({
            "fold": fold,
            "n": len(holdout_pos),
            "positive_rate": y[holdout_pos].mean(),
            "roc_auc": roc_auc_score(y[holdout_pos], probability),
            "pr_auc": average_precision_score(y[holdout_pos], probability),
        })
        print("fold", cv_rows[-1])
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(TAB / "09_catboost_group_cv.csv", index=False, encoding="utf-8-sig")

    cat = catboost_model()
    cat.fit(engineered.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
    cat_test = cat.predict_proba(engineered.loc[test_idx])[:, 1]

    probabilities = {"catboost": cat_test}
    for feature_name in ["Raw", "Cleaned"]:
        numeric, categorical = feature_sets()[feature_name]
        columns = numeric + categorical
        model = make_pipeline("HistGradientBoosting", numeric, categorical)
        model.fit(df.loc[train_idx, columns], y_train)
        probabilities[f"histgb_{feature_name.lower()}"] = model.predict_proba(df.loc[test_idx, columns])[:, 1]

    bootstrap = patient_bootstrap(
        y_test,
        df.loc[test_idx, "patient_nbr"].to_numpy(),
        probabilities,
    )
    for comparator in ["histgb_raw", "histgb_cleaned"]:
        bootstrap[f"pr_delta_catboost_vs_{comparator}"] = bootstrap["pr_catboost"] - bootstrap[f"pr_{comparator}"]
        bootstrap[f"roc_delta_catboost_vs_{comparator}"] = bootstrap["roc_catboost"] - bootstrap[f"roc_{comparator}"]
    bootstrap.to_csv(TAB / "10_catboost_paired_bootstrap.csv", index=False, encoding="utf-8-sig")

    summary = {
        "catboost_cv_roc_mean": float(cv_df["roc_auc"].mean()),
        "catboost_cv_roc_sd": float(cv_df["roc_auc"].std(ddof=1)),
        "catboost_cv_pr_mean": float(cv_df["pr_auc"].mean()),
        "catboost_cv_pr_sd": float(cv_df["pr_auc"].std(ddof=1)),
    }
    for comparator in ["histgb_raw", "histgb_cleaned"]:
        for metric in ["pr", "roc"]:
            key = f"{metric}_delta_catboost_vs_{comparator}"
            values = bootstrap[key]
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_ci95"] = [float(values.quantile(0.025)), float(values.quantile(0.975))]
            summary[f"{key}_positive_probability"] = float((values > 0).mean())
    (RES / "catboost_validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        RES / "test_probabilities.npz",
        y=y_test,
        groups=df.loc[test_idx, "patient_nbr"].to_numpy(),
        **probabilities,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
