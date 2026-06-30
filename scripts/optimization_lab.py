"""Model optimization lab. This script does not modify the report or its figures."""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bonus_experiments import catboost_frame  # noqa: E402
from diabetes_readmission_project import (  # noqa: E402
    SEED,
    choose_f2_threshold,
    feature_sets,
    make_pipeline,
    patient_level_split,
    prepare_data,
)


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"
for directory in (TAB, RES):
    directory.mkdir(parents=True, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def unique(items):
    return list(dict.fromkeys(items))


def add_rich_features(df: pd.DataFrame):
    frame = df.copy()
    for col in ["number_inpatient", "number_emergency", "number_outpatient", "num_medications", "num_lab_procedures"]:
        frame[f"log1p_{col}"] = np.log1p(frame[col].clip(lower=0))
    frame["prior_inpatient_emergency"] = frame["number_inpatient"] * frame["number_emergency"]
    frame["prior_any_acute"] = ((frame["number_inpatient"] + frame["number_emergency"]) > 0).astype(int)
    frame["medications_per_diagnosis"] = frame["num_medications"] / frame["number_diagnoses"].clip(lower=1)
    frame["labs_per_diagnosis"] = frame["num_lab_procedures"] / frame["number_diagnoses"].clip(lower=1)
    frame["age_prior_inpatient"] = frame["age_mid"] * frame["number_inpatient"]
    frame["age_utilization_band"] = (
        frame["age"].astype("string").fillna("Missing")
        + "|"
        + pd.cut(frame["prior_visits_total"], [-1, 0, 1, 3, 8, np.inf], labels=["0", "1", "2-3", "4-8", "9+"]).astype("string").fillna("Missing")
    )
    frame["discharge_prior_band"] = (
        frame["discharge_group"].astype("string").fillna("Missing")
        + "|"
        + pd.cut(frame["number_inpatient"], [-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3-4", "5+"]).astype("string").fillna("Missing")
    )

    engineered_num, engineered_cat = feature_sets()["Engineered"]
    numeric = unique(engineered_num + [
        "log1p_number_inpatient", "log1p_number_emergency", "log1p_number_outpatient",
        "log1p_num_medications", "log1p_num_lab_procedures", "prior_inpatient_emergency",
        "prior_any_acute", "medications_per_diagnosis", "labs_per_diagnosis", "age_prior_inpatient",
    ])
    categorical = unique(engineered_cat + [
        "age", "admission_type_raw", "discharge_disposition_raw", "admission_source_raw",
        "diag_1", "diag_2", "diag_3", "medical_specialty", "payer_code",
        "age_utilization_band", "discharge_prior_band",
    ])
    return frame, numeric, categorical


def patient_internal_split(df, train_idx):
    train_df = df.loc[train_idx]
    target = train_df.groupby("patient_nbr")["target_30d"].max()
    fit_patients, tune_patients = train_test_split(
        target.index.to_numpy(), test_size=0.22, random_state=2026, stratify=target.to_numpy()
    )
    fit_idx = train_df.index[train_df["patient_nbr"].isin(fit_patients)].to_numpy()
    tune_idx = train_df.index[train_df["patient_nbr"].isin(tune_patients)].to_numpy()
    return fit_idx, tune_idx


def metrics(name, y, probability, threshold=None):
    if threshold is None:
        threshold, _ = choose_f2_threshold(y, probability)
    prediction = probability >= threshold
    return {
        "model": name,
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "log_loss": log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "f2": fbeta_score(y, prediction, beta=2, zero_division=0),
    }


def operating_points(name, y_val, p_val, y_test, p_test):
    rows = []
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    for target_recall in [0.50, 0.60, 0.70, 0.80]:
        valid = np.flatnonzero(recall[:-1] >= target_recall)
        selected = valid[-1]
        threshold = thresholds[selected]
        pred = p_test >= threshold
        rows.append({
            "model": name,
            "validation_target_recall": target_recall,
            "threshold": threshold,
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "selected_fraction": pred.mean(),
        })
    return rows


def topk_rows(name, y, p):
    order = np.argsort(-p)
    rows = []
    for fraction in [0.03, 0.05, 0.10, 0.20]:
        k = int(np.ceil(len(y) * fraction))
        idx = order[:k]
        tp = y[idx].sum()
        rows.append({
            "model": name, "capacity": fraction, "selected_n": k,
            "precision": tp / k, "recall_capture": tp / y.sum(),
            "lift": (tp / k) / y.mean(),
        })
    return rows


def make_ordinal_pipeline(model, numeric, categorical):
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical),
    ], sparse_threshold=0)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def tune_catboost(frame, numeric, categorical, fit_idx, tune_idx, trials=18):
    X = catboost_frame(frame, numeric, categorical)
    cat_pos = [X.columns.get_loc(col) for col in categorical]
    y_fit = frame.loc[fit_idx, "target_30d"].to_numpy()
    y_tune = frame.loc[tune_idx, "target_30d"].to_numpy()

    def objective(trial):
        params = {
            "iterations": 900,
            "depth": trial.suggest_int("depth", 5, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.09, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 18.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.05, 1.5, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
            "loss_function": "Logloss", "eval_metric": "PRAUC",
            "random_seed": SEED, "allow_writing_files": False, "verbose": False,
        }
        model = CatBoostClassifier(**params)
        model.fit(X.loc[fit_idx], y_fit, cat_features=cat_pos,
                  eval_set=(X.loc[tune_idx], y_tune), early_stopping_rounds=70, verbose=False)
        p = model.predict_proba(X.loc[tune_idx])[:, 1]
        trial.set_user_attr("best_iteration", int(model.get_best_iteration()))
        trial.set_user_attr("roc_auc", float(roc_auc_score(y_tune, p)))
        return average_precision_score(y_tune, p)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study, X, cat_pos


def tune_lightgbm(frame, numeric, categorical, fit_idx, tune_idx, trials=22):
    y_fit = frame.loc[fit_idx, "target_30d"].to_numpy()
    y_tune = frame.loc[tune_idx, "target_30d"].to_numpy()

    def objective(trial):
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=1600,
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 110),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 25, 180),
            subsample=trial.suggest_float("subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.2, 12.0, log=True),
            verbosity=-1, random_state=SEED, n_jobs=-1,
        )
        pipe = make_ordinal_pipeline(model, numeric, categorical)
        prep = pipe.named_steps["preprocess"]
        X_fit = prep.fit_transform(frame.loc[fit_idx, numeric + categorical])
        X_tune = prep.transform(frame.loc[tune_idx, numeric + categorical])
        model.fit(X_fit, y_fit, eval_set=[(X_tune, y_tune)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(70, verbose=False)])
        p = model.predict_proba(X_tune)[:, 1]
        trial.set_user_attr("best_iteration", int(model.best_iteration_))
        trial.set_user_attr("roc_auc", float(roc_auc_score(y_tune, p)))
        return average_precision_score(y_tune, p)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED + 1))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study


def tune_xgboost(frame, numeric, categorical, fit_idx, tune_idx, trials=18):
    y_fit = frame.loc[fit_idx, "target_30d"].to_numpy()
    y_tune = frame.loc[tune_idx, "target_30d"].to_numpy()

    def objective(trial):
        model = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_estimators=1500,
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_float("min_child_weight", 2.0, 30.0, log=True),
            subsample=trial.suggest_float("subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
            gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.3, 15.0, log=True),
            early_stopping_rounds=70, random_state=SEED, n_jobs=-1,
        )
        pipe = make_ordinal_pipeline(model, numeric, categorical)
        prep = pipe.named_steps["preprocess"]
        X_fit = prep.fit_transform(frame.loc[fit_idx, numeric + categorical])
        X_tune = prep.transform(frame.loc[tune_idx, numeric + categorical])
        model.fit(X_fit, y_fit, eval_set=[(X_tune, y_tune)], verbose=False)
        p = model.predict_proba(X_tune)[:, 1]
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        trial.set_user_attr("roc_auc", float(roc_auc_score(y_tune, p)))
        return average_precision_score(y_tune, p)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED + 2))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study


def trial_table(name, study):
    rows = []
    for trial in study.trials:
        rows.append({
            "model": name, "trial": trial.number, "pr_auc": trial.value,
            "roc_auc": trial.user_attrs.get("roc_auc"),
            "best_iteration": trial.user_attrs.get("best_iteration"),
            **trial.params,
        })
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)


def fit_final_models(frame, numeric, categorical, train_idx, val_idx, test_idx, studies):
    columns = numeric + categorical
    y_train = frame.loc[train_idx, "target_30d"].to_numpy()
    y_val = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = frame.loc[test_idx, "target_30d"].to_numpy()
    predictions = {}
    details = {}

    # Current locked baseline.
    base_num, base_cat = feature_sets()["Cleaned"]
    base_cols = base_num + base_cat
    baseline = make_pipeline("HistGradientBoosting", base_num, base_cat)
    baseline.fit(frame.loc[train_idx, base_cols], y_train)
    predictions["HistGB baseline"] = {
        "val": baseline.predict_proba(frame.loc[val_idx, base_cols])[:, 1],
        "test": baseline.predict_proba(frame.loc[test_idx, base_cols])[:, 1],
    }

    # CatBoost.
    cat_study = studies["CatBoost Rich"]
    cat_params = dict(cat_study.best_params)
    cat_iterations = max(80, int(cat_study.best_trial.user_attrs["best_iteration"] * 1.15))
    X_cat = catboost_frame(frame, numeric, categorical)
    cat_pos = [X_cat.columns.get_loc(col) for col in categorical]
    cat = CatBoostClassifier(
        iterations=cat_iterations, **cat_params, loss_function="Logloss", eval_metric="PRAUC",
        random_seed=SEED, allow_writing_files=False, verbose=False,
    )
    cat.fit(X_cat.loc[train_idx], y_train, cat_features=cat_pos,
            eval_set=(X_cat.loc[val_idx], y_val), early_stopping_rounds=80, verbose=False)
    predictions["CatBoost Rich"] = {
        "val": cat.predict_proba(X_cat.loc[val_idx])[:, 1],
        "test": cat.predict_proba(X_cat.loc[test_idx])[:, 1],
    }
    details["CatBoost Rich"] = {"params": cat_params, "iteration": int(cat.get_best_iteration())}

    # Shared ordinal matrices for LightGBM/XGBoost.
    preprocess = make_ordinal_pipeline(lgb.LGBMClassifier(), numeric, categorical).named_steps["preprocess"]
    X_train = preprocess.fit_transform(frame.loc[train_idx, columns])
    X_val = preprocess.transform(frame.loc[val_idx, columns])
    X_test = preprocess.transform(frame.loc[test_idx, columns])

    lgb_study = studies["LightGBM Rich"]
    lgb_params = dict(lgb_study.best_params)
    light = lgb.LGBMClassifier(
        objective="binary", n_estimators=1800, **lgb_params,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )
    light.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(90, verbose=False)])
    predictions["LightGBM Rich"] = {
        "val": light.predict_proba(X_val)[:, 1], "test": light.predict_proba(X_test)[:, 1],
    }
    details["LightGBM Rich"] = {"params": lgb_params, "iteration": int(light.best_iteration_)}

    xgb_study = studies["XGBoost Rich"]
    xgb_params = dict(xgb_study.best_params)
    boost = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_estimators=1800,
        **xgb_params, early_stopping_rounds=90, random_state=SEED, n_jobs=-1,
    )
    boost.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    predictions["XGBoost Rich"] = {
        "val": boost.predict_proba(X_val)[:, 1], "test": boost.predict_proba(X_test)[:, 1],
    }
    details["XGBoost Rich"] = {"params": xgb_params, "iteration": int(boost.best_iteration)}
    return predictions, details, y_val, y_test


def rank01(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def search_blends(predictions, y_val):
    candidate_names = ["CatBoost Rich", "LightGBM Rich", "XGBoost Rich"]
    rows = []
    for weights in product(np.arange(0, 1.01, 0.1), repeat=3):
        if not np.isclose(sum(weights), 1.0):
            continue
        for method in ["probability", "rank"]:
            components = [
                predictions[name]["val"] if method == "probability" else rank01(predictions[name]["val"])
                for name in candidate_names
            ]
            probability = sum(weight * component for weight, component in zip(weights, components))
            rows.append({
                "method": method,
                "w_catboost": weights[0], "w_lightgbm": weights[1], "w_xgboost": weights[2],
                "validation_pr_auc": average_precision_score(y_val, probability),
                "validation_roc_auc": roc_auc_score(y_val, probability),
            })
    return pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False)


def apply_blend(row, predictions, split):
    names = ["CatBoost Rich", "LightGBM Rich", "XGBoost Rich"]
    weights = [row["w_catboost"], row["w_lightgbm"], row["w_xgboost"]]
    values = [predictions[name][split] for name in names]
    if row["method"] == "rank":
        values = [rank01(value) for value in values]
    return sum(weight * value for weight, value in zip(weights, values))


def main():
    start = time.time()
    df, _ = prepare_data()
    df, numeric, categorical = add_rich_features(df)
    train_idx, val_idx, test_idx, split = patient_level_split(df)
    fit_idx, tune_idx = patient_internal_split(df, train_idx)
    split.update({"internal_fit_rows": len(fit_idx), "internal_tune_rows": len(tune_idx)})

    print("Rich features", len(numeric), "numeric", len(categorical), "categorical")
    print("Internal rows", len(fit_idx), len(tune_idx))

    cat_study, _, _ = tune_catboost(df, numeric, categorical, fit_idx, tune_idx)
    print("CatBoost", cat_study.best_value, cat_study.best_params)
    lgb_study = tune_lightgbm(df, numeric, categorical, fit_idx, tune_idx)
    print("LightGBM", lgb_study.best_value, lgb_study.best_params)
    xgb_study = tune_xgboost(df, numeric, categorical, fit_idx, tune_idx)
    print("XGBoost", xgb_study.best_value, xgb_study.best_params)
    studies = {"CatBoost Rich": cat_study, "LightGBM Rich": lgb_study, "XGBoost Rich": xgb_study}

    pd.concat([trial_table(name, study) for name, study in studies.items()], ignore_index=True).to_csv(
        TAB / "01_tuning_trials.csv", index=False, encoding="utf-8-sig"
    )
    predictions, details, y_val, y_test = fit_final_models(
        df, numeric, categorical, train_idx, val_idx, test_idx, studies
    )

    validation_rows = []
    for name, pred in predictions.items():
        row = metrics(name, y_val, pred["val"])
        validation_rows.append(row)
    validation = pd.DataFrame(validation_rows).sort_values("pr_auc", ascending=False)
    validation.to_csv(TAB / "02_validation_models.csv", index=False, encoding="utf-8-sig")

    blends = search_blends(predictions, y_val)
    blends.to_csv(TAB / "03_blend_search.csv", index=False, encoding="utf-8-sig")
    best_blend = blends.iloc[0]
    blend_val = apply_blend(best_blend, predictions, "val")
    blend_test = apply_blend(best_blend, predictions, "test")
    predictions["Optimized blend"] = {"val": blend_val, "test": blend_test}

    test_rows, op_rows, capacity_rows = [], [], []
    for name, pred in predictions.items():
        threshold, _ = choose_f2_threshold(y_val, pred["val"])
        test_rows.append(metrics(name, y_test, pred["test"], threshold))
        op_rows.extend(operating_points(name, y_val, pred["val"], y_test, pred["test"]))
        capacity_rows.extend(topk_rows(name, y_test, pred["test"]))
    test_result = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False)
    test_result.to_csv(TAB / "04_test_models.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(op_rows).to_csv(TAB / "05_operating_points.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(capacity_rows).to_csv(TAB / "06_topk_capacity.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        RES / "optimized_probabilities.npz", y_val=y_val, y_test=y_test,
        **{f"val_{name.replace(' ', '_')}": pred["val"] for name, pred in predictions.items()},
        **{f"test_{name.replace(' ', '_')}": pred["test"] for name, pred in predictions.items()},
    )
    summary = {
        "runtime_seconds": time.time() - start,
        "split": split,
        "rich_feature_counts": {"numeric": len(numeric), "categorical": len(categorical)},
        "best_internal_tuning": {name: study.best_value for name, study in studies.items()},
        "best_params": details,
        "best_blend": best_blend.to_dict(),
        "validation": validation.to_dict(orient="records"),
        "test": test_result.to_dict(orient="records"),
    }
    (RES / "optimization_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
