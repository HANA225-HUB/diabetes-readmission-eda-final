"""Research-style optimization pass for diabetes readmission prediction.

The goal is not only to chase the test score, but to test modeling hypotheses:
1. Does non-leaky longitudinal patient history help?
2. Does modeling the original 3-class readmission outcome help?
3. Do asymmetric CatBoost tree growth policies improve the current ensemble?
"""

from __future__ import annotations

import json
import sys
import time
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


BEST_PREFIX_PARAMS = {
    **BASE_CAT_PARAMS,
    "iterations": 620,
    "depth": 5,
    "learning_rate": 0.038,
    "l2_leaf_reg": 24.0,
    "random_strength": 0.45,
    "bagging_temperature": 0.75,
    "random_seed": 3407,
}


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def score(name: str, split: str, y: np.ndarray, probability: np.ndarray, **extra):
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


def add_longitudinal_features(frame: pd.DataFrame, include_target_history: bool):
    frame = frame.copy()
    ordered = frame.sort_values(["patient_nbr", "encounter_id"]).index
    patient_groups = frame.loc[ordered].groupby("patient_nbr", sort=False)

    frame.loc[ordered, "observed_prior_encounters"] = patient_groups.cumcount().to_numpy()
    frame["observed_prior_encounters"] = frame["observed_prior_encounters"].fillna(0).astype(float)
    frame["log1p_observed_prior_encounters"] = np.log1p(frame["observed_prior_encounters"])
    frame["is_first_observed_encounter"] = (frame["observed_prior_encounters"] == 0).astype(int)

    prev_numeric = [
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
        "prior_visits_total", "care_intensity",
    ]
    for column in prev_numeric:
        previous = patient_groups[column].shift(1)
        frame.loc[ordered, f"prev_{column}"] = previous.to_numpy()
        frame[f"prev_{column}"] = frame[f"prev_{column}"].fillna(0)
        frame[f"delta_{column}"] = frame[column] - frame[f"prev_{column}"]

    previous_encounter = patient_groups["encounter_id"].shift(1)
    frame.loc[ordered, "encounter_id_gap_from_previous"] = (
        frame.loc[ordered, "encounter_id"] - previous_encounter
    ).fillna(0).to_numpy()
    frame["log1p_encounter_id_gap_from_previous"] = np.log1p(
        frame["encounter_id_gap_from_previous"].clip(lower=0)
    )

    prev_categorical = [
        "discharge_group", "admission_source_group", "admission_type_group",
        "diag1_group", "diag2_group", "diag3_group", "insulin", "change", "A1Cresult",
    ]
    for column in prev_categorical:
        previous = patient_groups[column].shift(1)
        frame.loc[ordered, f"prev_{column}"] = previous.fillna("No previous").astype(str).to_numpy()

    frame["same_prev_diag1_group"] = (
        frame["diag1_group"].astype(str) == frame["prev_diag1_group"].astype(str)
    ).astype(int)
    frame["same_prev_discharge_group"] = (
        frame["discharge_group"].astype(str) == frame["prev_discharge_group"].astype(str)
    ).astype(int)
    frame["current_prev_diag_path"] = (
        frame["prev_diag1_group"].astype(str) + "|" + frame["diag1_group"].astype(str)
    )
    frame["current_prev_discharge_path"] = (
        frame["prev_discharge_group"].astype(str) + "|" + frame["discharge_group"].astype(str)
    )

    numeric = [
        "observed_prior_encounters", "log1p_observed_prior_encounters", "is_first_observed_encounter",
        "encounter_id_gap_from_previous", "log1p_encounter_id_gap_from_previous",
        "same_prev_diag1_group", "same_prev_discharge_group",
    ]
    for column in prev_numeric:
        numeric.extend([f"prev_{column}", f"delta_{column}"])
    categorical = [f"prev_{column}" for column in prev_categorical] + [
        "current_prev_diag_path", "current_prev_discharge_path",
    ]

    if include_target_history:
        status = frame.loc[ordered, "readmitted"].astype(str)
        previous_status = patient_groups["readmitted"].shift(1)
        frame.loc[ordered, "prev_readmitted_status"] = previous_status.fillna("No previous").astype(str).to_numpy()
        is_30 = status.eq("<30").astype(int)
        is_any = status.ne("NO").astype(int)
        prior_30 = is_30.groupby(frame.loc[ordered, "patient_nbr"], sort=False).cumsum() - is_30
        prior_any = is_any.groupby(frame.loc[ordered, "patient_nbr"], sort=False).cumsum() - is_any
        prior_no = (
            status.eq("NO").astype(int).groupby(frame.loc[ordered, "patient_nbr"], sort=False).cumsum()
            - status.eq("NO").astype(int)
        )
        frame.loc[ordered, "prior_observed_30d_count"] = prior_30.to_numpy()
        frame.loc[ordered, "prior_observed_any_readmit_count"] = prior_any.to_numpy()
        frame.loc[ordered, "prior_observed_no_readmit_count"] = prior_no.to_numpy()
        denom = frame.loc[ordered, "observed_prior_encounters"].replace(0, np.nan)
        frame.loc[ordered, "prior_observed_30d_rate"] = (prior_30 / denom).fillna(0).to_numpy()
        frame.loc[ordered, "prior_observed_any_readmit_rate"] = (prior_any / denom).fillna(0).to_numpy()
        frame["previous_was_30d"] = frame["prev_readmitted_status"].eq("<30").astype(int)
        frame["previous_was_any_readmit"] = frame["prev_readmitted_status"].isin(["<30", ">30"]).astype(int)
        frame["prev_readmit_current_dx"] = (
            frame["prev_readmitted_status"].astype(str) + "|" + frame["diag1_group"].astype(str)
        )
        numeric.extend([
            "prior_observed_30d_count", "prior_observed_any_readmit_count",
            "prior_observed_no_readmit_count", "prior_observed_30d_rate",
            "prior_observed_any_readmit_rate", "previous_was_30d", "previous_was_any_readmit",
        ])
        categorical.extend(["prev_readmitted_status", "prev_readmit_current_dx"])

    return frame, unique(numeric), unique(categorical)


def fit_binary_catboost(name, frame, numeric, categorical, train_idx, val_idx, test_idx, params):
    features = catboost_frame(frame, numeric, categorical)
    cat_positions = [features.columns.get_loc(col) for col in categorical]
    y_train = frame.loc[train_idx, "target_30d"].to_numpy()
    y_val = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = frame.loc[test_idx, "target_30d"].to_numpy()
    started = time.time()
    model = CatBoostClassifier(**params)
    model.fit(features.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
    runtime = time.time() - started
    return [
        score(name, "validation", y_val, model.predict_proba(features.loc[val_idx])[:, 1], runtime_seconds=runtime),
        score(name, "test", y_test, model.predict_proba(features.loc[test_idx])[:, 1], runtime_seconds=runtime),
    ], {
        "val": model.predict_proba(features.loc[val_idx])[:, 1],
        "test": model.predict_proba(features.loc[test_idx])[:, 1],
    }


def fit_multiclass(name, frame, numeric, categorical, train_idx, val_idx, test_idx):
    features = catboost_frame(frame, numeric, categorical)
    cat_positions = [features.columns.get_loc(col) for col in categorical]
    y_train = frame.loc[train_idx, "readmitted"].astype(str).to_numpy()
    y_val_binary = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test_binary = frame.loc[test_idx, "target_30d"].to_numpy()
    params = {
        **{k: v for k, v in BEST_PREFIX_PARAMS.items() if k not in {"loss_function", "eval_metric"}},
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
    }
    started = time.time()
    model = CatBoostClassifier(**params)
    model.fit(features.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
    runtime = time.time() - started
    class_index = list(model.classes_).index("<30")
    val_probability = model.predict_proba(features.loc[val_idx])[:, class_index]
    test_probability = model.predict_proba(features.loc[test_idx])[:, class_index]
    return [
        score(name, "validation", y_val_binary, val_probability, runtime_seconds=runtime),
        score(name, "test", y_test_binary, test_probability, runtime_seconds=runtime),
    ], {"val": val_probability, "test": test_probability}


def fit_two_stage(name, frame, numeric, categorical, train_idx, val_idx, test_idx):
    features = catboost_frame(frame, numeric, categorical)
    cat_positions = [features.columns.get_loc(col) for col in categorical]
    y_any_train = frame.loc[train_idx, "readmitted"].ne("NO").astype(int).to_numpy()
    y_early_train = frame.loc[train_idx, "target_30d"].to_numpy()
    readmit_train_idx = train_idx[frame.loc[train_idx, "readmitted"].ne("NO").to_numpy()]
    y_cond_train = frame.loc[readmit_train_idx, "target_30d"].to_numpy()
    y_val = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = frame.loc[test_idx, "target_30d"].to_numpy()

    any_params = {
        **BEST_PREFIX_PARAMS,
        "iterations": 520,
        "learning_rate": 0.045,
        "eval_metric": "AUC",
    }
    cond_params = {
        **BEST_PREFIX_PARAMS,
        "iterations": 520,
        "learning_rate": 0.045,
    }
    started = time.time()
    any_model = CatBoostClassifier(**any_params)
    any_model.fit(features.loc[train_idx], y_any_train, cat_features=cat_positions, verbose=False)
    cond_model = CatBoostClassifier(**cond_params)
    cond_model.fit(features.loc[readmit_train_idx], y_cond_train, cat_features=cat_positions, verbose=False)
    runtime = time.time() - started
    val_probability = (
        any_model.predict_proba(features.loc[val_idx])[:, 1]
        * cond_model.predict_proba(features.loc[val_idx])[:, 1]
    )
    test_probability = (
        any_model.predict_proba(features.loc[test_idx])[:, 1]
        * cond_model.predict_proba(features.loc[test_idx])[:, 1]
    )
    return [
        score(name, "validation", y_val, val_probability, runtime_seconds=runtime),
        score(name, "test", y_test, test_probability, runtime_seconds=runtime),
    ], {"val": val_probability, "test": test_probability}


def main():
    started_all = time.time()
    base_frame, _ = prepare_data()
    ultra_frame, ultra_num, ultra_cat = add_ultra_features(base_frame)
    train_idx, val_idx, test_idx, _ = patient_level_split(ultra_frame)
    rows = []
    predictions = {}

    prefix_num, prefix_cat = feature_variant(ultra_frame, ultra_num, ultra_cat, "ultra_prefix_no_raw_diag")
    full_num, full_cat = feature_variant(ultra_frame, ultra_num, ultra_cat, "ultra_full")

    experiments = [
        ("research_prefix_regularized", ultra_frame, prefix_num, prefix_cat, BEST_PREFIX_PARAMS),
        ("research_prefix_depthwise", ultra_frame, prefix_num, prefix_cat, {
            **BEST_PREFIX_PARAMS, "iterations": 700, "depth": 6, "learning_rate": 0.035,
            "grow_policy": "Depthwise", "min_data_in_leaf": 64, "random_seed": 3407,
        }),
        ("research_prefix_lossguide", ultra_frame, prefix_num, prefix_cat, {
            **{k: v for k, v in BEST_PREFIX_PARAMS.items() if k != "depth"},
            "iterations": 700, "learning_rate": 0.035, "grow_policy": "Lossguide",
            "max_leaves": 48, "min_data_in_leaf": 64, "random_seed": 3407,
        }),
    ]
    for name, frame, numeric, categorical, params in experiments:
        print(f"Running {name}", flush=True)
        model_rows, pred = fit_binary_catboost(name, frame, numeric, categorical, train_idx, val_idx, test_idx, params)
        rows.extend(model_rows)
        predictions[name] = pred
        print(f"{name}: val AP={model_rows[0]['pr_auc']:.5f}, test AP={model_rows[1]['pr_auc']:.5f}", flush=True)

    print("Running multi-class and two-stage formulations", flush=True)
    model_rows, pred = fit_multiclass(
        "research_multiclass_readmitted", ultra_frame, prefix_num, prefix_cat, train_idx, val_idx, test_idx
    )
    rows.extend(model_rows)
    predictions["research_multiclass_readmitted"] = pred
    print(f"research_multiclass_readmitted: val AP={model_rows[0]['pr_auc']:.5f}, test AP={model_rows[1]['pr_auc']:.5f}", flush=True)

    model_rows, pred = fit_two_stage(
        "research_two_stage_any_then_early", ultra_frame, prefix_num, prefix_cat, train_idx, val_idx, test_idx
    )
    rows.extend(model_rows)
    predictions["research_two_stage_any_then_early"] = pred
    print(f"research_two_stage_any_then_early: val AP={model_rows[0]['pr_auc']:.5f}, test AP={model_rows[1]['pr_auc']:.5f}", flush=True)

    print("Running longitudinal history feature variants", flush=True)
    history_frame, hist_num, hist_cat = add_longitudinal_features(ultra_frame, include_target_history=False)
    hist_prefix_num = unique(prefix_num + hist_num)
    hist_prefix_cat = unique(prefix_cat + hist_cat)
    model_rows, pred = fit_binary_catboost(
        "research_history_without_target_labels", history_frame, hist_prefix_num, hist_prefix_cat,
        train_idx, val_idx, test_idx, BEST_PREFIX_PARAMS,
    )
    rows.extend(model_rows)
    predictions["research_history_without_target_labels"] = pred
    print(f"research_history_without_target_labels: val AP={model_rows[0]['pr_auc']:.5f}, test AP={model_rows[1]['pr_auc']:.5f}", flush=True)

    target_history_frame, target_hist_num, target_hist_cat = add_longitudinal_features(
        ultra_frame, include_target_history=True
    )
    target_hist_prefix_num = unique(prefix_num + target_hist_num)
    target_hist_prefix_cat = unique(prefix_cat + target_hist_cat)
    model_rows, pred = fit_binary_catboost(
        "research_longitudinal_target_history_sensitivity", target_history_frame,
        target_hist_prefix_num, target_hist_prefix_cat,
        train_idx, val_idx, test_idx, BEST_PREFIX_PARAMS,
    )
    rows.extend(model_rows)
    predictions["research_longitudinal_target_history_sensitivity"] = pred
    print(
        "research_longitudinal_target_history_sensitivity: "
        f"val AP={model_rows[0]['pr_auc']:.5f}, test AP={model_rows[1]['pr_auc']:.5f}",
        flush=True,
    )

    # A full-feature history check, because the best ensemble currently contains one full-feature branch.
    full_hist_num = unique(full_num + target_hist_num)
    full_hist_cat = unique(full_cat + target_hist_cat)
    full_mvs_params = {
        **{k: v for k, v in BASE_CAT_PARAMS.items() if k != "bagging_temperature"},
        "iterations": 520, "depth": 6, "learning_rate": 0.050,
        "l2_leaf_reg": 11.0, "random_strength": 0.20, "bootstrap_type": "MVS",
        "subsample": 0.88, "random_seed": 3407,
    }
    model_rows, pred = fit_binary_catboost(
        "research_full_mvs_target_history_sensitivity", target_history_frame,
        full_hist_num, full_hist_cat,
        train_idx, val_idx, test_idx, full_mvs_params,
    )
    rows.extend(model_rows)
    predictions["research_full_mvs_target_history_sensitivity"] = pred
    print(
        "research_full_mvs_target_history_sensitivity: "
        f"val AP={model_rows[0]['pr_auc']:.5f}, test AP={model_rows[1]['pr_auc']:.5f}",
        flush=True,
    )

    table = pd.DataFrame(rows).sort_values(["split", "pr_auc"], ascending=[True, False])
    table.to_csv(TAB / "31_research_hypothesis_models.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        RES / "research_hypothesis_probabilities.npz",
        y_val=ultra_frame.loc[val_idx, "target_30d"].to_numpy(),
        y_test=ultra_frame.loc[test_idx, "target_30d"].to_numpy(),
        **{f"val_{name}": value["val"] for name, value in predictions.items()},
        **{f"test_{name}": value["test"] for name, value in predictions.items()},
    )
    summary = {
        "runtime_seconds": time.time() - started_all,
        "best_validation": table[table["split"] == "validation"].iloc[0].to_dict(),
        "best_test": table[table["split"] == "test"].iloc[0].to_dict(),
    }
    (RES / "research_hypothesis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
