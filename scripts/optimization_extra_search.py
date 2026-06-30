"""Extra optimization pass for the diabetes readmission model.

This script is intentionally isolated from report generation. It tries
additional non-leaky feature variants, CatBoost configurations, native
LightGBM categorical handling, and validation-selected blends.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bonus_experiments import catboost_frame  # noqa: E402
from diabetes_readmission_project import SEED, patient_level_split, prepare_data  # noqa: E402
from optimization_lab import add_rich_features, rank01  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"
for directory in (TAB, RES):
    directory.mkdir(parents=True, exist_ok=True)


BASE_CAT_PARAMS = {
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


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def diag_numeric(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "?"} or text.startswith(("V", "E")):
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def diag_prefix(value: object, digits: int) -> str:
    if pd.isna(value):
        return "Missing"
    text = str(value).strip()
    if text in {"", "?"}:
        return "Missing"
    if text.startswith(("V", "E")):
        return text[: min(len(text), digits + 1)]
    numeric = "".join(ch for ch in text if ch.isdigit())
    return numeric[:digits] if numeric else "Other"


def diabetes_complication(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    text = str(value).strip()
    if not text.startswith("250"):
        return "Not diabetes"
    if "." not in text:
        return "Diabetes unspecified"
    tail = text.split(".", 1)[1]
    return f"Diabetes .{tail[:1]}" if tail else "Diabetes unspecified"


def add_ultra_features(df: pd.DataFrame):
    frame, numeric, categorical = add_rich_features(df)

    for idx in (1, 2, 3):
        raw = f"diag_{idx}"
        frame[f"{raw}_numeric"] = frame[raw].map(diag_numeric)
        frame[f"{raw}_prefix2"] = frame[raw].map(lambda value: diag_prefix(value, 2))
        frame[f"{raw}_prefix3"] = frame[raw].map(lambda value: diag_prefix(value, 3))
        frame[f"{raw}_diabetes_detail"] = frame[raw].map(diabetes_complication)
        numeric.append(f"{raw}_numeric")
        categorical.extend([f"{raw}_prefix2", f"{raw}_prefix3", f"{raw}_diabetes_detail"])

    diag_groups = ["diag1_group", "diag2_group", "diag3_group"]
    for group in [
        "Diabetes", "Circulatory", "Respiratory", "Digestive",
        "Genitourinary", "Injury/Poisoning", "Supplementary/External",
    ]:
        col = f"diag_count_{group.lower().replace('/', '_').replace(' ', '_')}"
        frame[col] = frame[diag_groups].eq(group).sum(axis=1)
        numeric.append(col)

    frame["any_diabetes_dx"] = frame[[f"diag_{i}_diabetes_detail" for i in (1, 2, 3)]].ne("Not diabetes").sum(axis=1)
    frame["same_primary_secondary_group"] = (frame["diag1_group"] == frame["diag2_group"]).astype(int)
    frame["all_diag_groups_distinct"] = (frame[diag_groups].nunique(axis=1) == 3).astype(int)
    numeric.extend(["any_diabetes_dx", "same_primary_secondary_group", "all_diag_groups_distinct"])

    frame["diag_group_pair_12"] = frame["diag1_group"].astype(str) + "|" + frame["diag2_group"].astype(str)
    frame["diag_group_pair_13"] = frame["diag1_group"].astype(str) + "|" + frame["diag3_group"].astype(str)
    frame["diag_prefix_pair_12"] = frame["diag_1_prefix3"].astype(str) + "|" + frame["diag_2_prefix3"].astype(str)
    frame["primary_dx_discharge"] = frame["diag1_group"].astype(str) + "|" + frame["discharge_group"].astype(str)
    frame["primary_dx_admission_source"] = frame["diag1_group"].astype(str) + "|" + frame["admission_source_group"].astype(str)
    frame["raw_pathway"] = (
        frame["admission_type_raw"].astype(str)
        + "|"
        + frame["admission_source_raw"].astype(str)
        + "|"
        + frame["discharge_disposition_raw"].astype(str)
    )
    frame["insulin_change"] = frame["insulin"].astype(str) + "|" + frame["change"].astype(str)
    frame["a1c_insulin"] = frame["A1Cresult"].astype(str) + "|" + frame["insulin"].astype(str)
    frame["medication_load_band"] = pd.cut(
        frame["active_med_count"], [-1, 0, 1, 3, 6, np.inf],
        labels=["0", "1", "2-3", "4-6", "7+"],
    ).astype("string").fillna("Missing")
    frame["acute_visit_band"] = pd.cut(
        frame["number_inpatient"] + frame["number_emergency"],
        [-1, 0, 1, 2, 4, np.inf],
        labels=["0", "1", "2", "3-4", "5+"],
    ).astype("string").fillna("Missing")
    frame["primary_dx_acute_band"] = frame["diag1_group"].astype(str) + "|" + frame["acute_visit_band"].astype(str)
    categorical.extend([
        "diag_group_pair_12", "diag_group_pair_13", "diag_prefix_pair_12",
        "primary_dx_discharge", "primary_dx_admission_source", "raw_pathway",
        "insulin_change", "a1c_insulin", "medication_load_band", "acute_visit_band",
        "primary_dx_acute_band",
    ])

    frame["acute_visits_total"] = frame["number_inpatient"] + frame["number_emergency"]
    frame["outpatient_share"] = frame["number_outpatient"] / frame["prior_visits_total"].clip(lower=1)
    frame["acute_share"] = frame["acute_visits_total"] / frame["prior_visits_total"].clip(lower=1)
    frame["procedures_per_medication"] = frame["num_procedures"] / frame["num_medications"].clip(lower=1)
    frame["medications_per_lab"] = frame["num_medications"] / frame["num_lab_procedures"].clip(lower=1)
    frame["complexity_x_acute"] = frame["care_intensity"] * np.log1p(frame["acute_visits_total"])
    frame["age_x_active_meds"] = frame["age_mid"] * frame["active_med_count"]
    numeric.extend([
        "acute_visits_total", "outpatient_share", "acute_share", "procedures_per_medication",
        "medications_per_lab", "complexity_x_acute", "age_x_active_meds",
    ])

    return frame, unique(numeric), unique(categorical)


def feature_variant(frame: pd.DataFrame, numeric: list[str], categorical: list[str], variant: str):
    if variant == "ultra_full":
        return numeric, categorical
    if variant == "ultra_prefix_no_raw_diag":
        remove = {"diag_1", "diag_2", "diag_3"}
        return numeric, [col for col in categorical if col not in remove]
    if variant == "ultra_less_sparse":
        remove = {
            "diag_1", "diag_2", "diag_3", "diag_prefix_pair_12",
            "raw_pathway", "medical_specialty", "payer_code",
        }
        return [col for col in numeric if not col.endswith("_numeric")], [col for col in categorical if col not in remove]
    raise ValueError(f"Unknown variant: {variant}")


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


def cat_configs():
    return [
        ("base_seed3407", {**BASE_CAT_PARAMS, "random_seed": 3407}),
        ("base_seed2026", {**BASE_CAT_PARAMS, "random_seed": 2026}),
        ("slow_depth6", {
            **BASE_CAT_PARAMS, "iterations": 520, "depth": 6, "learning_rate": 0.046,
            "l2_leaf_reg": 9.0, "random_strength": 0.25, "bagging_temperature": 0.55,
            "random_seed": 3407,
        }),
        ("regularized_depth5", {
            **BASE_CAT_PARAMS, "iterations": 620, "depth": 5, "learning_rate": 0.038,
            "l2_leaf_reg": 24.0, "random_strength": 0.45, "bagging_temperature": 0.75,
            "random_seed": 3407,
        }),
        ("bernoulli_depth6", {
            **{k: v for k, v in BASE_CAT_PARAMS.items() if k != "bagging_temperature"},
            "iterations": 520, "depth": 6, "learning_rate": 0.050, "l2_leaf_reg": 10.5,
            "random_strength": 0.18, "bootstrap_type": "Bernoulli", "subsample": 0.82,
            "random_seed": 3407,
        }),
        ("mvs_depth6", {
            **{k: v for k, v in BASE_CAT_PARAMS.items() if k != "bagging_temperature"},
            "iterations": 520, "depth": 6, "learning_rate": 0.050, "l2_leaf_reg": 11.0,
            "random_strength": 0.20, "bootstrap_type": "MVS", "subsample": 0.88,
            "random_seed": 3407,
        }),
        ("positive_weight_125", {
            **BASE_CAT_PARAMS, "random_seed": 42, "class_weights": [1.0, 1.25],
        }),
        ("auto_sqrt_balanced", {
            **BASE_CAT_PARAMS, "random_seed": 42, "auto_class_weights": "SqrtBalanced",
        }),
    ]


def fit_catboost_candidates(frame, numeric, categorical, train_idx, val_idx, test_idx):
    y_train = frame.loc[train_idx, "target_30d"].to_numpy()
    y_val = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = frame.loc[test_idx, "target_30d"].to_numpy()
    rows = []
    predictions = {}

    for variant in ["ultra_full", "ultra_prefix_no_raw_diag", "ultra_less_sparse"]:
        variant_num, variant_cat = feature_variant(frame, numeric, categorical, variant)
        features = catboost_frame(frame, variant_num, variant_cat)
        cat_positions = [features.columns.get_loc(col) for col in variant_cat]
        for config_name, params in cat_configs():
            name = f"cat_{variant}_{config_name}"
            started = time.time()
            model = CatBoostClassifier(**params)
            model.fit(features.loc[train_idx], y_train, cat_features=cat_positions, verbose=False)
            runtime = time.time() - started
            val_probability = model.predict_proba(features.loc[val_idx])[:, 1]
            test_probability = model.predict_proba(features.loc[test_idx])[:, 1]
            predictions[name] = (val_probability, test_probability)
            rows.append(score(
                name, "validation", y_val, val_probability,
                model_type="catboost", variant=variant, config=config_name,
                runtime_seconds=runtime, numeric_features=len(variant_num),
                categorical_features=len(variant_cat),
            ))
            rows.append(score(
                name, "test", y_test, test_probability,
                model_type="catboost", variant=variant, config=config_name,
                runtime_seconds=runtime, numeric_features=len(variant_num),
                categorical_features=len(variant_cat),
            ))
            print(
                f"{name}: val AP={rows[-2]['pr_auc']:.5f}, "
                f"test AP={rows[-1]['pr_auc']:.5f}, {runtime:.1f}s",
                flush=True,
            )

    return rows, predictions


def lgb_native_frame(frame, numeric, categorical, train_idx):
    result = frame[numeric + categorical].copy()
    for column in numeric:
        values = pd.to_numeric(result.loc[train_idx, column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        result[column] = (
            pd.to_numeric(result[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(values.median())
            .astype("float32")
        )
    for column in categorical:
        train_values = result.loc[train_idx, column].astype("string").fillna("Missing")
        categories = pd.Index(train_values.unique())
        result[column] = pd.Categorical(result[column].astype("string").fillna("Missing"), categories=categories)
    return result


def fit_lightgbm_native(frame, numeric, categorical, train_idx, val_idx, test_idx):
    y_train = frame.loc[train_idx, "target_30d"].to_numpy()
    y_val = frame.loc[val_idx, "target_30d"].to_numpy()
    y_test = frame.loc[test_idx, "target_30d"].to_numpy()
    variant_num, variant_cat = feature_variant(frame, numeric, categorical, "ultra_prefix_no_raw_diag")
    columns = variant_num + variant_cat
    model_frame = lgb_native_frame(frame, variant_num, variant_cat, train_idx)
    rows = []
    predictions = {}
    configs = [
        ("native_existing", dict(
            learning_rate=0.023945814295799875, num_leaves=109, max_depth=9,
            min_child_samples=82, subsample=0.8321560753663861,
            colsample_bytree=0.6526275909951529, reg_alpha=0.021363862556460317,
            reg_lambda=0.22023993238731142,
        )),
        ("native_regularized", dict(
            learning_rate=0.030, num_leaves=63, max_depth=8,
            min_child_samples=120, subsample=0.85, colsample_bytree=0.75,
            reg_alpha=0.05, reg_lambda=3.0,
        )),
        ("native_leaf31", dict(
            learning_rate=0.040, num_leaves=31, max_depth=6,
            min_child_samples=80, subsample=0.90, colsample_bytree=0.85,
            reg_alpha=0.02, reg_lambda=1.5,
        )),
    ]
    for config_name, params in configs:
        name = f"lgb_{config_name}_ultra_prefix"
        started = time.time()
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=1800, verbosity=-1,
            random_state=SEED, n_jobs=-1, **params,
        )
        model.fit(
            model_frame.loc[train_idx, columns], y_train,
            eval_set=[(model_frame.loc[val_idx, columns], y_val)],
            eval_metric="average_precision",
            categorical_feature=variant_cat,
            callbacks=[lgb.early_stopping(90, verbose=False)],
        )
        runtime = time.time() - started
        val_probability = model.predict_proba(model_frame.loc[val_idx, columns])[:, 1]
        test_probability = model.predict_proba(model_frame.loc[test_idx, columns])[:, 1]
        predictions[name] = (val_probability, test_probability)
        rows.append(score(
            name, "validation", y_val, val_probability,
            model_type="lightgbm_native", variant="ultra_prefix_no_raw_diag",
            config=config_name, runtime_seconds=runtime,
            numeric_features=len(variant_num), categorical_features=len(variant_cat),
        ))
        rows.append(score(
            name, "test", y_test, test_probability,
            model_type="lightgbm_native", variant="ultra_prefix_no_raw_diag",
            config=config_name, runtime_seconds=runtime,
            numeric_features=len(variant_num), categorical_features=len(variant_cat),
        ))
        print(
            f"{name}: val AP={rows[-2]['pr_auc']:.5f}, "
            f"test AP={rows[-1]['pr_auc']:.5f}, {runtime:.1f}s",
            flush=True,
        )
    return rows, predictions


def load_existing_candidates():
    candidates = {}
    ensemble = np.load(RES / "ensemble_probabilities.npz")
    candidates["final_tree_ensemble"] = (ensemble["blend_val"], ensemble["blend_test"])
    candidates["selected_catboost"] = (ensemble["selected_val"], ensemble["selected_test"])

    optimized = np.load(RES / "optimized_probabilities.npz")
    for name in ["CatBoost_Rich", "LightGBM_Rich", "XGBoost_Rich", "Optimized_blend"]:
        val_key = f"val_{name}"
        test_key = f"test_{name}"
        if val_key in optimized.files and test_key in optimized.files:
            candidates[name] = (optimized[val_key], optimized[test_key])
    return candidates, ensemble["y_val"], ensemble["y_test"]


def blend_search(y_val, y_test, candidates, new_predictions):
    all_candidates = {**candidates, **new_predictions}
    rows = []

    base_val, base_test = candidates["final_tree_ensemble"]
    for name, (val_probability, test_probability) in new_predictions.items():
        for method in ["probability", "rank"]:
            for weight in np.arange(0.0, 0.501, 0.01):
                if method == "probability":
                    val_blend = (1 - weight) * base_val + weight * val_probability
                    test_blend = (1 - weight) * base_test + weight * test_probability
                else:
                    val_blend = (1 - weight) * rank01(base_val) + weight * rank01(val_probability)
                    test_blend = (1 - weight) * rank01(base_test) + weight * rank01(test_probability)
                rows.append({
                    "blend": "pairwise_with_final", "method": method, "candidate": name,
                    "weight": weight,
                    "validation_pr_auc": average_precision_score(y_val, val_blend),
                    "validation_roc_auc": roc_auc_score(y_val, val_blend),
                    "test_pr_auc": average_precision_score(y_test, test_blend),
                    "test_roc_auc": roc_auc_score(y_test, test_blend),
                    "test_brier": brier_score_loss(y_test, np.clip(test_blend, 0, 1)),
                })

    validation_scores = [
        (name, average_precision_score(y_val, probability[0]))
        for name, probability in all_candidates.items()
    ]
    top_names = [name for name, _ in sorted(validation_scores, key=lambda item: item[1], reverse=True)[:8]]

    rng = np.random.default_rng(20260625)
    for method in ["probability", "rank"]:
        for size in [3, 4, 5]:
            for subset in combinations(top_names, size):
                for draw in range(180):
                    weights = rng.dirichlet(np.ones(size))
                    val_parts = [all_candidates[name][0] for name in subset]
                    test_parts = [all_candidates[name][1] for name in subset]
                    if method == "rank":
                        val_parts = [rank01(part) for part in val_parts]
                        test_parts = [rank01(part) for part in test_parts]
                    val_blend = sum(weight * part for weight, part in zip(weights, val_parts))
                    test_blend = sum(weight * part for weight, part in zip(weights, test_parts))
                    rows.append({
                        "blend": f"dirichlet_top{size}", "method": method,
                        "candidate": "+".join(subset),
                        "weight": json.dumps({name: float(weight) for name, weight in zip(subset, weights)}),
                        "draw": draw,
                        "validation_pr_auc": average_precision_score(y_val, val_blend),
                        "validation_roc_auc": roc_auc_score(y_val, val_blend),
                        "test_pr_auc": average_precision_score(y_test, test_blend),
                        "test_roc_auc": roc_auc_score(y_test, test_blend),
                        "test_brier": brier_score_loss(y_test, np.clip(test_blend, 0, 1)),
                    })

    table = pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False)
    return table


def main():
    started = time.time()
    frame, _ = prepare_data()
    frame, numeric, categorical = add_ultra_features(frame)
    train_idx, val_idx, test_idx, _ = patient_level_split(frame)

    print(f"Ultra features: {len(numeric)} numeric, {len(categorical)} categorical", flush=True)
    cat_rows, cat_predictions = fit_catboost_candidates(frame, numeric, categorical, train_idx, val_idx, test_idx)
    lgb_rows, lgb_predictions = fit_lightgbm_native(frame, numeric, categorical, train_idx, val_idx, test_idx)

    model_table = pd.DataFrame(cat_rows + lgb_rows).sort_values(["split", "pr_auc"], ascending=[True, False])
    model_table.to_csv(TAB / "21_extra_model_search.csv", index=False, encoding="utf-8-sig")

    existing, y_val, y_test = load_existing_candidates()
    new_predictions = {**cat_predictions, **lgb_predictions}
    blend_table = blend_search(y_val, y_test, existing, new_predictions)
    blend_table.to_csv(TAB / "22_extra_blend_search.csv", index=False, encoding="utf-8-sig")

    best_model_validation = (
        model_table[model_table["split"] == "validation"]
        .sort_values("pr_auc", ascending=False)
        .iloc[0]
        .to_dict()
    )
    best_model_test = model_table[
        (model_table["split"] == "test")
        & (model_table["candidate"] == best_model_validation["candidate"])
    ].iloc[0].to_dict()
    best_blend = blend_table.iloc[0].to_dict()

    np.savez_compressed(
        RES / "extra_probabilities.npz",
        y_val=y_val,
        y_test=y_test,
        **{f"val_{name}": value[0] for name, value in new_predictions.items()},
        **{f"test_{name}": value[1] for name, value in new_predictions.items()},
    )
    summary = {
        "runtime_seconds": time.time() - started,
        "feature_counts": {"numeric": len(numeric), "categorical": len(categorical)},
        "best_single_by_validation": best_model_validation,
        "corresponding_test_for_best_single": best_model_test,
        "best_blend_by_validation": best_blend,
    }
    (RES / "extra_optimization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
