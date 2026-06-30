"""Validate optimization gains without modifying report artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diabetes_readmission_project import feature_sets, make_pipeline, patient_level_split, prepare_data  # noqa: E402
from optimization_lab import metrics, operating_points, topk_rows  # noqa: E402


IN = HERE / "optimization" / "results" / "optimized_probabilities.npz"
ENSEMBLE_IN = HERE / "optimization" / "results" / "ensemble_probabilities.npz"
TAB = HERE / "optimization" / "tables"
RES = HERE / "optimization" / "results"


def rank01(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def precision_targets(name, y_val, p_val, y_test, p_test):
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    rows = []
    for target in [0.16, 0.18, 0.20, 0.25, 0.30]:
        valid = np.flatnonzero(precision[:-1] >= target)
        if not len(valid):
            continue
        chosen = valid[np.argmax(recall[:-1][valid])]
        threshold = thresholds[chosen]
        predicted = p_test >= threshold
        rows.append({
            "model": name, "validation_precision_target": target, "threshold": threshold,
            "test_precision": precision_score(y_test, predicted, zero_division=0),
            "test_recall": recall_score(y_test, predicted, zero_division=0),
            "selected_fraction": predicted.mean(),
        })
    return rows


def patient_bootstrap(y, groups, probabilities, repeats=1000):
    rng = np.random.default_rng(20260625)
    patients = np.unique(groups)
    locations = {patient: np.flatnonzero(groups == patient) for patient in patients}
    rows = []
    for repeat in range(repeats):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([locations[patient] for patient in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        row = {"repeat": repeat + 1}
        for name, p in probabilities.items():
            yy, pp = y[idx], p[idx]
            row[f"pr_{name}"] = average_precision_score(yy, pp)
            row[f"roc_{name}"] = roc_auc_score(yy, pp)
            k = max(1, int(np.ceil(len(yy) * 0.05)))
            top = np.argsort(-pp)[:k]
            row[f"top5_precision_{name}"] = yy[top].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_delta(bootstrap, better, baseline, metric):
    delta = bootstrap[f"{metric}_{better}"] - bootstrap[f"{metric}_{baseline}"]
    return {
        "mean": float(delta.mean()),
        "ci95": [float(delta.quantile(0.025)), float(delta.quantile(0.975))],
        "positive_probability": float((delta > 0).mean()),
    }


def main():
    saved = np.load(IN)
    y_val, y_test = saved["y_val"], saved["y_test"]
    cat_val, cat_test = saved["val_CatBoost_Rich"], saved["test_CatBoost_Rich"]
    lgb_val, lgb_test = saved["val_LightGBM_Rich"], saved["test_LightGBM_Rich"]
    xgb_val, xgb_test = saved["val_XGBoost_Rich"], saved["test_XGBoost_Rich"]
    clean_val, clean_test = saved["val_HistGB_baseline"], saved["test_HistGB_baseline"]

    blend_rows = []
    best = None
    for w_xgb in np.arange(0, 0.201, 0.02):
        remaining = 1 - w_xgb
        for w_cat in np.arange(0, remaining + 1e-9, 0.01):
            w_lgb = remaining - w_cat
            p = w_cat * cat_val + w_lgb * lgb_val + w_xgb * xgb_val
            row = {
                "w_catboost": w_cat, "w_lightgbm": w_lgb, "w_xgboost": w_xgb,
                "validation_pr_auc": average_precision_score(y_val, p),
                "validation_roc_auc": roc_auc_score(y_val, p),
            }
            blend_rows.append(row)
            if best is None or row["validation_pr_auc"] > best["validation_pr_auc"]:
                best = row
    blends = pd.DataFrame(blend_rows).sort_values("validation_pr_auc", ascending=False)
    blends.to_csv(TAB / "07_refined_blend.csv", index=False, encoding="utf-8-sig")
    refined_val = best["w_catboost"] * cat_val + best["w_lightgbm"] * lgb_val + best["w_xgboost"] * xgb_val
    refined_test = best["w_catboost"] * cat_test + best["w_lightgbm"] * lgb_test + best["w_xgboost"] * xgb_test

    # Recreate the strongest original-ranking baseline on the unchanged split.
    df, _ = prepare_data()
    train_idx, val_idx, test_idx, _ = patient_level_split(df)
    raw_num, raw_cat = feature_sets()["Raw"]
    raw_cols = raw_num + raw_cat
    raw_model = make_pipeline("HistGradientBoosting", raw_num, raw_cat)
    raw_model.fit(df.loc[train_idx, raw_cols], df.loc[train_idx, "target_30d"])
    raw_test = raw_model.predict_proba(df.loc[test_idx, raw_cols])[:, 1]
    groups = df.loc[test_idx, "patient_nbr"].to_numpy()

    probability = {
        "histgb_clean": clean_test,
        "histgb_raw": raw_test,
        "catboost_rich": cat_test,
        "refined_blend": refined_test,
    }
    final_val = final_test = None
    if ENSEMBLE_IN.exists():
        ensemble = np.load(ENSEMBLE_IN)
        final_val, final_test = ensemble["blend_val"], ensemble["blend_test"]
        probability["final_ensemble_blend"] = final_test
    bootstrap = patient_bootstrap(y_test, groups, probability)
    bootstrap.to_csv(TAB / "08_optimization_bootstrap.csv", index=False, encoding="utf-8-sig")

    target_rows = []
    for name, p_val, p_test in [
        ("HistGB Cleaned", clean_val, clean_test),
        ("CatBoost Rich", cat_val, cat_test),
        ("Refined blend", refined_val, refined_test),
    ] + ([] if final_val is None else [("Final ensemble blend", final_val, final_test)]):
        target_rows.extend(precision_targets(name, y_val, p_val, y_test, p_test))
    pd.DataFrame(target_rows).to_csv(TAB / "09_precision_targets.csv", index=False, encoding="utf-8-sig")

    final_rows = []
    final_operating_rows = []
    final_topk_rows = []
    candidates = [
        ("HistGB Cleaned", clean_val, clean_test),
        ("CatBoost Rich", cat_val, cat_test),
        ("Refined blend", refined_val, refined_test),
    ] + ([] if final_val is None else [("Final ensemble blend", final_val, final_test)])
    for name, p_val, p_test in candidates:
        threshold_row = metrics(name, y_val, p_val)
        final_rows.append(metrics(name, y_test, p_test, threshold_row["threshold"]))
        final_operating_rows.extend(operating_points(name, y_val, p_val, y_test, p_test))
        final_topk_rows.extend(topk_rows(name, y_test, p_test))
    pd.DataFrame(final_rows).to_csv(TAB / "14_final_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(final_operating_rows).to_csv(TAB / "15_final_operating_points.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(final_topk_rows).to_csv(TAB / "16_final_topk_capacity.csv", index=False, encoding="utf-8-sig")

    summary = {
        "refined_blend": best,
        "refined_validation": {
            "pr_auc": average_precision_score(y_val, refined_val),
            "roc_auc": roc_auc_score(y_val, refined_val),
        },
        "refined_test": {
            "pr_auc": average_precision_score(y_test, refined_test),
            "roc_auc": roc_auc_score(y_test, refined_test),
        },
        "point_metrics": {
            name: {"pr_auc": average_precision_score(y_test, p), "roc_auc": roc_auc_score(y_test, p)}
            for name, p in probability.items()
        },
        "bootstrap_deltas": {},
    }
    better_models = ["catboost_rich", "refined_blend"]
    if final_test is not None:
        better_models.append("final_ensemble_blend")
    for better in better_models:
        for baseline in ["histgb_clean", "histgb_raw"]:
            for metric in ["pr", "roc", "top5_precision"]:
                summary["bootstrap_deltas"][f"{metric}_{better}_vs_{baseline}"] = summarize_delta(
                    bootstrap, better, baseline, metric
                )
    (RES / "optimization_validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
