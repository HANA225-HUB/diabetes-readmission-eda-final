"""High-value follow-up experiments kept separate from the submitted report draft."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diabetes_readmission_project import (  # noqa: E402
    SEED,
    choose_f2_threshold,
    feature_sets,
    make_pipeline,
    metric_row,
    patient_level_split,
    prepare_data,
)


OUT = HERE / "bonus"
FIG = OUT / "figures"
TAB = OUT / "tables"
RES = OUT / "results"
for directory in (FIG, TAB, RES):
    directory.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font="Heiti TC", rc={"axes.unicode_minus": False})
COLORS = {"navy": "#245B78", "teal": "#2A9D8F", "gold": "#E9C46A", "coral": "#D2644A"}


def catboost_frame(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    frame = df[numeric + categorical].copy()
    for col in numeric:
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame[col] = values.fillna(values.median())
    for col in categorical:
        frame[col] = frame[col].astype("string").fillna("Missing").astype(str)
    return frame


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    total = len(y)
    ece = 0.0
    for index in range(bins):
        mask = ids == index
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(ece)


def calibration_diagnostics(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    model.fit(logits, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def probability_row(name: str, y: np.ndarray, probability: np.ndarray) -> dict:
    intercept, slope = calibration_diagnostics(y, probability)
    return {
        "method": name,
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "log_loss": log_loss(y, probability),
        "ece_10": expected_calibration_error(y, probability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_probability": float(probability.mean()),
        "observed_rate": float(y.mean()),
    }


def fit_catboost_candidates(df, train_idx, val_idx, test_idx):
    y_train = df.loc[train_idx, "target_30d"].to_numpy()
    y_val = df.loc[val_idx, "target_30d"].to_numpy()
    y_test = df.loc[test_idx, "target_30d"].to_numpy()
    candidates = [
        ("Cleaned", 6, 5.0),
        ("Cleaned", 8, 5.0),
        ("Cleaned", 6, 10.0),
        ("Engineered", 6, 5.0),
        ("Engineered", 8, 5.0),
        ("Engineered", 6, 10.0),
    ]
    rows = []
    fitted = {}
    frames = {}
    for feature_name, depth, l2 in candidates:
        numeric, categorical = feature_sets()[feature_name]
        if feature_name not in frames:
            frames[feature_name] = catboost_frame(df, numeric, categorical)
        frame = frames[feature_name]
        cat_positions = [frame.columns.get_loc(col) for col in categorical]
        model = CatBoostClassifier(
            iterations=700,
            depth=depth,
            learning_rate=0.045,
            l2_leaf_reg=l2,
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=SEED,
            random_strength=0.5,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            frame.loc[train_idx], y_train,
            cat_features=cat_positions,
            eval_set=(frame.loc[val_idx], y_val),
            early_stopping_rounds=70,
            verbose=False,
        )
        val_probability = model.predict_proba(frame.loc[val_idx])[:, 1]
        label = f"{feature_name}_d{depth}_l2{l2:g}"
        rows.append({
            "candidate": label,
            "feature_set": feature_name,
            "depth": depth,
            "l2_leaf_reg": l2,
            "best_iteration": model.get_best_iteration(),
            "validation_roc_auc": roc_auc_score(y_val, val_probability),
            "validation_pr_auc": average_precision_score(y_val, val_probability),
            "validation_brier": brier_score_loss(y_val, val_probability),
        })
        fitted[label] = (model, frame, categorical, val_probability)
        print(label, rows[-1])

    tuning = pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False)
    tuning.to_csv(TAB / "01_catboost_validation_tuning.csv", index=False, encoding="utf-8-sig")
    best_label = tuning.iloc[0]["candidate"]
    best_model, best_frame, categorical, val_probability = fitted[best_label]
    test_probability = best_model.predict_proba(best_frame.loc[test_idx])[:, 1]
    threshold, validation_f2 = choose_f2_threshold(y_val, val_probability)
    test_metrics = metric_row(y_test, test_probability, threshold)
    test_metrics.update({
        "candidate": best_label,
        "validation_f2": validation_f2,
        "best_iteration": int(best_model.get_best_iteration()),
    })
    pd.DataFrame([test_metrics]).to_csv(TAB / "02_catboost_test.csv", index=False, encoding="utf-8-sig")

    importance = pd.DataFrame({
        "feature": best_frame.columns,
        "importance": best_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(TAB / "03_catboost_importance.csv", index=False, encoding="utf-8-sig")
    return best_label, best_model, best_frame, val_probability, test_probability, y_val, y_test, test_metrics


def calibration_experiment(df, train_idx, val_idx, test_idx, cat_val, cat_test, y_val, y_test):
    numeric, categorical = feature_sets()["Cleaned"]
    columns = numeric + categorical
    hist = make_pipeline("HistGradientBoosting", numeric, categorical)
    hist.fit(df.loc[train_idx, columns], df.loc[train_idx, "target_30d"])
    hist_val = hist.predict_proba(df.loc[val_idx, columns])[:, 1]
    hist_test = hist.predict_proba(df.loc[test_idx, columns])[:, 1]

    rows = []
    calibrated = {}
    for model_name, val_probability, test_probability in [
        ("HistGB", hist_val, hist_test),
        ("CatBoost", cat_val, cat_test),
    ]:
        rows.append(probability_row(f"{model_name} uncalibrated", y_test, test_probability))
        val_clipped = np.clip(val_probability, 1e-6, 1 - 1e-6)
        test_clipped = np.clip(test_probability, 1e-6, 1 - 1e-6)
        platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        platt.fit(np.log(val_clipped / (1 - val_clipped)).reshape(-1, 1), y_val)
        platt_test = platt.predict_proba(np.log(test_clipped / (1 - test_clipped)).reshape(-1, 1))[:, 1]
        rows.append(probability_row(f"{model_name} Platt", y_test, platt_test))
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(val_probability, y_val)
        isotonic_test = isotonic.predict(test_probability)
        rows.append(probability_row(f"{model_name} isotonic", y_test, isotonic_test))
        calibrated[f"{model_name}_raw"] = test_probability
        calibrated[f"{model_name}_platt"] = platt_test
        calibrated[f"{model_name}_isotonic"] = isotonic_test

    calibration = pd.DataFrame(rows)
    calibration.to_csv(TAB / "04_calibration_comparison.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, model_name in zip(axes, ["HistGB", "CatBoost"]):
        for suffix, label, color in [
            ("raw", "未校准", COLORS["navy"]),
            ("platt", "Platt", COLORS["teal"]),
            ("isotonic", "Isotonic", COLORS["coral"]),
        ]:
            probability = calibrated[f"{model_name}_{suffix}"]
            bins = pd.qcut(probability, 10, duplicates="drop")
            curve = pd.DataFrame({"y": y_test, "p": probability, "bin": bins}).groupby("bin", observed=True).agg(
                observed=("y", "mean"), predicted=("p", "mean")
            )
            ax.plot(curve["predicted"], curve["observed"], marker="o", label=label, color=color)
        ax.plot([0, 0.35], [0, 0.35], ls="--", color="#64748B")
        ax.set_xlim(0, 0.35)
        ax.set_ylim(0, 0.35)
        ax.set_title(model_name, weight="bold")
        ax.set_xlabel("平均预测概率")
        ax.set_ylabel("实际发生率")
        ax.legend(frameon=False)
    fig.suptitle("校准方法对概率可信度的影响", weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "01_calibration_methods.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return hist_val, hist_test, calibrated, calibration


def learning_curve(df, train_idx, val_idx):
    numeric, categorical = feature_sets()["Cleaned"]
    columns = numeric + categorical
    train_df = df.loc[train_idx]
    patient_target = train_df.groupby("patient_nbr")["target_30d"].max()
    patients = patient_target.index.to_numpy()
    labels = patient_target.to_numpy()
    y_val = df.loc[val_idx, "target_30d"].to_numpy()
    rows = []
    for fraction in [0.20, 0.40, 0.60, 0.80, 1.00]:
        for repeat in range(3):
            if fraction == 1:
                selected = patients
            else:
                selected, _ = train_test_split(
                    patients,
                    train_size=fraction,
                    random_state=SEED + repeat,
                    stratify=labels,
                )
            subset_idx = train_df.index[train_df["patient_nbr"].isin(selected)]
            model = make_pipeline("HistGradientBoosting", numeric, categorical)
            model.fit(df.loc[subset_idx, columns], df.loc[subset_idx, "target_30d"])
            probability = model.predict_proba(df.loc[val_idx, columns])[:, 1]
            rows.append({
                "training_fraction": fraction,
                "repeat": repeat + 1,
                "patients": len(selected),
                "records": len(subset_idx),
                "validation_roc_auc": roc_auc_score(y_val, probability),
                "validation_pr_auc": average_precision_score(y_val, probability),
            })
            print("learning", rows[-1])
    result = pd.DataFrame(rows)
    result.to_csv(TAB / "05_learning_curve.csv", index=False, encoding="utf-8-sig")
    summary = result.groupby("training_fraction").agg(
        patients=("patients", "mean"),
        roc_mean=("validation_roc_auc", "mean"), roc_sd=("validation_roc_auc", "std"),
        pr_mean=("validation_pr_auc", "mean"), pr_sd=("validation_pr_auc", "std"),
    ).reset_index()
    summary.to_csv(TAB / "06_learning_curve_summary.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    for ax, mean, sd, label, color in [
        (axes[0], "roc_mean", "roc_sd", "ROC-AUC", COLORS["navy"]),
        (axes[1], "pr_mean", "pr_sd", "PR-AUC", COLORS["teal"]),
    ]:
        ax.errorbar(summary["patients"], summary[mean], yerr=summary[sd].fillna(0), marker="o", lw=2.2, color=color, capsize=3)
        ax.set_xlabel("训练患者数")
        ax.set_ylabel(label)
        ax.set_title(label, weight="bold")
    fig.suptitle("学习曲线：增加患者是否仍有收益", weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "02_learning_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def decision_curve(y_test, probabilities: dict[str, np.ndarray]):
    rows = []
    n = len(y_test)
    prevalence = y_test.mean()
    for threshold in np.linspace(0.02, 0.30, 57):
        all_benefit = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        rows.append({"model": "Treat all", "threshold": threshold, "net_benefit": all_benefit})
        rows.append({"model": "Treat none", "threshold": threshold, "net_benefit": 0.0})
        for name, probability in probabilities.items():
            predicted = probability >= threshold
            tp = np.sum(predicted & (y_test == 1))
            fp = np.sum(predicted & (y_test == 0))
            net_benefit = tp / n - fp / n * threshold / (1 - threshold)
            rows.append({"model": name, "threshold": threshold, "net_benefit": net_benefit})
    result = pd.DataFrame(rows)
    result.to_csv(TAB / "07_decision_curve.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    style = {
        "HistGB": (COLORS["navy"], "-"), "CatBoost": (COLORS["teal"], "-"),
        "Treat all": (COLORS["coral"], "--"), "Treat none": ("#64748B", ":"),
    }
    for name, group in result.groupby("model"):
        color, linestyle = style[name]
        ax.plot(group["threshold"], group["net_benefit"], label=name, color=color, ls=linestyle, lw=2)
    ax.set_ylim(-0.02, 0.12)
    ax.set_xlabel("风险阈值")
    ax.set_ylabel("净收益")
    ax.set_title("决策曲线：模型是否优于全干预/不干预", weight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "03_decision_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return result


def subgroup_threshold_audit(df, test_idx, y_test, probability, threshold):
    rows = []
    test = df.loc[test_idx]
    predicted = probability >= threshold
    for variable in ["gender", "race_clean", "age"]:
        for value in test[variable].astype(str).unique():
            mask = test[variable].astype(str).to_numpy() == value
            if mask.sum() < 100:
                continue
            rows.append({
                "variable": variable,
                "group": value,
                "n": int(mask.sum()),
                "positive_rate": float(y_test[mask].mean()),
                "roc_auc": roc_auc_score(y_test[mask], probability[mask]) if len(np.unique(y_test[mask])) == 2 else np.nan,
                "pr_auc": average_precision_score(y_test[mask], probability[mask]),
                "recall": recall_score(y_test[mask], predicted[mask], zero_division=0),
                "precision": precision_score(y_test[mask], predicted[mask], zero_division=0),
            })
    result = pd.DataFrame(rows)
    result.to_csv(TAB / "08_catboost_subgroups.csv", index=False, encoding="utf-8-sig")
    return result


def main():
    df, _ = prepare_data()
    train_idx, val_idx, test_idx, _ = patient_level_split(df)
    best_label, best_model, best_frame, cat_val, cat_test, y_val, y_test, cat_metrics = fit_catboost_candidates(
        df, train_idx, val_idx, test_idx
    )
    hist_val, hist_test, calibrated, calibration = calibration_experiment(
        df, train_idx, val_idx, test_idx, cat_val, cat_test, y_val, y_test
    )
    curve_summary = learning_curve(df, train_idx, val_idx)
    decision = decision_curve(y_test, {"HistGB": hist_test, "CatBoost": cat_test})
    subgroup = subgroup_threshold_audit(df, test_idx, y_test, cat_test, cat_metrics["threshold"])
    summary = {
        "best_catboost_candidate": best_label,
        "catboost_test": cat_metrics,
        "calibration": calibration.to_dict(orient="records"),
        "learning_curve": curve_summary.to_dict(orient="records"),
        "decision_curve_positive_threshold_range": [
            float(decision[(decision["model"] == "CatBoost") & (decision["net_benefit"] > 0)]["threshold"].min()),
            float(decision[(decision["model"] == "CatBoost") & (decision["net_benefit"] > 0)]["threshold"].max()),
        ],
        "catboost_subgroup_rows": int(len(subgroup)),
        "note": "encounter_id was not treated as a validated date field; no pseudo-temporal claim was made.",
    }
    (RES / "bonus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
