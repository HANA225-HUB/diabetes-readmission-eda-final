from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


SEED = 42
ROOT = Path("/Users/huahaowen/Documents/Codex/2026-06-24/6-30-23-59-1-pdf")
DATA_DIR = Path("/Users/huahaowen/Downloads/期末大作业")
WORK = ROOT / "work" / "eda_final"
FIG = WORK / "figures"
TAB = WORK / "tables"
RES = WORK / "results"
for directory in (FIG, TAB, RES):
    directory.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "navy": "#22577A",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "coral": "#D2644A",
    "ink": "#263238",
    "muted": "#64748B",
    "light": "#E8EEF2",
}

FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"
font_manager.fontManager.addfont(FONT_PATH)
CHINESE_FONT = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [CHINESE_FONT, "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 140,
    "savefig.dpi": 240,
    "axes.titlesize": 13,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})
sns.set_theme(
    style="whitegrid",
    font=CHINESE_FONT,
    font_scale=0.95,
    rc={"font.sans-serif": [CHINESE_FONT, "DejaVu Sans"], "axes.unicode_minus": False},
)


MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

BASE_NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
BASE_CATEGORICAL = [
    "race", "gender", "age", "admission_type_raw", "discharge_disposition_raw",
    "admission_source_raw", "max_glu_serum", "A1Cresult", "change", "diabetesMed",
] + MEDICATION_COLUMNS


def save_figure(fig: plt.Figure, name: str) -> None:
    path = FIG / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_string(series: pd.Series, missing_label: str = "Missing") -> pd.Series:
    return series.astype("string").replace({"?": pd.NA, "None": pd.NA}).fillna(missing_label)


def age_midpoint(value: str) -> float:
    if not isinstance(value, str) or "-" not in value:
        return np.nan
    text = value.replace("[", "").replace(")", "")
    lo, hi = text.split("-")
    return (float(lo) + float(hi)) / 2.0


def diagnosis_group(value: object) -> str:
    if pd.isna(value) or value == "?":
        return "Missing"
    text = str(value).strip()
    if text.startswith(("V", "E")):
        return "Supplementary/External"
    try:
        code = float(text)
    except ValueError:
        return "Other"
    if 250 <= code < 251:
        return "Diabetes"
    if (390 <= code <= 459) or code == 785:
        return "Circulatory"
    if (460 <= code <= 519) or code == 786:
        return "Respiratory"
    if (520 <= code <= 579) or code == 787:
        return "Digestive"
    if 140 <= code <= 239:
        return "Neoplasms"
    if (580 <= code <= 629) or code == 788:
        return "Genitourinary"
    if 710 <= code <= 739:
        return "Musculoskeletal"
    if 800 <= code <= 999:
        return "Injury/Poisoning"
    return "Other"


def admission_type_group(code: int) -> str:
    return {
        1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 7: "Trauma Center"
    }.get(int(code), "Unknown/Other")


def admission_source_group(code: int) -> str:
    code = int(code)
    if code == 7:
        return "Emergency Room"
    if code in {1, 2, 3}:
        return "Referral"
    if code in {4, 5, 6, 10, 18, 22, 25, 26}:
        return "Transfer"
    if code in {9, 15, 17, 20, 21}:
        return "Birth/Neonatal"
    return "Other/Unknown"


def discharge_group(code: int) -> str:
    code = int(code)
    if code == 1:
        return "Home"
    if code in {6, 8}:
        return "Home with support"
    if code in {2, 3, 4, 5, 9, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30}:
        return "Facility/Transfer"
    if code == 7:
        return "Left AMA"
    return "Other/Unknown"


def top_or_other(series: pd.Series, top_n: int = 12) -> pd.Series:
    cleaned = clean_string(series)
    top = cleaned[cleaned != "Missing"].value_counts().head(top_n).index
    return cleaned.where(cleaned.isin(top) | (cleaned == "Missing"), "Other specialty")


def prepare_data() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DATA_DIR / "diabetic_data.csv")
    original_rows = len(raw)
    question_missing = (raw.astype("string") == "?").sum().sort_values(ascending=False)
    actual_missing = raw.isna().sum()
    missing = pd.DataFrame({
        "question_mark_missing": question_missing,
        "actual_na": actual_missing,
    })
    missing["missing_total"] = missing.sum(axis=1)
    missing["missing_rate"] = missing["missing_total"] / len(raw)
    missing = missing.sort_values("missing_rate", ascending=False)
    missing.to_csv(TAB / "01_missingness.csv", encoding="utf-8-sig")

    death_hospice_codes = {11, 13, 14, 19, 20, 21}
    excluded_death_hospice = raw["discharge_disposition_id"].isin(death_hospice_codes).sum()
    excluded_invalid_gender = (raw["gender"] == "Unknown/Invalid").sum()
    df = raw.loc[
        ~raw["discharge_disposition_id"].isin(death_hospice_codes)
        & (raw["gender"] != "Unknown/Invalid")
    ].copy()

    df["target_30d"] = (df["readmitted"] == "<30").astype(int)
    df["admission_type_raw"] = df["admission_type_id"].astype(str)
    df["discharge_disposition_raw"] = df["discharge_disposition_id"].astype(str)
    df["admission_source_raw"] = df["admission_source_id"].astype(str)
    for col in ["race", "gender", "age", "max_glu_serum", "A1Cresult", "change", "diabetesMed"] + MEDICATION_COLUMNS:
        df[col] = clean_string(df[col])

    df["age_mid"] = df["age"].map(age_midpoint)
    df["race_clean"] = clean_string(df["race"])
    df["medical_specialty_group"] = top_or_other(df["medical_specialty"], 12)
    df["payer_available"] = np.where(df["payer_code"].astype("string").eq("?"), "Missing", "Recorded")
    df["admission_type_group"] = df["admission_type_id"].map(admission_type_group)
    df["admission_source_group"] = df["admission_source_id"].map(admission_source_group)
    df["discharge_group"] = df["discharge_disposition_id"].map(discharge_group)

    for idx in (1, 2, 3):
        df[f"diag{idx}_group"] = df[f"diag_{idx}"].map(diagnosis_group)
    df["primary_diabetes_dx"] = (df["diag1_group"] == "Diabetes").astype(int)
    df["diagnosis_diversity"] = df[["diag1_group", "diag2_group", "diag3_group"]].nunique(axis=1)

    medication = df[MEDICATION_COLUMNS]
    df["active_med_count"] = medication.ne("No").sum(axis=1)
    df["dose_change_count"] = medication.isin(["Up", "Down"]).sum(axis=1)
    df["insulin_active"] = df["insulin"].ne("No").astype(int)
    df["medication_changed"] = df["change"].eq("Ch").astype(int)
    df["diabetes_med_active"] = df["diabetesMed"].eq("Yes").astype(int)

    df["prior_visits_total"] = df[["number_outpatient", "number_emergency", "number_inpatient"]].sum(axis=1)
    df["prior_inpatient_flag"] = (df["number_inpatient"] > 0).astype(int)
    df["prior_emergency_flag"] = (df["number_emergency"] > 0).astype(int)
    df["prior_outpatient_flag"] = (df["number_outpatient"] > 0).astype(int)
    days = df["time_in_hospital"].clip(lower=1)
    df["labs_per_day"] = df["num_lab_procedures"] / days
    df["medications_per_day"] = df["num_medications"] / days
    df["procedures_per_day"] = df["num_procedures"] / days
    df["diagnoses_per_day"] = df["number_diagnoses"] / days
    df["care_intensity"] = (
        df["num_lab_procedures"] + 2 * df["num_procedures"] + df["num_medications"]
    ) / days

    df["A1C_tested"] = df["A1Cresult"].ne("Missing").astype(int)
    df["A1C_abnormal"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)
    df["glucose_tested"] = df["max_glu_serum"].ne("Missing").astype(int)
    df["glucose_abnormal"] = df["max_glu_serum"].isin([">200", ">300"]).astype(int)

    repeat_counts = df.groupby("patient_nbr").size()
    df["patient_encounter_count"] = df["patient_nbr"].map(repeat_counts)

    audit = {
        "original_rows": int(original_rows),
        "columns": int(raw.shape[1]),
        "excluded_death_hospice": int(excluded_death_hospice),
        "excluded_invalid_gender": int(excluded_invalid_gender),
        "analysis_rows": int(len(df)),
        "unique_patients": int(df["patient_nbr"].nunique()),
        "positive_count": int(df["target_30d"].sum()),
        "positive_rate": float(df["target_30d"].mean()),
        "repeat_patient_rate": float((repeat_counts > 1).mean()),
        "max_encounters_per_patient": int(repeat_counts.max()),
        "top_missing": {
            k: float(v) for k, v in missing["missing_rate"].head(8).items()
        },
    }
    (RES / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, audit


def make_eda_figures(df: pd.DataFrame, audit: dict) -> None:
    counts = df["target_30d"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    bars = ax.bar(["未在 30 天内再入院", "30 天内再入院"], counts.values,
                  color=[PALETTE["navy"], PALETTE["coral"]], width=0.62)
    ax.set_title("目标变量分布：30 天内再入院明显不平衡", weight="bold", pad=12)
    ax.set_ylabel("住院记录数")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.01,
                f"{value:,}\n({value / counts.sum():.1%})", ha="center", va="bottom", weight="bold")
    ax.set_ylim(0, counts.max() * 1.16)
    fig.tight_layout()
    save_figure(fig, "01_target_distribution.png")

    missing = pd.read_csv(TAB / "01_missingness.csv", index_col=0).head(12).sort_values("missing_rate")
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    colors = [PALETTE["coral"] if x > 0.4 else PALETTE["gold"] if x > 0.1 else PALETTE["teal"]
              for x in missing["missing_rate"]]
    ax.barh(missing.index, missing["missing_rate"], color=colors)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("缺失率最高的 12 个字段", weight="bold", pad=12)
    ax.set_xlabel("缺失比例")
    ax.spines[["top", "right", "left"]].set_visible(False)
    for y, value in enumerate(missing["missing_rate"]):
        ax.text(value + 0.008, y, f"{value:.1%}", va="center")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, "02_missingness.png")

    age_stats = df.groupby("age", observed=True)["target_30d"].agg(["mean", "sum", "count"]).reset_index()
    order = sorted(age_stats["age"], key=age_midpoint)
    age_stats["age"] = pd.Categorical(age_stats["age"], categories=order, ordered=True)
    age_stats = age_stats.sort_values("age")
    p = age_stats["mean"].to_numpy()
    n = age_stats["count"].to_numpy()
    se = np.sqrt(p * (1 - p) / n)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(age_stats["age"].astype(str), p, marker="o", lw=2.5, color=PALETTE["navy"])
    ax.fill_between(np.arange(len(p)), np.maximum(0, p - 1.96 * se), np.minimum(1, p + 1.96 * se),
                    color=PALETTE["navy"], alpha=0.16, label="95% 置信区间")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("30 天再入院率随年龄上升而增加", weight="bold", pad=12)
    ax.set_xlabel("年龄区间")
    ax.set_ylabel("30 天再入院率")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "03_readmission_by_age.png")
    age_stats.to_csv(TAB / "02_age_readmission.csv", index=False, encoding="utf-8-sig")

    prior = df.assign(prior_inpatient_capped=df["number_inpatient"].clip(upper=5))
    prior_stats = prior.groupby("prior_inpatient_capped")["target_30d"].agg(["mean", "count"]).reset_index()
    prior_stats["label"] = prior_stats["prior_inpatient_capped"].astype(str).replace({"5": "5+"})
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    bars = ax.bar(prior_stats["label"], prior_stats["mean"], color=PALETTE["teal"])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("既往住院次数是最强的风险梯度之一", weight="bold", pad=12)
    ax.set_xlabel("过去一年住院次数")
    ax.set_ylabel("30 天再入院率")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, rate, n_value in zip(bars, prior_stats["mean"], prior_stats["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.005,
                f"{rate:.1%}\nn={n_value:,}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, prior_stats["mean"].max() * 1.25)
    fig.tight_layout()
    save_figure(fig, "04_readmission_by_prior_inpatient.png")
    prior_stats.to_csv(TAB / "03_prior_inpatient_readmission.csv", index=False, encoding="utf-8-sig")

    admission = df.groupby("admission_type_group")["target_30d"].agg(["mean", "count"]).query("count >= 100").sort_values("mean")
    discharge = df.groupby("discharge_group")["target_30d"].agg(["mean", "count"]).query("count >= 100").sort_values("mean")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].barh(admission.index, admission["mean"], color=PALETTE["navy"])
    axes[0].set_title("按入院类型", weight="bold")
    axes[1].barh(discharge.index, discharge["mean"], color=PALETTE["gold"])
    axes[1].set_title("按出院去向", weight="bold")
    for ax in axes:
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("30 天再入院率")
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("就诊路径与再入院风险存在明显差异", fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "05_readmission_by_care_path.png")

    numeric_for_corr = BASE_NUMERIC + [
        "age_mid", "active_med_count", "dose_change_count", "prior_visits_total",
        "care_intensity", "target_30d",
    ]
    corr = df[numeric_for_corr].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="vlag", center=0, vmin=-0.5, vmax=0.5,
                square=True, linewidths=0.4, cbar_kws={"shrink": 0.72}, ax=ax)
    ax.set_title("数值与工程特征的 Spearman 相关矩阵", weight="bold", pad=14)
    fig.tight_layout()
    save_figure(fig, "06_correlation_heatmap.png")
    corr.to_csv(TAB / "04_spearman_correlation.csv", encoding="utf-8-sig")

    lab_stats = df.groupby(["A1C_tested", "A1C_abnormal"])["target_30d"].agg(["mean", "count"]).reset_index()
    lab_stats["group"] = np.select(
        [lab_stats["A1C_tested"].eq(0), lab_stats["A1C_abnormal"].eq(1)],
        ["未检测 HbA1c", "检测且异常"], default="检测且正常",
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    lab_stats = lab_stats.sort_values("mean")
    bars = ax.barh(lab_stats["group"], lab_stats["mean"], color=[PALETTE["teal"], PALETTE["gold"], PALETTE["coral"]])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("30 天再入院率")
    ax.set_title("HbA1c 检测状态与再入院率", weight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, rate, n_value in zip(bars, lab_stats["mean"], lab_stats["count"]):
        ax.text(rate + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}  (n={n_value:,})", va="center")
    fig.tight_layout()
    save_figure(fig, "07_a1c_readmission.png")

    repeat = df.groupby("patient_nbr").size().clip(upper=6).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [str(i) if i < 6 else "6+" for i in repeat.index]
    ax.bar(labels, repeat.values, color=PALETTE["navy"])
    ax.set_yscale("log")
    ax.set_xlabel("同一患者在数据中的住院记录数")
    ax.set_ylabel("患者数（对数刻度）")
    ax.set_title("重复患者说明必须按 patient_nbr 划分数据", weight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "08_repeat_patients.png")


def patient_level_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    patient_target = df.groupby("patient_nbr")["target_30d"].max()
    patients = patient_target.index.to_numpy()
    labels = patient_target.to_numpy()
    train_patients, temp_patients = train_test_split(
        patients, test_size=0.36, random_state=SEED, stratify=labels
    )
    temp_labels = patient_target.loc[temp_patients].to_numpy()
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=20 / 36, random_state=SEED, stratify=temp_labels
    )
    train_idx = df.index[df["patient_nbr"].isin(train_patients)].to_numpy()
    val_idx = df.index[df["patient_nbr"].isin(val_patients)].to_numpy()
    test_idx = df.index[df["patient_nbr"].isin(test_patients)].to_numpy()
    split = {
        "train_rows": int(len(train_idx)), "validation_rows": int(len(val_idx)), "test_rows": int(len(test_idx)),
        "train_patients": int(len(train_patients)), "validation_patients": int(len(val_patients)),
        "test_patients": int(len(test_patients)),
        "train_rate": float(df.loc[train_idx, "target_30d"].mean()),
        "validation_rate": float(df.loc[val_idx, "target_30d"].mean()),
        "test_rate": float(df.loc[test_idx, "target_30d"].mean()),
    }
    (RES / "split_summary.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    return train_idx, val_idx, test_idx, split


def feature_sets() -> dict[str, tuple[list[str], list[str]]]:
    cleaned_num = BASE_NUMERIC + ["age_mid"]
    cleaned_cat = [
        "race_clean", "gender", "admission_type_group", "admission_source_group", "discharge_group",
        "medical_specialty_group", "payer_available", "max_glu_serum", "A1Cresult", "change", "diabetesMed",
    ] + MEDICATION_COLUMNS
    engineered_num = cleaned_num + [
        "primary_diabetes_dx", "diagnosis_diversity", "active_med_count", "dose_change_count",
        "insulin_active", "medication_changed", "diabetes_med_active", "prior_visits_total",
        "prior_inpatient_flag", "prior_emergency_flag", "prior_outpatient_flag", "labs_per_day",
        "medications_per_day", "procedures_per_day", "diagnoses_per_day", "care_intensity",
        "A1C_tested", "A1C_abnormal", "glucose_tested", "glucose_abnormal",
    ]
    engineered_cat = cleaned_cat + ["diag1_group", "diag2_group", "diag3_group"]
    return {
        "Raw": (BASE_NUMERIC, BASE_CATEGORICAL),
        "Cleaned": (cleaned_num, cleaned_cat),
        "Engineered": (engineered_num, engineered_cat),
    }


def make_pipeline(model_name: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    if model_name == "Logistic Regression":
        pre = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=True)),
            ]), categorical),
        ])
        model = LogisticRegression(
            C=0.7, max_iter=500, solver="liblinear", random_state=SEED
        )
    elif model_name == "HistGradientBoosting":
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical),
        ], sparse_threshold=0)
        model = HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=220, max_leaf_nodes=31, min_samples_leaf=30,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.12,
            random_state=SEED,
        )
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocess", pre), ("model", model)])


def choose_f2_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.03, 0.50, 189)
    scores = [fbeta_score(y_true, probability >= threshold, beta=2, zero_division=0) for threshold in thresholds]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def metric_row(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, probability),
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "f2": fbeta_score(y_true, prediction, beta=2, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_models(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
    sets = feature_sets()
    y_train = df.loc[train_idx, "target_30d"].to_numpy()
    y_val = df.loc[val_idx, "target_30d"].to_numpy()
    y_test = df.loc[test_idx, "target_30d"].to_numpy()
    results = []
    predictions = {}
    models = {}

    for feature_name, (numeric, categorical) in sets.items():
        columns = numeric + categorical
        for model_name in ["Logistic Regression", "HistGradientBoosting"]:
            label = f"{feature_name} + {model_name}"
            print(f"\nTraining {label} ({len(columns)} input columns)")
            pipeline = make_pipeline(model_name, numeric, categorical)
            started = time.perf_counter()
            pipeline.fit(df.loc[train_idx, columns], y_train)
            fit_seconds = time.perf_counter() - started
            val_probability = pipeline.predict_proba(df.loc[val_idx, columns])[:, 1]
            threshold, val_f2 = choose_f2_threshold(y_val, val_probability)
            test_probability = pipeline.predict_proba(df.loc[test_idx, columns])[:, 1]
            metrics = metric_row(y_test, test_probability, threshold)
            metrics.update({
                "feature_set": feature_name,
                "model": model_name,
                "input_columns": len(columns),
                "fit_seconds": fit_seconds,
                "validation_f2": val_f2,
            })
            results.append(metrics)
            predictions[label] = {
                "val": val_probability,
                "test": test_probability,
                "threshold": threshold,
                "columns": columns,
            }
            models[label] = pipeline
            print({k: round(metrics[k], 4) for k in ["roc_auc", "pr_auc", "recall", "precision", "f2"]})

    result_df = pd.DataFrame(results).sort_values(["pr_auc", "roc_auc"], ascending=False)
    result_df.to_csv(TAB / "05_model_comparison.csv", index=False, encoding="utf-8-sig")
    return result_df, predictions, models, y_test


def model_figures(result_df: pd.DataFrame, predictions: dict, y_test: np.ndarray) -> str:
    order = ["Raw", "Cleaned", "Engineered"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, metric, title in [
        (axes[0], "roc_auc", "ROC-AUC"), (axes[1], "pr_auc", "PR-AUC（主要指标）")
    ]:
        pivot = result_df.pivot(index="feature_set", columns="model", values=metric).reindex(order)
        pivot.plot(kind="bar", ax=ax, color=[PALETTE["teal"], PALETTE["navy"]], width=0.72)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("特征方案")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=0)
        ax.legend(frameon=False, fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    fig.suptitle("同一患者级划分下：特征工程的影响并非单向", weight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "09_model_comparison.png")

    hgb = (
        result_df[result_df["model"] == "HistGradientBoosting"]
        .set_index("feature_set")
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(order))
    width = 0.24
    for offset, metric, color, label in [
        (-width, "recall", PALETTE["navy"], "召回率"),
        (0, "precision", PALETTE["gold"], "精确率"),
        (width, "f2", PALETTE["teal"], "F2"),
    ]:
        bars = ax.bar(x + offset, hgb[metric], width, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in hgb[metric]], fontsize=8, padding=2)
    ax.set_xticks(x, order)
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("测试集指标")
    ax.set_title("语义清洗提高召回率，但伴随更多假阳性", weight="bold", pad=12)
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "14_threshold_tradeoff.png")

    best = result_df.sort_values(["validation_f2", "pr_auc"], ascending=False).iloc[0]
    best_label = f"{best['feature_set']} + {best['model']}"
    best_probability = predictions[best_label]["test"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for feature_name, color in zip(order, [PALETTE["gold"], PALETTE["teal"], PALETTE["navy"]]):
        label = f"{feature_name} + {best['model']}"
        probability = predictions[label]["test"]
        fpr, tpr, _ = roc_curve(y_test, probability)
        precision, recall, _ = precision_recall_curve(y_test, probability)
        axes[0].plot(fpr, tpr, lw=2, color=color,
                     label=f"{feature_name} (AUC={roc_auc_score(y_test, probability):.3f})")
        axes[1].plot(recall, precision, lw=2, color=color,
                     label=f"{feature_name} (AP={average_precision_score(y_test, probability):.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color="#9CA3AF", lw=1)
    axes[1].axhline(y_test.mean(), ls="--", color="#9CA3AF", lw=1, label=f"基准率={y_test.mean():.1%}")
    axes[0].set(xlabel="假阳性率", ylabel="真阳性率", title="ROC 曲线")
    axes[1].set(xlabel="召回率", ylabel="精确率", title="Precision-Recall 曲线")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8.4)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"最佳模型族（{best['model']}）的特征方案比较", weight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "10_roc_pr_curves.png")

    threshold = float(best["threshold"])
    pred = best_probability >= threshold
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.3))
    sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", cbar=False,
                xticklabels=["预测未再入院", "预测 30 天再入院"],
                yticklabels=["实际未再入院", "实际 30 天再入院"], ax=axes[0])
    axes[0].set_title(f"测试集混淆矩阵（阈值={threshold:.3f}）", weight="bold")
    frac_pos, mean_pred = calibration_curve(y_test, best_probability, n_bins=10, strategy="quantile")
    axes[1].plot(mean_pred, frac_pos, marker="o", lw=2.2, color=PALETTE["teal"], label="模型")
    axes[1].plot([0, 1], [0, 1], ls="--", color="#9CA3AF", label="理想校准")
    axes[1].set(xlabel="预测概率", ylabel="实际发生率", title="概率校准曲线")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "11_confusion_calibration.png")
    return best_label


def bootstrap_group_ci(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    probability: np.ndarray,
    probability_raw: np.ndarray,
    repeats: int = 200,
) -> dict:
    rng = np.random.default_rng(SEED)
    patient_values = df_test["patient_nbr"].to_numpy()
    unique_patients = np.unique(patient_values)
    index_by_patient = {patient: np.flatnonzero(patient_values == patient) for patient in unique_patients}
    aucs, aps, deltas = [], [], []
    for _ in range(repeats):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([index_by_patient[p] for p in sampled])
        y = y_true[idx]
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, probability[idx]))
        aps.append(average_precision_score(y, probability[idx]))
        deltas.append(
            average_precision_score(y, probability[idx])
            - average_precision_score(y, probability_raw[idx])
        )
    summary = {
        "roc_auc": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "pr_auc": [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))],
        "pr_auc_delta_selected_vs_raw": [
            float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
        ],
        "bootstrap_repeats": len(aucs),
    }
    (RES / "bootstrap_ci.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def explain_best_model(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    result_df: pd.DataFrame,
    predictions: dict,
    models: dict,
    best_label: str,
    y_test: np.ndarray,
) -> None:
    best = result_df.iloc[0]
    columns = predictions[best_label]["columns"]
    model = models[best_label]
    sample_n = min(6000, len(test_idx))
    rng = np.random.default_rng(SEED)
    sample_positions = rng.choice(np.arange(len(test_idx)), size=sample_n, replace=False)
    sample_idx = test_idx[sample_positions]
    importance = permutation_importance(
        model,
        df.loc[sample_idx, columns],
        df.loc[sample_idx, "target_30d"],
        scoring="average_precision",
        n_repeats=4,
        random_state=SEED,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "feature": columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(TAB / "06_permutation_importance.csv", index=False, encoding="utf-8-sig")
    top = importance_df.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"],
            color=PALETTE["teal"], alpha=0.92, ecolor=PALETTE["muted"], capsize=2)
    ax.set_xlabel("置换后 PR-AUC 的平均下降")
    ax.set_title("最佳模型的全局置换重要性", weight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "12_permutation_importance.png")

    probability = predictions[best_label]["test"]
    threshold = predictions[best_label]["threshold"]
    test = df.loc[test_idx].copy()
    test["probability"] = probability
    test["prediction"] = (probability >= threshold).astype(int)
    test["error_type"] = np.select(
        [
            (test["target_30d"] == 1) & (test["prediction"] == 0),
            (test["target_30d"] == 0) & (test["prediction"] == 1),
        ],
        ["False negative", "False positive"], default="Correct",
    )
    error_summary = test.groupby("error_type").agg(
        count=("target_30d", "size"),
        mean_probability=("probability", "mean"),
        mean_age=("age_mid", "mean"),
        mean_prior_inpatient=("number_inpatient", "mean"),
        mean_time_in_hospital=("time_in_hospital", "mean"),
        mean_diagnoses=("number_diagnoses", "mean"),
    ).reset_index()
    error_summary.to_csv(TAB / "07_error_analysis.csv", index=False, encoding="utf-8-sig")

    subgroup_rows = []
    for variable in ["gender", "race_clean", "age"]:
        for value, group in test.groupby(variable, observed=True):
            if len(group) < 250 or group["target_30d"].nunique() < 2:
                continue
            y = group["target_30d"].to_numpy()
            p = group["probability"].to_numpy()
            pred = group["prediction"].to_numpy()
            subgroup_rows.append({
                "variable": variable, "group": str(value), "n": len(group),
                "positive_rate": y.mean(), "roc_auc": roc_auc_score(y, p),
                "pr_auc": average_precision_score(y, p),
                "recall": recall_score(y, pred, zero_division=0),
                "precision": precision_score(y, pred, zero_division=0),
            })
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(TAB / "08_subgroup_performance.csv", index=False, encoding="utf-8-sig")

    race = subgroup[subgroup["variable"] == "race_clean"].sort_values("n", ascending=False).head(6)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    axes[0].barh(race["group"], race["recall"], color=PALETTE["navy"])
    axes[1].barh(race["group"], race["precision"], color=PALETTE["gold"])
    axes[0].set_title("召回率", weight="bold")
    axes[1].set_title("精确率", weight="bold")
    for ax in axes:
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlim(0, max(0.8, subgroup[["recall", "precision"]].max().max() * 1.1))
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("主要种族组的测试集表现（描述性公平性审计）", weight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "13_subgroup_performance.png")


def extended_experiments(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    result_df: pd.DataFrame,
    predictions: dict,
) -> dict:
    numeric, categorical = feature_sets()["Cleaned"]
    columns = numeric + categorical
    y = df["target_30d"].to_numpy()
    y_train = df.loc[train_idx, "target_30d"].to_numpy()
    y_val = df.loc[val_idx, "target_30d"].to_numpy()
    y_test = df.loc[test_idx, "target_30d"].to_numpy()
    selected_label = "Cleaned + HistGradientBoosting"
    test_probability = predictions[selected_label]["test"]

    # 1) Five-fold patient-level stability.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for fold, (fit_pos, holdout_pos) in enumerate(
        cv.split(df[columns], y, groups=df["patient_nbr"]), start=1
    ):
        pipeline = make_pipeline("HistGradientBoosting", numeric, categorical)
        pipeline.fit(df.iloc[fit_pos][columns], y[fit_pos])
        probability = pipeline.predict_proba(df.iloc[holdout_pos][columns])[:, 1]
        cv_rows.append({
            "fold": fold,
            "n": len(holdout_pos),
            "positive_rate": y[holdout_pos].mean(),
            "roc_auc": roc_auc_score(y[holdout_pos], probability),
            "pr_auc": average_precision_score(y[holdout_pos], probability),
            "brier": brier_score_loss(y[holdout_pos], probability),
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(TAB / "09_group_cv.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.1))
    for ax, metric, color, title in [
        (axes[0], "roc_auc", PALETTE["navy"], "ROC-AUC"),
        (axes[1], "pr_auc", PALETTE["teal"], "PR-AUC"),
    ]:
        ax.plot(cv_df["fold"], cv_df[metric], marker="o", lw=2.2, color=color)
        mean = cv_df[metric].mean()
        ax.axhline(mean, ls="--", color=PALETTE["muted"], label=f"均值={mean:.3f}")
        ax.set_xticks(cv_df["fold"])
        ax.set_xlabel("患者级折")
        ax.set_ylabel(metric.upper())
        ax.set_title(title, weight="bold")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cleaned + HistGradientBoosting 的 5 折稳定性", weight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "15_group_cv_stability.png")

    # 2) Top-k capacity analysis for operational follow-up resources.
    capacity_rows = []
    order = np.argsort(-test_probability)
    positives = y_test.sum()
    for fraction in [0.05, 0.10, 0.20, 0.30, 0.40]:
        k = max(1, int(np.ceil(len(y_test) * fraction)))
        selected = order[:k]
        tp = int(y_test[selected].sum())
        precision = tp / k
        recall = tp / positives
        capacity_rows.append({
            "capacity": fraction, "selected_n": k, "true_positive": tp,
            "precision": precision, "recall_capture": recall,
            "lift": precision / y_test.mean(),
        })
    capacity = pd.DataFrame(capacity_rows)
    capacity.to_csv(TAB / "10_capacity_analysis.csv", index=False, encoding="utf-8-sig")
    fig, ax1 = plt.subplots(figsize=(7.7, 4.6))
    ax1.plot(capacity["capacity"], capacity["recall_capture"], marker="o", lw=2.5,
             color=PALETTE["navy"], label="捕获的再入院患者比例")
    ax1.plot(capacity["capacity"], capacity["precision"], marker="s", lw=2.5,
             color=PALETTE["teal"], label="名单精确率")
    ax1.xaxis.set_major_formatter(PercentFormatter(1))
    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax1.set_xlabel("可随访容量（测试集风险最高的比例）")
    ax1.set_ylabel("比例")
    ax1.set_title("有限随访资源下的 top-k 收益", weight="bold", pad=12)
    ax1.legend(frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "16_capacity_curve.png")

    # 3) Critical feature ablation on the exact same split.
    ablations = {
        "完整 Cleaned": [],
        "移除既往住院": ["number_inpatient"],
        "移除出院去向": ["discharge_group"],
        "移除 payer 可得性": ["payer_available"],
        "移除前三项": ["number_inpatient", "discharge_group", "payer_available"],
    }
    ablation_rows = []
    for name, removed in ablations.items():
        num = [col for col in numeric if col not in removed]
        cat = [col for col in categorical if col not in removed]
        cols = num + cat
        pipeline = make_pipeline("HistGradientBoosting", num, cat)
        pipeline.fit(df.loc[train_idx, cols], y_train)
        probability = pipeline.predict_proba(df.loc[test_idx, cols])[:, 1]
        ablation_rows.append({
            "experiment": name,
            "removed": ", ".join(removed) if removed else "None",
            "roc_auc": roc_auc_score(y_test, probability),
            "pr_auc": average_precision_score(y_test, probability),
        })
    ablation = pd.DataFrame(ablation_rows)
    base_ap = ablation.loc[ablation["experiment"] == "完整 Cleaned", "pr_auc"].iloc[0]
    ablation["pr_auc_change"] = ablation["pr_auc"] - base_ap
    ablation.to_csv(TAB / "11_ablation.csv", index=False, encoding="utf-8-sig")
    plot_ablation = ablation.iloc[1:].sort_values("pr_auc_change")
    fig, ax = plt.subplots(figsize=(7.7, 4.4))
    colors = [PALETTE["coral"] if value < 0 else PALETTE["teal"] for value in plot_ablation["pr_auc_change"]]
    bars = ax.barh(plot_ablation["experiment"], plot_ablation["pr_auc_change"], color=colors)
    ax.axvline(0, color=PALETTE["ink"], lw=1)
    ax.set_xlabel("相对完整 Cleaned 的 PR-AUC 变化")
    ax.set_title("关键特征消融：既往住院贡献不可替代", weight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.bar_label(bars, labels=[f"{v:+.4f}" for v in plot_ablation["pr_auc_change"]], fontsize=9)
    fig.tight_layout()
    save_figure(fig, "17_feature_ablation.png")

    # 4) Deliberately leaky row-level split as a methodological counterexample.
    positions = np.arange(len(df))
    row_train, row_temp = train_test_split(
        positions, test_size=0.36, random_state=SEED, stratify=y
    )
    row_val, row_test = train_test_split(
        row_temp, test_size=20 / 36, random_state=SEED, stratify=y[row_temp]
    )
    row_model = make_pipeline("HistGradientBoosting", numeric, categorical)
    row_model.fit(df.iloc[row_train][columns], y[row_train])
    row_val_prob = row_model.predict_proba(df.iloc[row_val][columns])[:, 1]
    row_threshold, _ = choose_f2_threshold(y[row_val], row_val_prob)
    row_prob = row_model.predict_proba(df.iloc[row_test][columns])[:, 1]
    row_metrics = metric_row(y[row_test], row_prob, row_threshold)
    patient_test = result_df[
        (result_df["feature_set"] == "Cleaned")
        & (result_df["model"] == "HistGradientBoosting")
    ].iloc[0]
    overlap = len(
        set(df.iloc[row_train]["patient_nbr"].unique())
        & set(df.iloc[row_test]["patient_nbr"].unique())
    )
    leakage = pd.DataFrame([
        {
            "split": "患者级互斥",
            "roc_auc": patient_test["roc_auc"], "pr_auc": patient_test["pr_auc"],
            "recall": patient_test["recall"], "precision": patient_test["precision"],
            "overlap_patients": 0,
        },
        {
            "split": "按记录随机",
            "roc_auc": row_metrics["roc_auc"], "pr_auc": row_metrics["pr_auc"],
            "recall": row_metrics["recall"], "precision": row_metrics["precision"],
            "overlap_patients": overlap,
        },
    ])
    leakage.to_csv(TAB / "12_split_leakage.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(2)
    width = 0.32
    for offset, metric, color in [(-width / 2, "roc_auc", PALETTE["navy"]), (width / 2, "pr_auc", PALETTE["teal"])]:
        bars = ax.bar(x + offset, leakage[metric], width, label=metric.upper(), color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    ax.set_xticks(x, leakage["split"])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("测试指标")
    ax.set_title("按记录随机拆分会混入重复患者", weight="bold", pad=12)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "18_split_leakage.png")

    extended = {
        "cv_roc_mean": float(cv_df["roc_auc"].mean()),
        "cv_roc_sd": float(cv_df["roc_auc"].std(ddof=1)),
        "cv_pr_mean": float(cv_df["pr_auc"].mean()),
        "cv_pr_sd": float(cv_df["pr_auc"].std(ddof=1)),
        "row_split_overlap_patients": int(overlap),
        "row_split_roc_auc": float(row_metrics["roc_auc"]),
        "row_split_pr_auc": float(row_metrics["pr_auc"]),
    }
    (RES / "extended_experiments.json").write_text(
        json.dumps(extended, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return extended


def main() -> None:
    started = time.perf_counter()
    df, audit = prepare_data()
    make_eda_figures(df, audit)
    train_idx, val_idx, test_idx, split = patient_level_split(df)
    result_df, predictions, models, y_test = run_models(df, train_idx, val_idx, test_idx)
    best_label = model_figures(result_df, predictions, y_test)

    best_row = result_df.sort_values(["validation_f2", "pr_auc"], ascending=False).iloc[0]
    raw_label = f"Raw + {best_row['model']}"
    ci = bootstrap_group_ci(
        df.loc[test_idx], y_test, predictions[best_label]["test"], predictions[raw_label]["test"]
    )
    explain_best_model(df, test_idx, result_df, predictions, models, best_label, y_test)
    extended = extended_experiments(df, train_idx, val_idx, test_idx, result_df, predictions)

    summary = {
        "best_label": best_label,
        "best_metrics": {
            key: (float(best_row[key]) if isinstance(best_row[key], (np.floating, float)) else best_row[key])
            for key in ["roc_auc", "pr_auc", "brier", "threshold", "precision", "recall", "specificity", "f1", "f2"]
        },
        "bootstrap_ci": ci,
        "extended_experiments": extended,
        "runtime_seconds": time.perf_counter() - started,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        },
    }
    (RES / "project_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
