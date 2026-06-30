from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font="Heiti TC", rc={"axes.unicode_minus": False})
navy, teal, coral, muted = "#245B78", "#2A9D8F", "#D2644A", "#64748B"


hist = pd.read_csv(ROOT / "tables/09_group_cv.csv")
cat = pd.read_csv(ROOT / "bonus/tables/09_catboost_group_cv.csv")
comparison = pd.DataFrame({
    "fold": hist["fold"],
    "HistGB PR-AUC": hist["pr_auc"],
    "CatBoost PR-AUC": cat["pr_auc"],
})
comparison.to_csv(ROOT / "bonus/tables/11_catboost_fold_comparison.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
axes[0].plot(comparison["fold"], comparison["HistGB PR-AUC"], marker="o", lw=2.2, color=navy, label="HistGB")
axes[0].plot(comparison["fold"], comparison["CatBoost PR-AUC"], marker="o", lw=2.2, color=teal, label="CatBoost")
axes[0].set_xticks(comparison["fold"])
axes[0].set_xlabel("患者分组折")
axes[0].set_ylabel("PR-AUC")
axes[0].set_title("五折配对比较", weight="bold")
axes[0].legend(frameon=False)

bootstrap = pd.read_csv(ROOT / "bonus/tables/10_catboost_paired_bootstrap.csv")
delta_clean = bootstrap["pr_delta_catboost_vs_histgb_cleaned"]
delta_raw = bootstrap["pr_delta_catboost_vs_histgb_raw"]
for values, label, color in [(delta_clean, "vs Cleaned HistGB", teal), (delta_raw, "vs Raw HistGB", coral)]:
    sns.kdeplot(values, ax=axes[1], label=label, color=color, lw=2.2, fill=False)
    axes[1].axvline(values.mean(), color=color, ls="--", lw=1.4)
axes[1].axvline(0, color=muted, lw=1)
axes[1].set_xlabel("CatBoost - HistGB 的 PR-AUC 差")
axes[1].set_ylabel("密度")
axes[1].set_title("500 次患者级 Bootstrap", weight="bold")
axes[1].legend(frameon=False, fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "catboost_validation.png", dpi=240, bbox_inches="tight")
plt.close(fig)


calibration = pd.read_csv(ROOT / "bonus/tables/04_calibration_comparison.csv")
selected = calibration[calibration["method"].isin([
    "HistGB uncalibrated", "HistGB Platt", "CatBoost uncalibrated", "CatBoost Platt"
])].copy()
selected["label"] = selected["method"].replace({
    "HistGB uncalibrated": "HistGB\n原始",
    "HistGB Platt": "HistGB\nPlatt",
    "CatBoost uncalibrated": "CatBoost\n原始",
    "CatBoost Platt": "CatBoost\nPlatt",
})
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
axes[0].bar(selected["label"], selected["brier"], color=[navy, teal, coral, "#E9C46A"])
axes[0].set_ylim(0.0948, 0.0955)
axes[0].set_ylabel("Brier score（越低越好）")
axes[0].set_title("概率误差", weight="bold")
axes[1].scatter(selected["calibration_slope"], selected["calibration_intercept"], s=70,
                c=[navy, teal, coral, "#E9C46A"])
for _, row in selected.iterrows():
    axes[1].annotate(row["label"].replace("\n", " "), (row["calibration_slope"], row["calibration_intercept"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=8)
axes[1].axvline(1, color=muted, ls="--")
axes[1].axhline(0, color=muted, ls="--")
axes[1].set_xlabel("校准斜率（理想值 1）")
axes[1].set_ylabel("校准截距（理想值 0）")
axes[1].set_title("校准偏差", weight="bold")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "calibration_summary.png", dpi=240, bbox_inches="tight")
plt.close(fig)
