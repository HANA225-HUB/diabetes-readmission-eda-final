from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font="Heiti TC", rc={"axes.unicode_minus": False})
navy, teal, coral, gold, muted = "#245B78", "#2A9D8F", "#D2644A", "#E9C46A", "#64748B"


def prettify(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


metrics = pd.read_csv(ROOT / "optimization/tables/35_history_final_metrics.csv")
order = ["focused_extra_ensemble", "strict_history_ensemble", "target_history_sensitivity_ensemble"]
labels = {
    "focused_extra_ensemble": "非历史增强\n集成",
    "strict_history_ensemble": "严格纵向历史\n主模型",
    "target_history_sensitivity_ensemble": "目标历史\n敏感性上限",
}
metrics = metrics.set_index("model").loc[order].reset_index()

fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
for ax, metric, title, ylim, color in [
    (axes[0], "pr_auc", "PR-AUC（主指标）", (0.235, 0.270), navy),
    (axes[1], "roc_auc", "ROC-AUC", (0.670, 0.695), teal),
    (axes[2], "brier", "Brier score（越低越好）", (0.0920, 0.0940), coral),
]:
    bars = ax.bar([labels[x] for x in metrics["model"]], metrics[metric], color=color, alpha=0.88)
    ax.set_title(title, weight="bold")
    ax.set_ylim(*ylim)
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    for bar, value in zip(bars, metrics[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                ha="center", va="bottom" if metric != "brier" else "top",
                fontsize=8.5, weight="bold")
    prettify(ax)
fig.suptitle("纵向历史特征带来主要性能跃迁", weight="bold", y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "history_final_metrics.png", dpi=240, bbox_inches="tight")
plt.close(fig)


cv = pd.read_csv(ROOT / "optimization/tables/42_history_group_cv.csv")
pivot = cv.pivot(index="fold", columns="model", values="pr_auc")
summary = pd.read_csv(ROOT / "optimization/tables/43_history_group_cv_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
axes[0].plot(pivot.index, pivot["prefix_catboost_no_history"], marker="o", lw=2.2,
             color=muted, label="无历史特征")
axes[0].plot(pivot.index, pivot["prefix_catboost_strict_history"], marker="o", lw=2.2,
             color=teal, label="严格历史特征")
axes[0].set_xticks(pivot.index)
axes[0].set_xlabel("患者级分组折")
axes[0].set_ylabel("PR-AUC")
axes[0].set_title("五折稳定性", weight="bold")
axes[0].legend(frameon=False)
delta = pivot["prefix_catboost_strict_history"] - pivot["prefix_catboost_no_history"]
bars = axes[1].bar(delta.index, delta.values, color=navy)
axes[1].axhline(0, color=muted, lw=1)
axes[1].set_xticks(delta.index)
axes[1].set_xlabel("患者级分组折")
axes[1].set_ylabel("PR-AUC 增益")
axes[1].set_title(f"5/5 折正增益，均值 +{delta.mean():.3f}", weight="bold")
for bar, value in zip(bars, delta.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"+{value:.3f}",
                 ha="center", va="bottom", fontsize=8.5)
for ax in axes:
    prettify(ax)
fig.tight_layout()
fig.savefig(OUT / "history_cv_stability.png", dpi=240, bbox_inches="tight")
plt.close(fig)


sub = pd.read_csv(ROOT / "optimization/tables/39_history_subgroup_effect.csv")
sub = sub[sub["group_type"].eq("observed_history_group")].copy()
sub["group"] = sub["group"].replace({
    "first_observed_encounter": "首次观测住院",
    "has_prior_observed_encounter": "已有既往观测住院",
})
sub["model"] = sub["model"].replace({
    "focused_extra_ensemble": "非历史增强",
    "strict_history_ensemble": "严格历史主模型",
    "target_history_sensitivity_ensemble": "目标历史敏感性",
})
fig, ax = plt.subplots(figsize=(8.4, 4.2))
sns.barplot(data=sub, x="group", y="pr_auc", hue="model", palette=[muted, teal, coral], ax=ax)
ax.set_xlabel("")
ax.set_ylabel("测试集 PR-AUC")
ax.set_title("历史特征的增益集中在有既往观测住院的患者", weight="bold")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
prettify(ax)
fig.tight_layout()
fig.savefig(OUT / "history_subgroup_effect.png", dpi=240, bbox_inches="tight")
plt.close(fig)


boot_summary = pd.read_json(ROOT / "optimization/results/history_bootstrap_summary.json").T.reset_index()
boot_summary[["low", "high"]] = pd.DataFrame(boot_summary["ci95"].tolist(), index=boot_summary.index)
boot_summary["mean_value"] = boot_summary["mean"].astype(float)
selected = boot_summary[boot_summary["index"].isin([
    "pr_strict_history_vs_focused_extra",
    "roc_strict_history_vs_focused_extra",
    "top5_precision_strict_history_vs_focused_extra",
    "pr_target_history_sensitivity_vs_focused_extra",
    "top5_precision_target_history_sensitivity_vs_focused_extra",
])].copy()
selected["label"] = selected["index"].replace({
    "pr_strict_history_vs_focused_extra": "严格历史 vs 非历史\nPR-AUC",
    "roc_strict_history_vs_focused_extra": "严格历史 vs 非历史\nROC-AUC",
    "top5_precision_strict_history_vs_focused_extra": "严格历史 vs 非历史\nTop5精确率",
    "pr_target_history_sensitivity_vs_focused_extra": "敏感性上限 vs 非历史\nPR-AUC",
    "top5_precision_target_history_sensitivity_vs_focused_extra": "敏感性上限 vs 非历史\nTop5精确率",
})
fig, ax = plt.subplots(figsize=(8.8, 4.0))
y = np.arange(len(selected))
ax.errorbar(selected["mean_value"], y,
            xerr=[selected["mean_value"] - selected["low"], selected["high"] - selected["mean_value"]],
            fmt="o", color=navy, ecolor=teal, elinewidth=2.2, capsize=4)
ax.axvline(0, color=muted, lw=1)
ax.set_yticks(y, selected["label"])
ax.set_xlabel("患者级 Bootstrap 增益均值与 95% CI")
ax.set_title("纵向历史模型相对非历史增强模型的稳定增益", weight="bold")
ax.invert_yaxis()
prettify(ax)
fig.tight_layout()
fig.savefig(OUT / "history_bootstrap_delta.png", dpi=240, bbox_inches="tight")
plt.close(fig)
