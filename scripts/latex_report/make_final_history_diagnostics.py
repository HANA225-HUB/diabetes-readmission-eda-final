from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
TAB = ROOT / "optimization" / "tables"
RES = ROOT / "optimization" / "results"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font="Heiti TC", rc={"axes.unicode_minus": False})
navy, teal, coral, gold, muted = "#245B78", "#2A9D8F", "#D2644A", "#E9C46A", "#64748B"


def prettify(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.28)


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi)
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def calibration_slope_intercept(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    model.fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def calibration_bins(y, p, model_name, bins=10):
    order = np.argsort(p)
    groups = np.array_split(order, bins)
    rows = []
    for idx, group in enumerate(groups, start=1):
        rows.append({
            "model": model_name,
            "bin": idx,
            "n": len(group),
            "mean_predicted": float(p[group].mean()),
            "observed_rate": float(y[group].mean()),
        })
    return rows


def decision_curve(y, probabilities):
    prevalence = float(y.mean())
    n = len(y)
    rows = []
    for threshold in np.linspace(0.02, 0.30, 57):
        rows.append({
            "model": "全部干预",
            "threshold": threshold,
            "net_benefit": prevalence - (1 - prevalence) * threshold / (1 - threshold),
        })
        rows.append({"model": "不干预", "threshold": threshold, "net_benefit": 0.0})
        for name, p in probabilities.items():
            pred = p >= threshold
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            rows.append({
                "model": name,
                "threshold": threshold,
                "net_benefit": tp / n - fp / n * threshold / (1 - threshold),
            })
    return pd.DataFrame(rows)


def topk_absolute(y, probabilities):
    rows = []
    prevalence = float(y.mean())
    total_positive = int(y.sum())
    focused_reference = None
    for model_name, p in probabilities.items():
        order = np.argsort(-p)
        for fraction in [0.03, 0.05, 0.10, 0.20]:
            k = max(1, int(np.ceil(len(y) * fraction)))
            selected = order[:k]
            tp = int(y[selected].sum())
            fp = int(k - tp)
            expected_random = k * prevalence
            row = {
                "model": model_name,
                "capacity": fraction,
                "selected_n": k,
                "true_positive": tp,
                "false_positive": fp,
                "precision": tp / k,
                "recall_capture": tp / total_positive,
                "lift": (tp / k) / prevalence,
                "expected_true_positive_random": expected_random,
                "extra_true_positive_vs_random": tp - expected_random,
            }
            rows.append(row)
            if model_name == "非历史增强集成":
                focused_reference = focused_reference or {}
                focused_reference[fraction] = tp
    table = pd.DataFrame(rows)
    table["extra_true_positive_vs_non_history"] = np.nan
    for idx, row in table.iterrows():
        if row["model"] != "非历史增强集成" and focused_reference is not None:
            table.loc[idx, "extra_true_positive_vs_non_history"] = (
                row["true_positive"] - focused_reference[row["capacity"]]
            )
    return table


def main():
    data = np.load(RES / "history_final_probabilities.npz")
    y = data["y_test"].astype(int)
    probabilities = {
        "非历史增强集成": data["focused_extra_test"],
        "严格历史主模型": data["strict_history_ensemble_test"],
        "目标历史敏感性": data["target_history_sensitivity_ensemble_test"],
    }

    calibration_rows = []
    bin_rows = []
    baseline_brier = brier_score_loss(y, np.repeat(y.mean(), len(y)))
    for name, p in probabilities.items():
        intercept, slope = calibration_slope_intercept(y, p)
        brier = brier_score_loss(y, p)
        calibration_rows.append({
            "model": name,
            "brier": brier,
            "brier_skill_vs_prevalence": 1 - brier / baseline_brier,
            "ece_10_quantile": ece(y, p),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        })
        bin_rows.extend(calibration_bins(y, p, name))
    pd.DataFrame(calibration_rows).to_csv(
        TAB / "57_final_history_calibration.csv", index=False, encoding="utf-8-sig"
    )
    calibration_bin_table = pd.DataFrame(bin_rows)
    calibration_bin_table.to_csv(
        TAB / "58_final_history_calibration_bins.csv", index=False, encoding="utf-8-sig"
    )

    dca = decision_curve(y, probabilities)
    dca.to_csv(TAB / "59_final_history_decision_curve.csv", index=False, encoding="utf-8-sig")

    topk = topk_absolute(y, probabilities)
    topk.to_csv(TAB / "60_final_history_topk_absolute.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    plot_bins = calibration_bin_table[calibration_bin_table["model"].isin(["非历史增强集成", "严格历史主模型"])]
    for name, color in [("非历史增强集成", muted), ("严格历史主模型", teal)]:
        group = plot_bins[plot_bins["model"].eq(name)]
        axes[0].plot(group["mean_predicted"], group["observed_rate"], marker="o", lw=2.2, color=color, label=name)
    low = min(plot_bins["mean_predicted"].min(), plot_bins["observed_rate"].min())
    high = max(plot_bins["mean_predicted"].max(), plot_bins["observed_rate"].max())
    axes[0].plot([low, high], [low, high], ls="--", color=navy, lw=1.2, label="理想校准")
    axes[0].set_xlabel("分箱平均预测概率")
    axes[0].set_ylabel("分箱实际发生率")
    axes[0].set_title("最终模型可靠性曲线", weight="bold")
    axes[0].legend(frameon=False, fontsize=9)

    for name, color, linestyle in [
        ("非历史增强集成", muted, "-"),
        ("严格历史主模型", teal, "-"),
        ("全部干预", coral, "--"),
        ("不干预", navy, ":"),
    ]:
        group = dca[dca["model"].eq(name)]
        axes[1].plot(group["threshold"], group["net_benefit"], color=color, ls=linestyle, lw=2.0, label=name)
    axes[1].set_xlabel("风险阈值")
    axes[1].set_ylabel("净收益")
    axes[1].set_title("最终模型决策曲线", weight="bold")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        prettify(ax)
    fig.tight_layout()
    fig.savefig(OUT / "final_history_calibration_dca.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    topk_plot = topk[topk["model"].isin(["非历史增强集成", "严格历史主模型"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sns.barplot(
        data=topk_plot[topk_plot["capacity"].isin([0.03, 0.05, 0.10])],
        x="capacity",
        y="true_positive",
        hue="model",
        palette=[muted, teal],
        ax=ax,
    )
    ax.set_xticks([0, 1, 2], ["Top 3%", "Top 5%", "Top 10%"])
    ax.set_xlabel("随访容量")
    ax.set_ylabel("名单中真实 30 天再入院数")
    ax.set_title("同等随访名额下严格历史模型捕获更多病例", weight="bold")
    ax.legend(frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)
    prettify(ax)
    fig.tight_layout()
    fig.savefig(OUT / "final_history_topk_absolute.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    gap_metrics_path = TAB / "55_history_gap_ablation_final_metrics.csv"
    gap_topk_path = TAB / "56_history_gap_ablation_topk.csv"
    if gap_metrics_path.exists() and gap_topk_path.exists():
        gap_metrics = pd.read_csv(gap_metrics_path)
        gap_topk = pd.read_csv(gap_topk_path)
        model_order = [
            "focused_extra_ensemble",
            "strict_history_no_gap_ensemble",
            "strict_history_ensemble_full",
        ]
        gap_labels = {
            "focused_extra_ensemble": "非历史增强",
            "strict_history_no_gap_ensemble": "严格历史\n去 gap",
            "strict_history_ensemble_full": "严格历史\n完整",
        }
        gap_metrics = gap_metrics.set_index("model").loc[model_order].reset_index()
        top5 = (
            gap_topk[gap_topk["capacity"].eq(0.05)]
            .set_index("model")
            .loc[model_order]
            .reset_index()
        )
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
        axes[0].bar([gap_labels[x] for x in gap_metrics["model"]], gap_metrics["pr_auc"],
                    color=[muted, gold, teal])
        axes[0].set_ylim(0.238, 0.264)
        axes[0].set_ylabel("测试集 AP/PR-AUC")
        axes[0].set_title("去除 encounter-id gap 后排序仍然稳定", weight="bold")
        for i, value in enumerate(gap_metrics["pr_auc"]):
            axes[0].text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5, weight="bold")

        axes[1].bar([gap_labels[x] for x in top5["model"]], top5["precision"],
                    color=[muted, gold, teal])
        axes[1].set_ylim(0.32, 0.39)
        axes[1].set_ylabel("Top 5% 精确率")
        axes[1].set_title("容量名单表现没有依赖 gap 特征", weight="bold")
        for i, value in enumerate(top5["precision"]):
            axes[1].text(i, value, f"{value:.1%}", ha="center", va="bottom", fontsize=8.5, weight="bold")
        for ax in axes:
            prettify(ax)
        fig.tight_layout()
        fig.savefig(OUT / "history_gap_ablation.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
