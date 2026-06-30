"""Evaluate modern tabular deep-learning models on the locked patient split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diabetes_readmission_project import patient_level_split, prepare_data  # noqa: E402
from optimization_lab import add_rich_features  # noqa: E402


OUT = HERE / "optimization"
TAB = OUT / "tables"
RES = OUT / "results"
PREDICTION_FILE = RES / "sota_probabilities.npz"
SUMMARY_FILE = RES / "sota_summary.json"
TAB.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def prepare_model_frame(frame, numeric, categorical, train_idx):
    model_frame = frame[numeric + categorical].copy()
    for column in numeric:
        train_values = pd.to_numeric(model_frame.loc[train_idx, column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = train_values.median()
        model_frame[column] = (
            pd.to_numeric(model_frame[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(median)
            .astype("float32")
        )
    for column in categorical:
        model_frame[column] = model_frame[column].astype("string").fillna("Missing").astype(str)
    return model_frame


def metric_row(name, split, y, probability, runtime_seconds):
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "model": name,
        "split": split,
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "log_loss": log_loss(y, clipped),
        "runtime_seconds": runtime_seconds,
    }


def make_model(name, device, seed, smoke):
    from pytabkit import FTT_D_Classifier, RealMLP_TD_Classifier, TabM_D_Classifier

    if name in {"tabm", "tabm_pwl"}:
        return TabM_D_Classifier(
            device=device,
            random_state=seed,
            n_cv=1,
            n_refit=0,
            n_epochs=2 if smoke else 96,
            patience=2 if smoke else 12,
            batch_size=256,
            num_emb_type="pwl" if name == "tabm_pwl" else "none",
            num_emb_n_bins=48,
            val_metric_name="cross_entropy",
            verbosity=1,
        )
    if name == "realmlp":
        return RealMLP_TD_Classifier(
            device=device,
            random_state=seed,
            n_cv=1,
            n_refit=0,
            n_epochs=2 if smoke else 128,
            batch_size=256,
            val_metric_name="cross_entropy",
            use_ls=False,
            verbosity=1,
        )
    if name == "fttransformer":
        return FTT_D_Classifier(
            device=device,
            random_state=seed,
            n_cv=1,
            n_refit=0,
            max_epochs=2 if smoke else 96,
            es_patience=2 if smoke else 12,
            batch_size=256,
            val_metric_name="cross_entropy",
            verbosity=1,
        )
    raise ValueError(f"Unknown model: {name}")


def load_existing_predictions():
    if not PREDICTION_FILE.exists():
        return {}
    saved = np.load(PREDICTION_FILE)
    return {key: saved[key] for key in saved.files}


def save_results(predictions, rows):
    np.savez_compressed(PREDICTION_FILE, **predictions)
    pd.DataFrame(rows).to_csv(TAB / "17_sota_models.csv", index=False, encoding="utf-8-sig")
    SUMMARY_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tabm", "tabm_pwl", "realmlp", "fttransformer"], required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    frame, _ = prepare_data()
    frame, numeric, categorical = add_rich_features(frame)
    train_idx, val_idx, test_idx, _ = patient_level_split(frame)
    model_frame = prepare_model_frame(frame, numeric, categorical, train_idx)
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(train_idx, size=min(5000, len(train_idx)), replace=False)
        val_idx = rng.choice(val_idx, size=min(1200, len(val_idx)), replace=False)
        test_idx = rng.choice(test_idx, size=min(1200, len(test_idx)), replace=False)

    columns = numeric + categorical
    x_train, y_train = model_frame.loc[train_idx, columns], frame.loc[train_idx, "target_30d"].to_numpy()
    x_val, y_val = model_frame.loc[val_idx, columns], frame.loc[val_idx, "target_30d"].to_numpy()
    x_test, y_test = model_frame.loc[test_idx, columns], frame.loc[test_idx, "target_30d"].to_numpy()

    model = make_model(args.model, args.device, args.seed, args.smoke)
    start = time.time()
    model.fit(x_train, y_train, X_val=x_val, y_val=y_val, cat_col_names=categorical)
    runtime = time.time() - start
    val_probability = model.predict_proba(x_val)[:, 1]
    test_probability = model.predict_proba(x_test)[:, 1]
    rows = [
        metric_row(args.model, "validation", y_val, val_probability, runtime),
        metric_row(args.model, "test", y_test, test_probability, runtime),
    ]
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    if not args.smoke:
        predictions = load_existing_predictions()
        predictions.update({
            "y_val": y_val,
            "y_test": y_test,
            f"val_{args.model}": val_probability,
            f"test_{args.model}": test_probability,
        })
        old_rows = []
        if SUMMARY_FILE.exists():
            old_rows = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
            old_rows = [row for row in old_rows if row["model"] != args.model]
        save_results(predictions, old_rows + rows)


if __name__ == "__main__":
    main()
