"""Train churn models, evaluate them, and save the best XGBoost bundle."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import (
    build_preprocessor,
    clean_dataset,
    feature_names,
    load_dataset,
    make_models_dir,
    make_reports_dir,
    split_data,
    split_features_target,
)


def evaluate_model(name: str, model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "precision_churn": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall_churn": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1_churn": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "probabilities": probabilities,
        "predictions": predictions,
    }
    return metrics


def plot_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pr_curve(y_test: pd.Series, probabilities: np.ndarray, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2, color="#1f77b4")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_metrics_report(metrics: dict, path: Path) -> None:
    serializable = {key: value for key, value in metrics.items() if key not in {"probabilities", "predictions"}}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    reports_dir = make_reports_dir()
    models_dir = make_models_dir()

    raw_df = load_dataset()
    cleaned_df = clean_dataset(raw_df)
    features, target = split_features_target(raw_df)
    x_train, x_test, y_train, y_test = split_data(features, target)

    preprocessor = build_preprocessor(x_train)

    logreg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    logreg_pipeline.fit(x_train, y_train)
    logreg_metrics = evaluate_model("logistic_regression", logreg_pipeline, x_test, y_test)

    xgb_preprocessor = build_preprocessor(x_train)
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", xgb_preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [150, 250],
        "model__max_depth": [3, 4],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__min_child_weight": [1, 5],
    }

    grid_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(x_train, y_train)

    best_xgb_pipeline = grid_search.best_estimator_
    xgb_metrics = evaluate_model("xgboost", best_xgb_pipeline, x_test, y_test)

    logreg_model_path = models_dir / "churn_logreg_pipeline.joblib"
    xgb_model_path = models_dir / "churn_xgb_bundle.joblib"
    joblib.dump(logreg_pipeline, logreg_model_path)
    joblib.dump(
        {
            "pipeline": best_xgb_pipeline,
            "feature_names": feature_names(best_xgb_pipeline.named_steps["preprocessor"]),
            "best_params": grid_search.best_params_,
            "cv_roc_auc": float(grid_search.best_score_),
        },
        xgb_model_path,
    )

    plot_confusion_matrix(np.array(xgb_metrics["confusion_matrix"]), reports_dir / "confusion_matrix_xgb.png")
    plot_pr_curve(y_test, xgb_metrics["probabilities"], reports_dir / "pr_curve_xgb.png")

    metrics_report = {
        "logistic_regression": {
            key: value
            for key, value in logreg_metrics.items()
            if key not in {"probabilities", "predictions"}
        },
        "xgboost": {key: value for key, value in xgb_metrics.items() if key not in {"probabilities", "predictions"}},
        "best_xgb_params": grid_search.best_params_,
        "best_xgb_cv_roc_auc": float(grid_search.best_score_),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(len(feature_names(best_xgb_pipeline.named_steps["preprocessor"]))),
    }

    save_metrics_report(metrics_report, reports_dir / "metrics.json")

    metrics_text = [
        "Customer Churn Predictor Metrics",
        "=================================",
        "",
        "Logistic Regression",
        f"  ROC-AUC: {logreg_metrics['roc_auc']:.4f}",
        f"  Precision (churn): {logreg_metrics['precision_churn']:.4f}",
        f"  Recall (churn): {logreg_metrics['recall_churn']:.4f}",
        f"  F1 (churn): {logreg_metrics['f1_churn']:.4f}",
        "",
        "XGBoost",
        f"  ROC-AUC: {xgb_metrics['roc_auc']:.4f}",
        f"  Precision (churn): {xgb_metrics['precision_churn']:.4f}",
        f"  Recall (churn): {xgb_metrics['recall_churn']:.4f}",
        f"  F1 (churn): {xgb_metrics['f1_churn']:.4f}",
        f"  Average precision: {xgb_metrics['average_precision']:.4f}",
        "",
        f"Best XGBoost params: {grid_search.best_params_}",
        f"Scale pos weight: {scale_pos_weight:.4f}",
        f"Saved model bundle: {xgb_model_path}",
    ]
    (reports_dir / "metrics.txt").write_text("\n".join(metrics_text), encoding="utf-8")

    print("\n".join(metrics_text))

    cleaned_df.to_csv(reports_dir / "cleaned_dataset_snapshot.csv", index=False)


if __name__ == "__main__":
    main()
