"""Generate SHAP explanations for the churn model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from preprocess import clean_dataset, feature_names, load_dataset, make_reports_dir, split_data, split_features_target


def load_bundle():
    bundle_path = Path(__file__).resolve().parents[1] / "models" / "churn_xgb_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError("Model bundle not found. Run src/train.py first.")
    return joblib.load(bundle_path)


def pick_shap_values(explainer, transformed_row):
    shap_values = explainer.shap_values(transformed_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[0]
    return shap_values, expected_value


def main() -> None:
    reports_dir = make_reports_dir()
    bundle = load_bundle()
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    raw_df = load_dataset()
    cleaned_df = clean_dataset(raw_df)
    features, target = split_features_target(raw_df)
    x_train, x_test, y_train, y_test = split_data(features, target)

    transformed_train = preprocessor.transform(x_train)
    transformed_test = preprocessor.transform(x_test)
    names = bundle.get("feature_names") or feature_names(preprocessor)

    background = shap.sample(transformed_train, 200, random_state=42)
    explainer = shap.TreeExplainer(model, data=background)

    shap_values = explainer.shap_values(transformed_train)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap.summary_plot(shap_values, transformed_train, feature_names=names, show=False)
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    selected_index = int(np.argmax(probabilities))
    selected_row = transformed_test[selected_index : selected_index + 1]
    selected_shap_values, expected_value = pick_shap_values(explainer, selected_row)

    shap.waterfall_plot(
        shap.Explanation(
            values=selected_shap_values[0],
            base_values=expected_value,
            data=selected_row[0],
            feature_names=names,
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_waterfall_customer.png", dpi=200, bbox_inches="tight")
    plt.close()

    top_features = np.argsort(np.abs(selected_shap_values[0]))[::-1][:10]
    top_driver_table = [
        {
            "feature": names[index],
            "shap_value": float(selected_shap_values[0][index]),
            "absolute_contribution": float(abs(selected_shap_values[0][index])),
        }
        for index in top_features
    ]
    (reports_dir / "shap_top_drivers.json").write_text(json.dumps(top_driver_table, indent=2), encoding="utf-8")

    cleaned_df.to_csv(reports_dir / "explain_cleaned_snapshot.csv", index=False)
    print("Saved SHAP summary and waterfall plots to reports/")


if __name__ == "__main__":
    main()
