"""Streamlit app for the Customer Churn Predictor."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.preprocess import clean_dataset, load_dataset


st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


CUSTOMER_FEATURES = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "tenure": (0, 72),
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "MonthlyCharges": (0.0, 150.0),
    "TotalCharges": (0.0, 10000.0),
}


@st.cache_resource
def load_bundle():
    bundle_path = Path(__file__).resolve().parent / "models" / "churn_xgb_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError("Model bundle not found. Run src/train.py first.")
    return joblib.load(bundle_path)


@st.cache_data
def load_reference_data():
    return clean_dataset(load_dataset())


def build_customer_row(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([inputs])


def predict_customer(bundle, customer_row: pd.DataFrame):
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = bundle.get("feature_names")

    transformed = preprocessor.transform(customer_row)
    probability = float(model.predict_proba(transformed)[:, 1][0])

    reference_features = load_reference_data().drop(columns=["Churn", "customerID"])
    background = shap.sample(preprocessor.transform(reference_features), 200, random_state=42)
    explainer = shap.TreeExplainer(model, data=background)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if feature_names is None:
        feature_names = list(preprocessor.get_feature_names_out())
    contributions = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_values[0],
            "absolute_contribution": np.abs(shap_values[0]),
        }
    ).sort_values("absolute_contribution", ascending=False)
    return probability, contributions


def score_batch(bundle, batch_df: pd.DataFrame) -> pd.DataFrame:
    pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba(batch_df)[:, 1]
    scored = batch_df.copy()
    scored["churn_probability"] = probabilities
    scored["churn_prediction"] = np.where(probabilities >= 0.5, "Yes", "No")
    return scored


def customer_form(reference_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Single customer scoring")
    st.caption("Enter a customer profile to get churn probability and SHAP drivers.")

    with st.form("customer_form"):
        cols = st.columns(2)
        inputs = {}
        feature_order = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
        ]
        for index, feature in enumerate(feature_order):
            column = cols[index % 2]
            default_series = reference_df[feature] if feature in reference_df.columns else None
            if feature == "SeniorCitizen":
                options = CUSTOMER_FEATURES[feature]
                default_value = int(reference_df[feature].mode(dropna=True).iloc[0]) if default_series is not None and not default_series.mode(dropna=True).empty else 0
                inputs[feature] = int(column.selectbox(feature, options, index=options.index(default_value)))
            elif isinstance(CUSTOMER_FEATURES[feature], list):
                options = CUSTOMER_FEATURES[feature]
                default_value = options[0]
                if default_series is not None:
                    most_common = default_series.mode(dropna=True)
                    if not most_common.empty and most_common.iloc[0] in options:
                        default_value = most_common.iloc[0]
                inputs[feature] = column.selectbox(feature, options, index=options.index(default_value))
            else:
                min_value, max_value = CUSTOMER_FEATURES[feature]
                default_value = float(reference_df[feature].median()) if default_series is not None else float(min_value)
                # All of number_input's numeric args must share one type; value/
                # min/max are floats below, so keep step a float too (tenure steps by 1).
                step = 1.0 if feature == "tenure" else 0.1
                inputs[feature] = column.number_input(
                    feature,
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=float(default_value),
                    step=step,
                )

        submitted = st.form_submit_button("Predict churn")

    if submitted:
        return build_customer_row(inputs)
    return pd.DataFrame()


def main() -> None:
    st.title("Customer Churn Predictor")
    st.write("Predict whether a telecom customer is likely to churn using a tuned XGBoost model and SHAP explanations.")

    try:
        bundle = load_bundle()
        reference_df = load_reference_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    tab_single, tab_batch = st.tabs(["Single customer", "Batch scoring"])

    with tab_single:
        customer_row = customer_form(reference_df)
        if not customer_row.empty:
            probability, contributions = predict_customer(bundle, customer_row)
            st.metric("Churn probability", f"{probability:.1%}")
            st.progress(probability)
            st.subheader("Top SHAP drivers")
            st.dataframe(contributions.head(10), use_container_width=True)
            st.caption("Positive SHAP values push the prediction toward churn; negative values push it away.")

    with tab_batch:
        st.subheader("Upload a CSV for batch scoring")
        st.caption("The CSV should contain the raw customer features used by the model. If Churn is present, it will be ignored.")
        upload = st.file_uploader("Choose a CSV file", type=["csv"])

        if upload is not None:
            batch_df = pd.read_csv(upload)
            if "Churn" in batch_df.columns:
                batch_df = batch_df.drop(columns=["Churn"])
            if "customerID" not in batch_df.columns:
                batch_df.insert(0, "customerID", [f"row_{index + 1}" for index in range(len(batch_df))])

            try:
                scored = score_batch(bundle, batch_df.drop(columns=["customerID"]))
                scored.insert(0, "customerID", batch_df["customerID"].values)
                st.dataframe(scored, use_container_width=True)

                csv_bytes = scored.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download scored results",
                    data=csv_bytes,
                    file_name="churn_scores.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Scoring failed: {exc}")

    with st.expander("Project notes"):
        st.markdown(
            """
            - Model bundle: `models/churn_xgb_bundle.joblib`
            - Reports: `reports/`
            - Training script: `src/train.py`
            - Explainability script: `src/explain.py`
            """
        )


if __name__ == "__main__":
    main()
