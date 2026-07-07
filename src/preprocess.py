"""Shared preprocessing utilities for the churn project."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, urlretrieve

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_path() -> Path:
    return project_root() / "data" / "Telco-Customer-Churn.csv"


def load_dataset(path: Path | None = None, download_if_missing: bool = True) -> pd.DataFrame:
    dataset_path = Path(path) if path else data_path()

    if not dataset_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        try:
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(DATA_URL, timeout=30) as response:
                if response.status != 200:
                    raise HTTPError(DATA_URL, response.status, "Unexpected HTTP status", hdrs=None, fp=None)
            urlretrieve(DATA_URL, dataset_path)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                "Failed to download the dataset automatically.\n"
                f"Kaggle fallback: {KAGGLE_URL}\n"
                f"Expected local path: {dataset_path}"
            ) from exc

    return pd.read_csv(dataset_path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)
    return cleaned


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    cleaned = clean_dataset(df)
    features = cleaned.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    target = cleaned[TARGET_COLUMN]
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [column for column in NUMERIC_FEATURES if column in features.columns]
    categorical_features = [column for column in features.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def make_reports_dir() -> Path:
    reports_dir = project_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def make_models_dir() -> Path:
    models_dir = project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

