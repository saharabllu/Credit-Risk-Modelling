"""Data loading and preprocessing utilities for the German Credit dataset."""

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

RANDOM_STATE = 42
TARGET_COL = "default"


def load_german_credit() -> pd.DataFrame:
    """Fetch German Credit (OpenML id 31) and encode the target as binary default.

    The original `class` column (good/bad) is replaced by `default`, where
    1 = bad credit (default) and 0 = good credit (performing).
    """
    raw = fetch_openml(data_id=31, as_frame=True, parser="auto").frame
    df = raw.copy()
    df[TARGET_COL] = (df["class"] == "bad").astype(int)
    df = df.drop(columns="class")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors (X) from the binary default indicator (y)."""
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return df[feature_cols], df[TARGET_COL]


def get_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) based on dtype."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def make_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """One-hot encode categorical columns; optionally standardise numeric ones.

    Scaling is only applied to logistic regression. Decision trees are
    invariant to monotonic transformations, so `scale_numeric=False`
    is passed when building tree pipelines.
    """
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", numeric_step, numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])


def split_train_test(X, y, test_size: float = 0.20, random_state: int = RANDOM_STATE):
    """Stratified train/test split that preserves the default rate."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
