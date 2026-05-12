from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a one-hot encoder compatible with older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocess_data(
    df: pd.DataFrame, target_column: str
) -> Tuple[object, pd.Series, Dict[str, object]]:
    """Split features/target, impute missing values, and one-hot encode."""
    if target_column not in df.columns:
        raise ValueError("Target column not found in dataframe.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns found.")

    preprocessor = ColumnTransformer(transformers)
    X_processed = preprocessor.fit_transform(X)

    feature_names = None
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = None

    details = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "feature_names": list(feature_names) if feature_names is not None else None,
        "preprocessor": preprocessor,
        "processed_shape": X_processed.shape,
    }

    return X_processed, y, details


def build_preprocessor(
    df: pd.DataFrame, target_col: str
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build a ColumnTransformer for numeric and categorical features."""
    features = df.drop(columns=[target_col])
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in features.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns found.")

    preprocessor = ColumnTransformer(transformers)
    return preprocessor, numeric_cols, categorical_cols
