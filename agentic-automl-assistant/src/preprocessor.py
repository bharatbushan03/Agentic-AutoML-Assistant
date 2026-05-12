from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    df: pd.DataFrame, target_col: str
) -> Tuple[ColumnTransformer, List[str], List[str]]:
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
