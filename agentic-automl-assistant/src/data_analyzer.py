import pandas as pd


def analyze_dataframe(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }


def guess_task_type(target_series: pd.Series, classification_threshold: int = 20) -> str:
    if pd.api.types.is_numeric_dtype(target_series):
        unique_count = target_series.nunique(dropna=True)
        if unique_count <= classification_threshold:
            return "classification"
        return "regression"
    return "classification"
