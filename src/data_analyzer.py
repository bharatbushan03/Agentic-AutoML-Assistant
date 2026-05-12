import pandas as pd


def _to_python_value(value):
    """Convert numpy scalars to Python-native values."""
    if hasattr(value, "item"):
        return value.item()
    return value


def _normalize_nested_dict(data: dict) -> dict:
    """Normalize nested dict values into JSON-serializable types."""
    normalized = {}
    for key, values in data.items():
        normalized[key] = {k: _to_python_value(v) for k, v in values.items()}
    return normalized


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Return dataset structure, missing values, and summary statistics."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    numeric_summary = {}
    if numeric_cols:
        numeric_summary = _normalize_nested_dict(df[numeric_cols].describe().to_dict())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "numerical_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numerical_summary": numeric_summary,
        "unique_values": df.nunique(dropna=True).to_dict(),
    }


def analyze_dataframe(df: pd.DataFrame) -> dict:
    """Return a compact dataset summary for quick inspection."""
    dataset = analyze_dataset(df)

    return {
        "rows": dataset["rows"],
        "columns": dataset["columns"],
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": dataset["missing_values"],
        "dtypes": dataset["dtypes"],
        "numeric_columns": dataset["numerical_columns"],
        "categorical_columns": dataset["categorical_columns"],
    }


def guess_task_type(target_series: pd.Series, classification_threshold: int = 20) -> str:
    """Guess classification vs regression based on target cardinality."""
    if pd.api.types.is_numeric_dtype(target_series):
        unique_count = target_series.nunique(dropna=True)
        if unique_count <= classification_threshold:
            return "classification"
        return "regression"
    return "classification"
