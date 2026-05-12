import warnings
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def evaluate_classification(model, X_test, y_test) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_regression(model, X_test, y_test) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def evaluate_models(
    models: Dict[str, object], X_test, y_test, problem_type: str
) -> Tuple[pd.DataFrame, Optional[str]]:
    if not models:
        return pd.DataFrame(), None

    rows = []
    for name, model in models.items():
        try:
            if problem_type == "classification":
                metrics = evaluate_classification(model, X_test, y_test)
            elif problem_type == "regression":
                metrics = evaluate_regression(model, X_test, y_test)
            else:
                raise ValueError("problem_type must be 'classification' or 'regression'.")

            row = {"model": name}
            row.update(metrics)
            rows.append(row)
        except Exception as exc:
            warnings.warn(f"Model '{name}' evaluation failed: {exc}")

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df, None

    if problem_type == "classification":
        best_index = results_df["f1"].astype(float).idxmax()
    else:
        best_index = results_df["r2"].astype(float).idxmax()

    best_name = results_df.loc[best_index, "model"]
    return results_df, best_name
