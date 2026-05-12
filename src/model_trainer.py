import os
from datetime import datetime
import warnings
from typing import Dict, List

import joblib
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from evaluator import evaluate_classification, evaluate_regression


def get_candidate_models(task_type: str):
    """Return baseline estimators for the requested task type."""
    if task_type == "classification":
        return [
            ("log_reg", LogisticRegression(max_iter=500)),
            ("rf_clf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("gb_clf", GradientBoostingClassifier(random_state=42)),
        ]

    return [
        ("lin_reg", LinearRegression()),
        ("rf_reg", RandomForestRegressor(n_estimators=200, random_state=42)),
        ("gb_reg", GradientBoostingRegressor(random_state=42)),
    ]


def train_models(X_train, y_train, problem_type: str) -> Dict[str, object]:
    """Train baseline models and return those that succeed."""
    if problem_type == "classification":
        candidates = [
            ("logistic_regression", LogisticRegression(max_iter=500)),
            (
                "random_forest_classifier",
                RandomForestClassifier(n_estimators=200, random_state=42),
            ),
            (
                "gradient_boosting_classifier",
                GradientBoostingClassifier(random_state=42),
            ),
        ]
    elif problem_type == "regression":
        candidates = [
            ("linear_regression", LinearRegression()),
            (
                "random_forest_regressor",
                RandomForestRegressor(n_estimators=200, random_state=42),
            ),
            (
                "gradient_boosting_regressor",
                GradientBoostingRegressor(random_state=42),
            ),
        ]
    else:
        raise ValueError("problem_type must be 'classification' or 'regression'.")

    trained_models: Dict[str, object] = {}
    for name, model in candidates:
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as exc:
            warnings.warn(f"Model '{name}' failed to train: {exc}")

    return trained_models


def save_model(model, model_name: str) -> str:
    """Persist a trained model to the models folder with a timestamp."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace(" ", "_").lower()
    filename = f"{safe_name}_{timestamp}.joblib"
    file_path = os.path.join(model_dir, filename)

    joblib.dump(model, file_path)
    return file_path


def _metric_score(metrics: Dict[str, float], metric: str) -> float:
    """Normalize metrics so higher is better for comparisons."""
    if metric not in metrics:
        raise ValueError(f"Metric '{metric}' is not available in results.")
    value = metrics[metric]
    if metric in {"rmse", "mae"}:
        return -value
    return value


def train_and_select_model(
    df,
    target_col: str,
    preprocessor,
    task_type: str,
    metric: str,
    model_dir: str,
):
    """Train pipeline models, score them, and save the best model."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = None
    if task_type == "classification" and y.nunique() > 1:
        min_class_count = y.value_counts().min()
        if min_class_count >= 2:
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    best = None
    all_results: List[Dict[str, float]] = []

    for name, model in get_candidate_models(task_type):
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        if task_type == "classification":
            metrics = evaluate_classification(pipeline, X_test, y_test)
        else:
            metrics = evaluate_regression(pipeline, X_test, y_test)

        result_row = {"model": name}
        result_row.update(metrics)
        all_results.append(result_row)

        score = _metric_score(metrics, metric)
        if best is None or score > best["score"]:
            best = {"name": name, "model": pipeline, "score": score, "metrics": metrics}

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{best['name']}.joblib")
    joblib.dump(best["model"], model_path)

    return best, all_results, model_path
