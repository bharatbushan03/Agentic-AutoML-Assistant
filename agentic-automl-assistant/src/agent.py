from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from data_analyzer import analyze_dataset
from evaluator import evaluate_models
from model_trainer import save_model, train_models
from preprocessor import preprocess_data
from report_generator import generate_report


class AutoMLAgent:
    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def detect_problem_type(self, target_series: pd.Series) -> str:
        if pd.api.types.is_object_dtype(target_series) or pd.api.types.is_categorical_dtype(
            target_series
        ):
            return "classification"
        if pd.api.types.is_numeric_dtype(target_series):
            unique_count = int(target_series.nunique(dropna=True))
            return "classification" if unique_count <= 20 else "regression"
        return "classification"

    def run(self, df: pd.DataFrame, target_column: str) -> Dict[str, object]:
        dataset_analysis = analyze_dataset(df)
        y = df[target_column]
        problem_type = self.detect_problem_type(y)

        X_processed, y_processed, prep_details = preprocess_data(df, target_column)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed,
            y_processed,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        models = train_models(X_train, y_train, problem_type)
        evaluation_results = pd.DataFrame()
        best_model_name: Optional[str] = None
        if models:
            evaluation_results, best_model_name = evaluate_models(
                models, X_test, y_test, problem_type
            )

        saved_model_path = None
        if best_model_name:
            best_model = models.get(best_model_name)
            if best_model is not None:
                saved_model_path = save_model(best_model, best_model_name)

        report_path = generate_report(
            dataset_analysis=dataset_analysis,
            problem_type=problem_type,
            target_column=target_column,
            evaluation_results=evaluation_results,
            best_model_name=best_model_name,
        )

        return {
            "dataset_analysis": dataset_analysis,
            "problem_type": problem_type,
            "evaluation_results": evaluation_results,
            "best_model_name": best_model_name,
            "saved_model_path": saved_model_path,
            "report_path": report_path,
            "processed_shape": prep_details.get("processed_shape"),
            "train_rows": X_train.shape[0],
            "test_rows": X_test.shape[0],
            "final_features": X_train.shape[1],
        }
