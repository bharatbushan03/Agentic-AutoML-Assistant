from typing import Callable, Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from data_analyzer import analyze_dataset
from evaluator import evaluate_models
from model_trainer import save_model, train_models
from preprocessor import preprocess_data
from report_generator import generate_report


class AutoMLAgent:
    """Coordinate the AutoML workflow steps."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        max_target_missing_ratio: float = 0.5,
    ) -> None:
        """Initialize the agent with split and validation settings."""
        self.test_size = test_size
        self.random_state = random_state
        self.max_target_missing_ratio = max_target_missing_ratio

    def _emit(
        self,
        messages: List[str],
        message: str,
        on_step: Optional[Callable[[str], None]],
    ) -> None:
        """Record a progress message and optionally stream it to the UI."""
        messages.append(message)
        if on_step:
            on_step(message)

    def detect_problem_type(self, target_series: pd.Series) -> str:
        """Infer classification or regression from the target series."""
        if pd.api.types.is_object_dtype(target_series) or isinstance(
            target_series.dtype, pd.CategoricalDtype
        ):
            return "classification"
        if pd.api.types.is_numeric_dtype(target_series):
            unique_count = int(target_series.nunique(dropna=True))
            return "classification" if unique_count <= 20 else "regression"
        return "classification"

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, object]:
        """Run the full pipeline and return results and artifacts."""
        if df is None or df.empty or df.shape[1] == 0:
            raise ValueError("The uploaded CSV is empty or has no columns.")
        if df.shape[1] < 2:
            raise ValueError("Dataset must contain at least 2 columns (features + target).")
        if target_column not in df.columns:
            raise ValueError("Target column not found in the dataset.")

        target_series = df[target_column]
        target_missing_ratio = float(target_series.isna().mean())
        if target_missing_ratio > self.max_target_missing_ratio:
            raise ValueError(
                "Target column has too many missing values. "
                "Please choose another column or clean the data."
            )

        df_clean = df.loc[target_series.notna()].copy()
        if df_clean.shape[0] < 2:
            raise ValueError(
                "Not enough rows after removing missing target values."
            )

        messages: List[str] = []
        self._emit(messages, "Analyzing dataset structure...", on_step)
        dataset_analysis = analyze_dataset(df)

        self._emit(messages, "Detecting problem type...", on_step)
        y = df_clean[target_column]
        problem_type = self.detect_problem_type(y)

        self._emit(
            messages,
            "Preprocessing numerical and categorical columns...",
            on_step,
        )
        try:
            X_processed, y_processed, prep_details = preprocess_data(
                df_clean, target_column
            )
        except ValueError as exc:
            if "No usable feature columns" in str(exc):
                raise ValueError(
                    "Dataset has no usable features after preprocessing. "
                    "Add feature columns or fix data types."
                ) from exc
            raise
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed,
            y_processed,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self._emit(messages, "Training multiple machine learning models...", on_step)
        models = train_models(X_train, y_train, problem_type)
        if not models:
            raise RuntimeError(
                "Model training failed. No models could be trained on this dataset."
            )

        self._emit(messages, "Evaluating model performance...", on_step)
        evaluation_results = pd.DataFrame()
        best_model_name: Optional[str] = None
        if models:
            evaluation_results, best_model_name = evaluate_models(
                models, X_test, y_test, problem_type
            )

        self._emit(messages, "Selecting the best model...", on_step)
        saved_model_path = None
        if best_model_name:
            self._emit(messages, "Saving model...", on_step)
            best_model = models.get(best_model_name)
            if best_model is not None:
                saved_model_path = save_model(best_model, best_model_name)
        else:
            self._emit(messages, "Saving model...", on_step)

        self._emit(messages, "Generating report...", on_step)
        report_path = None
        report_error = None
        try:
            report_path = generate_report(
                dataset_analysis=dataset_analysis,
                problem_type=problem_type,
                target_column=target_column,
                evaluation_results=evaluation_results,
                best_model_name=best_model_name,
            )
        except Exception as exc:
            report_error = (
                "Report generation failed. Please try again or check the logs."
            )

        return {
            "dataset_analysis": dataset_analysis,
            "problem_type": problem_type,
            "evaluation_results": evaluation_results,
            "best_model_name": best_model_name,
            "saved_model_path": saved_model_path,
            "report_path": report_path,
            "report_error": report_error,
            "processed_shape": prep_details.get("processed_shape"),
            "train_rows": X_train.shape[0],
            "test_rows": X_test.shape[0],
            "final_features": X_train.shape[1],
            "messages": messages,
        }
