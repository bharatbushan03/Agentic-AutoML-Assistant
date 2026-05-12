import os
from typing import Dict, Optional

import pandas as pd


def _format_evaluation_results(evaluation_results: Optional[pd.DataFrame]) -> str:
    if evaluation_results is None or evaluation_results.empty:
        return "No evaluation results available."

    headers = evaluation_results.columns.tolist()
    lines = [" | ".join(headers)]
    for row in evaluation_results.itertuples(index=False):
        lines.append(" | ".join(str(value) for value in row))
    return "\n".join(lines)


def _get_best_row(
    evaluation_results: Optional[pd.DataFrame], best_model_name: Optional[str]
) -> Optional[Dict[str, object]]:
    if evaluation_results is None or evaluation_results.empty or not best_model_name:
        return None
    row = evaluation_results[evaluation_results["model"] == best_model_name]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _build_context(
    dataset_analysis: Dict,
    problem_type: str,
    target_column: str,
    evaluation_results: Optional[pd.DataFrame],
    best_model_name: Optional[str],
) -> str:
    missing_values = dataset_analysis.get("missing_values", {})
    missing_pairs = [
        f"{col}: {count}" for col, count in missing_values.items() if count
    ]
    missing_summary = ", ".join(missing_pairs) if missing_pairs else "None"

    model_names = []
    if evaluation_results is not None and not evaluation_results.empty:
        model_names = evaluation_results["model"].astype(str).tolist()

    context_lines = [
        "Dataset summary:",
        f"- Rows: {dataset_analysis.get('rows', 0)}",
        f"- Columns: {dataset_analysis.get('columns', 0)}",
        f"- Target column: {target_column}",
        f"- Problem type: {problem_type}",
        f"- Missing values: {missing_summary}",
        f"- Models trained: {', '.join(model_names) if model_names else 'None'}",
        "",
        "Evaluation results:",
        _format_evaluation_results(evaluation_results),
        "",
        f"Best model: {best_model_name or 'None'}",
    ]

    return "\n".join(context_lines)


def _get_llm_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    return OpenAI(api_key=api_key)


def _fallback_answer(
    question: str,
    dataset_analysis: Dict,
    problem_type: str,
    target_column: str,
    evaluation_results: Optional[pd.DataFrame],
    best_model_name: Optional[str],
) -> str:
    normalized = question.lower()
    best_row = _get_best_row(evaluation_results, best_model_name)

    if "dataset" in normalized and ("contain" in normalized or "what" in normalized):
        return (
            f"The dataset has {dataset_analysis.get('rows', 0)} rows and "
            f"{dataset_analysis.get('columns', 0)} columns. The target column is "
            f"'{target_column}', and the detected problem type is {problem_type}."
        )
    if "best" in normalized and "model" in normalized:
        if best_model_name:
            return f"The best model is '{best_model_name}'."
        return "No best model could be identified from the current results."
    if "why" in normalized and "better" in normalized:
        if best_row:
            metric = "f1" if problem_type == "classification" else "r2"
            metric_value = best_row.get(metric, "")
            return (
                f"The model '{best_model_name}' scored best on {metric} "
                f"({metric_value})."
            )
        return "The best model is not available yet to compare performance."
    if "improve" in normalized:
        return (
            "Try feature engineering, hyperparameter tuning, and cross-validation. "
            "You can also collect more data or handle class imbalance if needed."
        )
    if "feature" in normalized and "important" in normalized:
        return (
            "Feature importance is not computed yet. Consider using tree-based "
            "feature_importances_ or permutation importance to estimate it."
        )

    return (
        "Ask about the dataset summary, model performance, best model, or ways to "
        "improve the results."
    )


def answer_question(
    question: str,
    dataset_analysis: Dict,
    problem_type: str,
    target_column: str,
    evaluation_results: Optional[pd.DataFrame],
    best_model_name: Optional[str],
) -> str:
    context = _build_context(
        dataset_analysis=dataset_analysis,
        problem_type=problem_type,
        target_column=target_column,
        evaluation_results=evaluation_results,
        best_model_name=best_model_name,
    )

    client = _get_llm_client()
    if client is not None:
        try:
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AutoML assistant. Use the provided "
                            "context to answer user questions concisely. If the "
                            "context is missing details, state that and give a "
                            "practical next step."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return _fallback_answer(
                question,
                dataset_analysis,
                problem_type,
                target_column,
                evaluation_results,
                best_model_name,
            )

    return _fallback_answer(
        question,
        dataset_analysis,
        problem_type,
        target_column,
        evaluation_results,
        best_model_name,
    )
