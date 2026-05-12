import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


def generate_markdown_report(
    task_type: str,
    target_col: str,
    best_name: str,
    best_metrics: Dict[str, float],
    all_results: List[Dict[str, float]],
    dataset_summary: Dict,
    model_path: str,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_names = [key for key in best_metrics.keys()]

    lines = [
        "# AutoML Report",
        "",
        f"- Generated: {timestamp}",
        f"- Task: {task_type}",
        f"- Target column: {target_col}",
        f"- Best model: {best_name}",
        f"- Model path: {model_path}",
        "",
        "## Best model metrics",
    ]

    for key, value in best_metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        lines.append(f"- {key}: {formatted}")

    lines.append("")
    lines.append("## All model results")
    lines.append("| Model | " + " | ".join(metric_names) + " |")
    lines.append("| " + " | ".join(["---"] * (len(metric_names) + 1)) + " |")

    for row in all_results:
        values = []
        for name in metric_names:
            value = row.get(name, "")
            values.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
        lines.append("| " + row["model"] + " | " + " | ".join(values) + " |")

    lines.append("")
    lines.append("## Dataset summary")
    lines.append(f"- Rows: {dataset_summary.get('rows', 0)}")
    lines.append(f"- Columns: {dataset_summary.get('columns', 0)}")
    lines.append(f"- Missing total: {dataset_summary.get('missing_total', 0)}")

    numeric = dataset_summary.get("numeric_columns", [])
    categorical = dataset_summary.get("categorical_columns", [])
    lines.append(f"- Numeric columns: {', '.join(numeric) if numeric else 'None'}")
    lines.append(f"- Categorical columns: {', '.join(categorical) if categorical else 'None'}")

    return "\n".join(lines)


def _markdown_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    if not headers:
        return []
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def generate_report(
    dataset_analysis: Dict,
    problem_type: str,
    target_column: str,
    evaluation_results: Optional[pd.DataFrame],
    best_model_name: Optional[str],
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Agentic AutoML Assistant Report",
        "",
        f"- Generated: {timestamp}",
        "",
        "## Dataset Summary",
        f"- Rows: {dataset_analysis.get('rows', 0)}",
        f"- Columns: {dataset_analysis.get('columns', 0)}",
    ]

    numeric_cols = dataset_analysis.get("numerical_columns", [])
    categorical_cols = dataset_analysis.get("categorical_columns", [])
    lines.append(
        f"- Numerical columns: {', '.join(numeric_cols) if numeric_cols else 'None'}"
    )
    lines.append(
        f"- Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'None'}"
    )
    lines.append("")

    lines.append("## Target Column")
    lines.append(f"- {target_column}")
    lines.append("")

    lines.append("## Detected Problem Type")
    lines.append(f"- {problem_type}")
    lines.append("")

    lines.append("## Missing Value Summary")
    missing_values = dataset_analysis.get("missing_values", {})
    if missing_values:
        rows = [
            [str(col), str(count)] for col, count in missing_values.items()
        ]
        lines.extend(_markdown_table(["Column", "Missing Values"], rows))
    else:
        lines.append("No missing values found.")
    lines.append("")

    lines.append("## Models Trained")
    model_names: List[str] = []
    if evaluation_results is not None and not evaluation_results.empty:
        model_names = evaluation_results["model"].astype(str).tolist()
    if model_names:
        lines.extend([f"- {name}" for name in model_names])
    else:
        lines.append("No models were trained.")
")
    lines.append("")

    lines.append("## Evaluation Results")
    if evaluation_results is not None and not evaluation_results.empty:
        headers = [str(col) for col in evaluation_results.columns]
        rows = [
            [str(value) for value in row]
            for row in evaluation_results.values.tolist()
        ]
        lines.extend(_markdown_table(headers, rows))
    else:
        lines.append("No evaluation results available.")
    lines.append("")

    lines.append("## Best Model")
    if best_model_name:
        lines.append(f"- {best_model_name}")
    else:
        lines.append("No best model selected.")
    lines.append("")

    lines.append("## Conclusion")
    if best_model_name:
        lines.append(
            f"The best model based on the selected metric is {best_model_name}. "
            "Consider validating results with cross-validation or additional data."
        )
    else:
        lines.append(
            "No best model could be identified. Review the dataset and training "
            "configuration for potential issues."
        )

    report_content = "\n".join(lines)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_dir = os.path.join(root_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = os.path.join(report_dir, filename)

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_content)

    return report_path
