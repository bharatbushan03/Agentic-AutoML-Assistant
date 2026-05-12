from datetime import datetime
from typing import Dict, List


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
