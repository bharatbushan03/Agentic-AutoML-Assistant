import os
from datetime import datetime
from typing import Dict

from data_analyzer import analyze_dataframe
from model_trainer import train_and_select_model
from preprocessor import build_preprocessor
from report_generator import generate_markdown_report


def run_automl(
    df,
    target_col: str,
    task_type: str,
    metric: str,
    model_dir: str,
    report_dir: str,
    dataset_summary: Dict = None,
):
    if dataset_summary is None:
        dataset_summary = analyze_dataframe(df)

    preprocessor, _, _ = build_preprocessor(df, target_col)
    best, all_results, model_path = train_and_select_model(
        df,
        target_col=target_col,
        preprocessor=preprocessor,
        task_type=task_type,
        metric=metric,
        model_dir=model_dir,
    )

    report_markdown = generate_markdown_report(
        task_type=task_type,
        target_col=target_col,
        best_name=best["name"],
        best_metrics=best["metrics"],
        all_results=all_results,
        dataset_summary=dataset_summary,
        model_path=model_path,
    )

    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"report_{timestamp}.md")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_markdown)

    return {
        "best_name": best["name"],
        "best_metrics": best["metrics"],
        "all_results": all_results,
        "model_path": model_path,
        "report_path": report_path,
        "report_markdown": report_markdown,
    }
