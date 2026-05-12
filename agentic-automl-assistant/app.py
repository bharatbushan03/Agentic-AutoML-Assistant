import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_analyzer import analyze_dataframe, guess_task_type
from data_loader import load_csv
from agent import run_automl

MODEL_DIR = os.path.join(ROOT_DIR, "models")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

st.set_page_config(page_title="Agentic AutoML Assistant", layout="wide")
st.title("Agentic AutoML Assistant")
st.write("Upload a CSV dataset and train baseline models automatically.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

try:
    df = load_csv(uploaded_file)
except Exception as exc:
    st.error(f"Failed to read CSV: {exc}")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(50), use_container_width=True)

summary = analyze_dataframe(df)
with st.expander("Dataset summary"):
    st.json(summary)

if df.shape[1] < 2:
    st.warning("Dataset needs at least 2 columns (features + target).")
    st.stop()

target_col = st.selectbox("Select target column", df.columns.tolist())

inferred_task = guess_task_type(df[target_col])
task_type = st.radio(
    "Task type",
    ["classification", "regression"],
    index=0 if inferred_task == "classification" else 1,
    help="Auto-detected from target column, but you can override.",
)

st.subheader("Target distribution")
fig, ax = plt.subplots()
if pd.api.types.is_numeric_dtype(df[target_col]):
    ax.hist(df[target_col].dropna(), bins=20)
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
else:
    value_counts = df[target_col].astype(str).value_counts().head(20)
    ax.bar(range(len(value_counts)), value_counts.values)
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index, rotation=45, ha="right")
    ax.set_ylabel("Count")
ax.set_title(f"{target_col}")
st.pyplot(fig)

if task_type == "classification":
    metric = st.selectbox(
        "Primary metric", ["f1", "accuracy", "precision", "recall"], index=0
    )
else:
    metric = st.selectbox("Primary metric", ["r2", "rmse", "mae"], index=0)

if task_type == "regression" and not pd.api.types.is_numeric_dtype(df[target_col]):
    st.error("Regression requires a numeric target column.")
    st.stop()

if st.button("Train models"):
    with st.spinner("Training models..."):
        results = run_automl(
            df=df,
            target_col=target_col,
            task_type=task_type,
            metric=metric,
            model_dir=MODEL_DIR,
            report_dir=REPORT_DIR,
            dataset_summary=summary,
        )

    st.success("Training complete.")
    st.subheader("Best model")
    st.write(f"Model: {results['best_name']}")
    st.json(results["best_metrics"])

    st.subheader("All model results")
    st.dataframe(results["all_results"], use_container_width=True)

    st.subheader("Artifacts")
    st.write(f"Saved model: {results['model_path']}")
    st.write(f"Saved report: {results['report_path']}")

    st.download_button(
        "Download report",
        data=results["report_markdown"],
        file_name=os.path.basename(results["report_path"]),
        mime="text/markdown",
    )
