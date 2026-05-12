import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError, ParserError

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from agent import AutoMLAgent
from assistant import answer_question

st.set_page_config(page_title="Agentic AutoML Assistant", layout="wide")
st.title("Agentic AutoML Assistant")
st.write("Upload a CSV file to preview your dataset and see basic stats.")

uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

if not uploaded_file.name.lower().endswith(".csv"):
    st.error("Invalid file format. Please upload a CSV file.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except EmptyDataError:
    st.error("The CSV file is empty.")
    st.stop()
except ParserError:
    st.error("Invalid CSV format. Please upload a valid CSV file.")
    st.stop()
except UnicodeDecodeError:
    st.error("Unable to read the file. Please upload a valid UTF-8 CSV.")
    st.stop()
except Exception as exc:
    st.error(f"Failed to read CSV: {exc}")
    st.stop()

if df.empty or df.shape[1] == 0:
    st.error("The CSV file is empty or has no columns.")
    st.stop()

st.subheader("Preview (first 10 rows)")
st.dataframe(df.head(10), width="stretch")

st.subheader("Target selection")
if df.shape[1] < 2:
    st.error("Dataset must contain at least 2 columns (features + target).")
    st.stop()

target_col = st.selectbox("Select target column", df.columns.tolist())
X = df.drop(columns=[target_col])
y = df[target_col]

st.write(f"Selected target column: {target_col}")
st.write("Feature columns:")
st.write(", ".join(X.columns.tolist()))

target_missing_ratio = float(y.isna().mean())
if target_missing_ratio > 0.5:
    st.error(
        "Target column has too many missing values. "
        "Please choose another column or clean the data."
    )
    st.stop()

agent = AutoMLAgent()
st.subheader("Agent progress")
progress_messages = []
progress_placeholder = st.empty()


def on_step(message: str) -> None:
    """Append a progress message and refresh the UI list."""
    progress_messages.append(message)
    progress_placeholder.markdown(
        "\n".join(f"- {msg}" for msg in progress_messages)
    )


with st.spinner("Running AutoML pipeline..."):
    try:
        results = agent.run(df, target_col, on_step=on_step)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"AutoML pipeline failed: {exc}")
        st.stop()

analysis = results["dataset_analysis"]
problem_type = results["problem_type"]
results_df = results["evaluation_results"]
best_model = results["best_model_name"]
model_path = results["saved_model_path"]
report_path = results["report_path"]
report_error = results.get("report_error")

st.write(f"Detected problem type: {problem_type}")

st.subheader("Dataset shape")
shape_cols = st.columns(2)
shape_cols[0].metric("Rows", analysis["rows"])
shape_cols[1].metric("Columns", analysis["columns"])

st.subheader("Column overview")
column_info_df = pd.DataFrame(
    {
        "column": analysis["column_names"],
        "dtype": [analysis["dtypes"][col] for col in analysis["column_names"]],
        "missing_values": [
            analysis["missing_values"][col] for col in analysis["column_names"]
        ],
        "unique_values": [
            analysis["unique_values"][col] for col in analysis["column_names"]
        ],
    }
)
st.dataframe(column_info_df, width="stretch")

st.subheader("Missing values chart")
missing_counts = pd.Series(analysis["missing_values"])
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
if missing_counts.empty:
    st.info("No missing values found.")
else:
    fig, ax = plt.subplots()
    ax.bar(range(len(missing_counts)), missing_counts.values)
    ax.set_xticks(range(len(missing_counts)))
    ax.set_xticklabels(missing_counts.index.astype(str), rotation=45, ha="right")
    ax.set_ylabel("Missing values")
    ax.set_title("Missing values by column")
    st.pyplot(fig)

st.subheader("Target distribution")
fig, ax = plt.subplots()
if problem_type == "classification":
    class_counts = y.dropna().astype(str).value_counts()
    if class_counts.empty:
        st.info("No target values available to plot.")
    else:
        ax.bar(range(len(class_counts)), class_counts.values)
        ax.set_xticks(range(len(class_counts)))
        ax.set_xticklabels(class_counts.index, rotation=45, ha="right")
        ax.set_ylabel("Count")
else:
    ax.hist(y.dropna(), bins=20)
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
ax.set_title("Target distribution")
if problem_type != "classification" or not class_counts.empty:
    st.pyplot(fig)

st.subheader("Preprocessing")
processed_shape = results.get("processed_shape")
if processed_shape:
    st.write(
        f"Processed features shape: {processed_shape[0]} rows, "
        f"{processed_shape[1]} features"
    )

st.subheader("Train/test split")
split_cols = st.columns(3)
split_cols[0].metric("Training rows", results.get("train_rows", 0))
split_cols[1].metric("Testing rows", results.get("test_rows", 0))
split_cols[2].metric("Final features", results.get("final_features", 0))

st.subheader("Model evaluation")
if results_df is None or results_df.empty:
    st.warning("No evaluation results available.")
else:
    st.dataframe(results_df, width="stretch")
    if best_model:
        st.write(f"Best model: {best_model}")
        if model_path:
            st.success(f"Best model saved to: {model_path}")

    metric_name = "f1" if problem_type == "classification" else "r2"
    if metric_name in results_df.columns:
        st.subheader("Model comparison")
        labels = results_df["model"].astype(str).tolist()
        values = results_df[metric_name].astype(float).tolist()

        fig, ax = plt.subplots()
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("F1 score" if metric_name == "f1" else "R2 score")
        ax.set_title("Model comparison")
        st.pyplot(fig)

st.subheader("Report")
if report_path:
    with open(report_path, "r", encoding="utf-8") as handle:
        report_content = handle.read()
    st.success(f"Report saved to: {report_path}")
    st.download_button(
        "Download report",
        data=report_content,
        file_name=os.path.basename(report_path),
        mime="text/markdown",
    )
else:
    st.error(report_error or "Report could not be generated.")

st.subheader("Numerical columns")
if analysis["numerical_columns"]:
    st.write(", ".join(analysis["numerical_columns"]))
else:
    st.write("None")

st.subheader("Categorical columns")
if analysis["categorical_columns"]:
    st.write(", ".join(analysis["categorical_columns"]))
else:
    st.write("None")

st.subheader("Numerical summary")
if analysis["numerical_summary"]:
    summary_df = pd.DataFrame(analysis["numerical_summary"]).T
    st.dataframe(summary_df, width="stretch")
else:
    st.info("No numerical columns found.")

st.subheader("Assistant")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not os.getenv("OPENAI_API_KEY"):
    st.info(
        "Set OPENAI_API_KEY to enable LLM responses. A fallback responder will be used otherwise."
    )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question about your dataset or models")
if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer_question(
                question=user_question,
                dataset_analysis=analysis,
                problem_type=problem_type,
                target_column=target_col,
                evaluation_results=results_df,
                best_model_name=best_model,
            )
        st.markdown(response)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )
