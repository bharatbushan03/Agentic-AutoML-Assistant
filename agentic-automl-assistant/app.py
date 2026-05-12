import os
import sys

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_analyzer import analyze_dataset
from evaluator import evaluate_models
from model_trainer import train_models
from preprocessor import preprocess_data

st.set_page_config(page_title="Agentic AutoML Assistant", layout="wide")
st.title("Agentic AutoML Assistant")
st.write("Upload a CSV file to preview your dataset and see basic stats.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Failed to read CSV: {exc}")
    st.stop()

analysis = analyze_dataset(df)

st.subheader("Preview (first 10 rows)")
st.dataframe(df.head(10), use_container_width=True)

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
st.dataframe(column_info_df, use_container_width=True)

st.subheader("Target selection")
if df.shape[1] < 2:
    st.warning("Add at least 2 columns to select a target.")
    st.stop()

target_col = st.selectbox("Select target column", analysis["column_names"])
X = df.drop(columns=[target_col])
y = df[target_col]

st.write(f"Selected target column: {target_col}")
st.write("Feature columns:")
st.write(", ".join(X.columns.tolist()))

if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
    problem_type = "classification"
elif pd.api.types.is_numeric_dtype(y):
    unique_count = int(y.nunique(dropna=True))
    problem_type = "classification" if unique_count <= 20 else "regression"
else:
    problem_type = "classification"

st.write(f"Detected problem type: {problem_type}")

st.subheader("Preprocessing")
try:
    X_processed, y_processed, prep_details = preprocess_data(df, target_col)
    st.write(
        f"Processed features shape: {prep_details['processed_shape'][0]} rows, "
        f"{prep_details['processed_shape'][1]} features"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )
    st.subheader("Train/test split")
    split_cols = st.columns(3)
    split_cols[0].metric("Training rows", X_train.shape[0])
    split_cols[1].metric("Testing rows", X_test.shape[0])
    split_cols[2].metric("Final features", X_train.shape[1])

    st.subheader("Model evaluation")
    models = train_models(X_train, y_train, problem_type)
    if not models:
        st.warning("No models were trained successfully.")
    else:
        results_df, best_model = evaluate_models(
            models, X_test, y_test, problem_type
        )
        if results_df.empty:
            st.warning("No evaluation results available.")
        else:
            st.dataframe(results_df, use_container_width=True)
            if best_model:
                st.write(f"Best model: {best_model}")
except Exception as exc:
    st.error(f"Preprocessing failed: {exc}")

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
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("No numerical columns found.")
