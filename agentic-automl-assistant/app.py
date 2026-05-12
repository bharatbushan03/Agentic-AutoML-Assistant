import pandas as pd
import streamlit as st

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

st.subheader("Preview (first 10 rows)")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset shape")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

st.subheader("Columns and data types")
dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
st.dataframe(dtypes_df, use_container_width=True)

st.subheader("Missing values by column")
missing_df = pd.DataFrame(
    {"column": df.columns, "missing_values": df.isna().sum().values}
)
st.dataframe(missing_df, use_container_width=True)
