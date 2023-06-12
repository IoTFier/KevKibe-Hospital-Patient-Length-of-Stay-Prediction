import streamlit as st
import pandas as pd
import requests
import io

st.title("Hospital length of stay Prediction")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xls", "xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.write("Original DataFrame:")
        st.dataframe(df)

        response = requests.post('http://localhost:8000/predict', json=df.to_dict())

        predictions = response.json()["predictions"]

        df["Prediction"] = predictions

        st.write("Modified DataFrame:")
        st.dataframe(df)

        modified_file = io.BytesIO()
        with pd.ExcelWriter(modified_file, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")

        modified_file.seek(0)
        st.download_button("Download Modified File", data=modified_file, file_name="prediction_file.xlsx")
    except Exception as e:
        st.error(f"Error: {e}")

