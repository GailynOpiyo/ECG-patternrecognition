import pandas as pd
import json
import streamlit as st

def download_button(data, file_name, file_type="csv"):
    """
    Creates a download button for CSV, JSON, or TXT files.
    :param data: Data to be downloaded.
    :param file_name: The name of the file to be downloaded.
    :param file_type: Type of the file ("csv", "json", "txt").
    """
    if file_type == "csv":
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=file_name,
            mime="text/csv"
        )
    elif file_type == "json":
        json_data = json.dumps(data)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=file_name,
            mime="application/json"
        )
    elif file_type == "txt":
        text_data = "\n".join(data)
        st.download_button(
            label="Download TXT",
            data=text_data,
            file_name=file_name,
            mime="text/plain"
        )
