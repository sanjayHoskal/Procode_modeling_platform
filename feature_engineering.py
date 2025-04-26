import pandas as pd
import streamlit as st

def feature_engineering(data):
    preserved_columns = ['ds', 'y']
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]
    date_columns = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]

    for date_col in date_columns:
        try:
            data[f"{date_col}_year"] = data[date_col].dt.year
            data[f"{date_col}_month"] = data[date_col].dt.month
            data[f"{date_col}_day"] = data[date_col].dt.day
            data[f"{date_col}_dayofweek"] = data[date_col].dt.dayofweek
        except Exception as e:
            st.warning(f"Date processing warning in column '{date_col}': {e}")

    return data
