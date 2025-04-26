import pandas as pd
import numpy as np

def clean_data(data):
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].replace(["Not Available", "NULL", "Missing", "Unknown", " "], np.nan)
            if any(kw in column for kw in ['date', 'time']):
                try:
                    temp_series = pd.to_datetime(data[column], errors='coerce')
                    if temp_series.notna().mean() > 0.5:
                        data[column] = temp_series
                except:
                    pass
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if data[column].dtype in ['float64', 'int64']:
                data[column] = pd.to_numeric(data[column], errors='coerce')
                data[column].fillna(data[column].median(), inplace=True)
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                data[column].fillna(data[column].mode()[0], inplace=True)
            elif data[column].dtype == 'object':
                data[column].fillna("Unknown", inplace=True)
    data = data.drop_duplicates()
    return data