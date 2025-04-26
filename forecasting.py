from prophet import Prophet
import pandas as pd
import numpy as np

def prepare_dataset(data, date_column, value_column):
    df = data.rename(columns={date_column: "ds", value_column: "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors='coerce')
    df = df.dropna(subset=["ds", "y"])
    return df

def train_prophet_model(data):
    model = Prophet()
    model.fit(data)
    return model

def generate_forecast(model, periods=365):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def auto_detect_columns(data):
    date_col = next((col for col in data.columns if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(data[col])), data.columns[0])
    num_col = next((col for col in data.select_dtypes(include=np.number).columns if col.lower() in ['sales', 'revenue', 'profit']), data.select_dtypes(include=np.number).columns[0])
    return date_col, num_col
