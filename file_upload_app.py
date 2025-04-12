import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import matplotlib.pyplot as plt
import chardet

# Validation Framework
def validate_columns_and_values(data):
    validation_results = {}
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]

    for column in data.columns:
        validation_results[column] = {
            "missing_count": 0,
            "missing_values": [],
            "invalid_count": 0,
            "invalid_values": [],
            "outlier_count": 0,
            "outlier_values": []
        }

        if data[column].dtype in ['float64', 'int64']:
            missing = data[data[column].isnull()]
            validation_results[column]["missing_count"] = missing.shape[0]
            validation_results[column]["missing_values"] = list(missing[column].unique())
            invalid = data[data[column] < 0]
            validation_results[column]["invalid_count"] = invalid.shape[0]
            validation_results[column]["invalid_values"] = list(invalid[column].unique())
            data[column] = data[column].apply(lambda x: max(x, 0))
            z_scores = (data[column] - data[column].mean()) / data[column].std()
            outliers = data[np.abs(z_scores) > 3]
            validation_results[column]["outlier_count"] = outliers.shape[0]
            validation_results[column]["outlier_values"] = list(outliers[column].unique())

        elif data[column].dtype == 'object':
            missing = data[data[column].isnull() | (data[column] == "")]
            validation_results[column]["missing_count"] = missing.shape[0]
            validation_results[column]["missing_values"] = list(missing[column].unique())
            invalid = data[data[column].str.contains(r"NULL|Not Available|Missing", na=False)]
            validation_results[column]["invalid_count"] = invalid.shape[0]
            validation_results[column]["invalid_values"] = list(invalid[column].unique())

        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            missing = data[data[column].isnull()]
            validation_results[column]["missing_count"] = missing.shape[0]
            validation_results[column]["missing_values"] = list(missing[column].unique())
            invalid = data[data[column] > pd.Timestamp.now()]
            validation_results[column]["invalid_count"] = invalid.shape[0]
            validation_results[column]["invalid_values"] = list(invalid[column].unique())

    return validation_results

# Data Cleaning Framework
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

# Feature Engineering Framework
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

# Prophet Functions
def prepare_dataset(data, date_column, value_column):
    data = data.rename(columns={date_column: "ds", value_column: "y"})
    data["ds"] = pd.to_datetime(data["ds"], errors='coerce')
    return data.dropna(subset=["ds", "y"])

def train_prophet_model(data):
    model = Prophet()
    model.fit(data)
    return model

def generate_forecast(model, periods=365):
    future = model.make_future_dataframe(periods=periods)
    return model.predict(future)

# Streamlit App
st.set_page_config(layout="wide")
st.title("🔮 Procode Forecasting Platform")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            raw_bytes = uploaded_file.read()
            encoding = chardet.detect(raw_bytes)['encoding']
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, encoding=encoding)
        else:
            data = pd.read_excel(uploaded_file)

        st.subheader("📊 Raw Data Preview")
        st.dataframe(data.head())

        # Column cleanup UI
        col_action = st.radio("Choose column operation:", ["Keep specific columns", "Drop specific columns"])
        if col_action == "Keep specific columns":
            cols = st.multiselect("Select columns to keep", data.columns.tolist(), default=data.columns.tolist())
            data = data[cols]
        else:
            drops = st.multiselect("Select columns to drop", data.columns.tolist())
            data = data.drop(columns=drops)

        # Clean data
        data = clean_data(data)
        st.subheader("🧹 Cleaned Data")
        st.dataframe(data.head())

        # Detect columns
        auto_ds = next((col for col in data.columns if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(data[col])), None)
        auto_y = next((col for col in data.columns if col.lower() in ['sales', 'revenue', 'profit'] or np.issubdtype(data[col].dtype, np.number)), None)

        st.markdown("### 📌 Column Mapping (Edit if needed)")
        ds_column = st.selectbox("Select Date Column (`ds`)", data.columns.tolist(), index=data.columns.get_loc(auto_ds) if auto_ds else 0)
        y_column = st.selectbox("Select Target Column (`y`)", data.columns.tolist(), index=data.columns.get_loc(auto_y) if auto_y else 0)

        # Feature engineering
        data = feature_engineering(data)

        if st.button("🚀 Run Forecast"):
            dataset = prepare_dataset(data[[ds_column, y_column]].copy(), ds_column, y_column)
            if dataset.empty:
                st.error("Dataset is empty after cleaning or missing required columns.")
            else:
                model = train_prophet_model(dataset)
                forecast = generate_forecast(model)

                st.subheader("📈 Forecast Table")
                st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

                st.subheader("📊 Forecast Graph")
                st.pyplot(model.plot(forecast))

                st.subheader("🧩 Forecast Components")
                st.pyplot(model.plot_components(forecast))

                st.download_button(
                    "📥 Download Forecast CSV",
                    data=forecast.to_csv(index=False),
                    file_name="forecast_output.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error while processing file: {e}")
