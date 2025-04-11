import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import matplotlib.pyplot as plt

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

        # Handle numeric columns
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
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]
    date_columns = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
    for date_col in date_columns:
        try:
            data[f"{date_col}_year"] = data[date_col].dt.year
            data[f"{date_col}_month"] = data[date_col].dt.month
            data[f"{date_col}_day"] = data[date_col].dt.day
            data[f"{date_col}_dayofweek"] = data[date_col].dt.dayofweek
        except Exception as e:
            st.warning(f"Date processing warning: {e}")
    return data

# Sales Prediction Framework
# Function for Prophet Sales Prediction
def prepare_dataset(data, date_column, value_column):
    prophet_data = data.rename(columns={date_column: "ds", value_column: "y"})
    prophet_data["ds"] = pd.to_datetime(prophet_data["ds"], errors='coerce')
    prophet_data = prophet_data.dropna(subset=["ds", "y"])
    return prophet_data

def train_prophet_model(data):
    model = Prophet()
    model.fit(data)
    return model

def generate_forecast(model, periods=365):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Streamlit App Interface
# Streamlit App Interface
st.title("Procode Modeling Platform")

# Upload Options: Automated or Manual Feature-Engineered Data Upload
upload_option = st.radio(
    "Choose Upload Option:",
    ("Automated Feature Engineering", "Manual Feature Engineered File")
)

if upload_option == "Automated Feature Engineering":
    uploaded_file = st.file_uploader("Upload Raw Dataset (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                raw_data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                raw_data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                st.stop()

            st.subheader("Raw Data Preview")
            st.dataframe(raw_data.head())

            # Column Workflow Selection
            column_workflow = st.radio(
                "Choose how you want to modify columns:",
                ("Delete Unnecessary Columns", "Select Necessary Columns")
            )

            if column_workflow == "Delete Unnecessary Columns":
                columns_to_delete = st.multiselect(
                    "Select columns you want to delete:",
                    options=raw_data.columns.tolist()
                )
                if columns_to_delete:
                    st.write(f"Deleting columns: {columns_to_delete}")
                    raw_data = raw_data.drop(columns=columns_to_delete)
            elif column_workflow == "Select Necessary Columns":
                columns_to_keep = st.multiselect(
                    "Select columns you want to keep:",
                    options=raw_data.columns.tolist()
                )
                if columns_to_keep:
                    st.write(f"Keeping only columns: {columns_to_keep}")
                    raw_data = raw_data[columns_to_keep]

            st.write("### Updated Data After Column Modification")
            st.dataframe(raw_data.head())

            if st.button("Save and Continue"):
                cleaned_data = clean_data(raw_data.copy())
                if cleaned_data.empty:
                    st.error("Cleaned data is empty after processing. Please check your dataset or column selection.")
                else:
                    st.subheader("Cleaned Data Preview")
                    st.dataframe(cleaned_data.head())
                    feature_data = feature_engineering(cleaned_data.copy())
                    st.subheader("Feature Engineered Data")
                    st.dataframe(feature_data.head())

                    sales_data = prepare_dataset(cleaned_data, "order_date", "amount_paid")
                    model = train_prophet_model(sales_data)
                    forecast = generate_forecast(model)

                    st.subheader("Sales Forecast")
                    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
                    st.subheader("Forecast Visualization")
                    fig = model.plot(forecast)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Processing Error: {e}")

else:
    manual_file = st.file_uploader("Upload Manually Feature Engineered File", type=["csv", "xlsx"])
    if manual_file is not None:
        try:
            # Read the uploaded file
            if manual_file.name.endswith('.csv'):
                manual_data = pd.read_csv(manual_file)
            elif manual_file.name.endswith('.xlsx'):
                manual_data = pd.read_excel(manual_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                st.stop()

            st.subheader("Manually Feature Engineered Data Preview")
            st.dataframe(manual_data.head())

            # Validate required columns for Prophet
            required_columns = ["ds", "y"]
            missing_columns = [col for col in required_columns if col not in manual_data.columns]

            if missing_columns:
                st.warning(f"The uploaded file is missing required columns: {', '.join(missing_columns)}.")
                
                # Allow users to dynamically select columns for 'ds' and 'y'
                st.write("Please select columns for 'ds' (date) and 'y' (value):")
                ds_column = st.selectbox("Select the column for dates (ds):", manual_data.columns.tolist())
                y_column = st.selectbox("Select the column for values (y):", manual_data.columns.tolist())

                if ds_column and y_column:
                    manual_data = manual_data.rename(columns={ds_column: "ds", y_column: "y"})
                    st.success(f"Columns mapped: 'ds' -> {ds_column}, 'y' -> {y_column}")

            # Convert 'ds' to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(manual_data['ds']):
                manual_data['ds'] = pd.to_datetime(manual_data['ds'], errors='coerce')

            # Check for missing values
            if manual_data[['ds', 'y']].isnull().any().any():
                st.error("The dataset contains missing values in 'ds' or 'y' columns. Please clean your data and try again.")
            else:
                # Proceed with forecasting
                sales_data = manual_data[['ds', 'y']]
                model = train_prophet_model(sales_data)
                forecast = generate_forecast(model)

                st.subheader("Sales Forecast")
                st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
                st.subheader("Forecast Visualization")
                fig = model.plot(forecast)
                st.pyplot(fig)

                # Allow downloading forecasted data
                st.download_button(
                    label="Download Forecasted Data",
                    data=forecast.to_csv(index=False),
                    file_name="forecasted_sales.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Processing Error: {e}")
