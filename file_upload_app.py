import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Validation Framework
def validate_columns_and_values(data):
    validation_results = {}

    # Standardize column names
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]

    # Iterate through columns and validate
    for column in data.columns:
        validation_results[column] = {"missing_count": 0, "missing_values": [],
                                      "invalid_count": 0, "invalid_values": [],
                                      "outlier_count": 0, "outlier_values": []}

        # Handle numeric columns
        if data[column].dtype in ['float64', 'int64']:
            # Missing values
            missing = data[data[column].isnull()]
            validation_results[column]["missing_count"] = missing.shape[0]
            validation_results[column]["missing_values"] = list(missing[column].unique())

            # Negative values (invalid) - Replace with 0
            invalid = data[data[column] < 0]
            validation_results[column]["invalid_count"] = invalid.shape[0]
            validation_results[column]["invalid_values"] = list(invalid[column].unique())
            data[column] = data[column].apply(lambda x: 0 if x < 0 else x)  # Replace negative values

            # Outliers (using Z-score method)
            z_scores = (data[column] - data[column].mean()) / data[column].std()
            outliers = data[np.abs(z_scores) > 3]
            validation_results[column]["outlier_count"] = outliers.shape[0]
            validation_results[column]["outlier_values"] = list(outliers[column].unique())

        # Handle categorical columns
        elif data[column].dtype == 'object':
            # Missing or empty values
            missing = data[data[column].isnull() | (data[column] == "")]
            validation_results[column]["missing_count"] = missing.shape[0]
            validation_results[column]["missing_values"] = list(missing[column].unique())

            # Invalid entries (e.g., "NULL", "Not Available")
            invalid = data[data[column].str.contains(r"NULL|Not Available|Missing", na=False)]
            validation_results[column]["invalid_count"] = invalid.shape[0]
            validation_results[column]["invalid_values"] = list(invalid[column].unique())

        # Handle date columns
        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            # Missing values
            missing = data[data[column].isnull()]
            validation_results[column]["missing_count"] = missing.shape[0]
            validation_results[column]["missing_values"] = list(missing[column].unique())

            # Invalid dates (e.g., future dates)
            invalid = data[data[column] > pd.Timestamp.now()]
            validation_results[column]["invalid_count"] = invalid.shape[0]
            validation_results[column]["invalid_values"] = list(invalid[column].unique())

    return validation_results

# Function to display structured validation summary
def display_validation_summary(validation_summary):
    summary_data = []
    for column, details in validation_summary.items():
        summary_data.append([
            column.upper(),
            details['missing_count'], 
            ", ".join(map(str, details['missing_values'])) if details['missing_values'] else "None",
            details['invalid_count'], 
            ", ".join(map(str, details['invalid_values'])) if details['invalid_values'] else "None",
            details['outlier_count'], 
            ", ".join(map(str, details['outlier_values'])) if details['outlier_values'] else "None"
        ])

    summary_df = pd.DataFrame(summary_data, columns=[
        "Column Name", "Missing Count", "Missing Values",
        "Invalid Count", "Invalid Values", "Outlier Count", "Outlier Values"
    ])

    # Display in Streamlit
    st.write("### Validation Summary:")
    st.dataframe(summary_df, height=300)

# Data Cleaning Framework

import re

def is_alphanumeric_id(series):
    """Check if a column is likely an ID based on alphanumeric pattern"""
    if series.dtype == 'object':
        return series.str.match(r'^[A-Za-z0-9\-\_]+$', na=False).sum() > (0.8 * len(series))  # 80% rule
    return False

def clean_data(data):
    # Standardize column names
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]

    # Identify alphanumeric ID columns dynamically
    id_columns = [col for col in data.columns if is_alphanumeric_id(data[col].astype(str))]

    print(f"🔍 Detected ID Columns: {id_columns}")  # Debugging Step

    for column in data.columns:
        if column in id_columns:
            # Force IDs to remain as strings & fill missing with "Unknown_ID"
            data[column] = data[column].astype(str).fillna("Unknown_ID")
            print(f"✅ Column {column}: Sample values -> {data[column].head(5).tolist()}")  # Debugging Step
            continue

        # Handle date columns dynamically
        if data[column].dtype == 'object':
            if any(data[column].astype(str).str.contains(r"\d{4}-\d{2}-\d{2}|/|,|[-]", na=False)):
                try:
                    data[column] = pd.to_datetime(data[column], errors='coerce')
                except Exception as e:
                    print(f"⚠️ Error parsing column '{column}': {e}")

    # Handle missing values for other columns
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if data[column].dtype in ['float64', 'int64']:
                data[column].fillna(data[column].mean(), inplace=True)
            elif data[column].dtype == 'object' and column not in id_columns:
                data[column].fillna("Unknown", inplace=True)

    # Replace negative values with 0
    for column in data.columns:
        if data[column].dtype in ['float64', 'int64']:
            data[column] = data[column].apply(lambda x: 0 if x < 0 else x)

    # Remove duplicate rows
    data = data.drop_duplicates()

    return data



# Feature Engineering Framework
def feature_engineering(data):
    # Standardize column names
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]

    # Identify column types dynamically
    date_columns = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col]) or "date" in col.lower()]
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    # Handle date columns
    for date_col in date_columns:
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')  # Convert to datetime
            # Add new date-related features
            data[f"{date_col}_year"] = data[date_col].dt.year
            data[f"{date_col}_month"] = data[date_col].dt.month
            data[f"{date_col}_weekday"] = data[date_col].dt.weekday
        except Exception as e:
            print(f"Could not process date column '{date_col}': {e}")

    # Handle categorical columns
    for cat_col in categorical_columns:
        # Apply one-hot encoding for categorical variables
        if data[cat_col].nunique() < 20:  # Limit encoding for high-cardinality columns
            data = pd.get_dummies(data, columns=[cat_col], prefix=cat_col)

    # Scale numeric columns
    scaler = MinMaxScaler()
    for num_col in numeric_columns:
        data[num_col] = pd.to_numeric(data[num_col], errors='coerce')  # Convert numeric-like strings
        data[num_col].fillna(0, inplace=True)  # Handle missing values before scaling
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Drop irrelevant or empty columns
    for col in data.columns:
        if data[col].isnull().mean() > 0.5:  # Drop columns with more than 50% missing values
            data.drop(columns=[col], inplace=True)
        elif data[col].nunique() == 1:  # Drop columns with a single unique value
            data.drop(columns=[col], inplace=True)

    return data

# Streamlit App
st.title("Data Upload and Validation")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)

        st.write("### Preview of Uploaded File:")
        st.dataframe(data)

        # Perform validation
        validation_summary = validate_columns_and_values(data)
        display_validation_summary(validation_summary)

        # Cleaning
        cleaned_data = clean_data(data)
        st.write("### Cleaned Data:")
        st.dataframe(cleaned_data)

        # Feature Engineering
        feature_data = feature_engineering(cleaned_data)
        st.write("### Feature Engineered Data:")
        st.dataframe(feature_data)

        # Option to download feature-engineered data
        st.download_button(
            label="Download Feature-Engineered Data",
            data=feature_data.to_csv(index=False),
            file_name="feature_engineered_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing the file: {e}")
