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
            # Missing values
            missing = data[data[column].isnull()]
            validation_results[column]["missing_count"] = missing.shape[0]
            validation_results[column]["missing_values"] = list(missing[column].unique())

            # Negative values (invalid)
            invalid = data[data[column] < 0]
            validation_results[column]["invalid_count"] = invalid.shape[0]
            validation_results[column]["invalid_values"] = list(invalid[column].unique())
            data[column] = data[column].apply(lambda x: max(x, 0))

            # Outliers (using Z-score)
            z_scores = (data[column] - data[column].mean()) / data[column].std()
            outliers = data[np.abs(z_scores) > 3]
            validation_results[column]["outlier_count"] = outliers.shape[0]
            validation_results[column]["outlier_values"] = list(outliers[column].unique())

        # Handle categorical columns
        elif data[column].dtype == 'object':
            # Missing/empty values
            missing = data[data[column].isnull() | (data[column] == "")]
            validation_results[column]["missing_count"] = missing.shape[0]
            validation_results[column]["missing_values"] = list(missing[column].unique())

            # Invalid entries
            invalid = data[data[column].str.contains(r"NULL|Not Available|Missing", na=False)]
            validation_results[column]["invalid_count"] = invalid.shape[0]
            validation_results[column]["invalid_values"] = list(invalid[column].unique())

        # Handle date columns
        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            # Missing values
            missing = data[data[column].isnull()]
            validation_results[column]["missing_count"] = missing.shape[0]
            validation_results[column]["missing_values"] = list(missing[column].unique())

            # Future dates
            invalid = data[data[column] > pd.Timestamp.now()]
            validation_results[column]["invalid_count"] = invalid.shape[0]
            validation_results[column]["invalid_values"] = list(invalid[column].unique())

    return validation_results

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
    st.write("### Validation Summary:")
    st.dataframe(summary_df, height=300)

# Data Cleaning Framework
def is_alphanumeric_id(series):
    if series.dtype == 'object':
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        match_count = non_null.astype(str).str.match(r'^[A-Za-z0-9\-_]+$', na=False).sum()
        return (match_count / len(non_null)) >= 0.6
    return False

def clean_data(data):
    # Preserve original copy
    original_data = data.copy()
    
    # Standardize column names
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]
    
    # Identify ID columns
    id_columns = [col for col in data.columns if is_alphanumeric_id(data[col].astype(str))]
    
    for column in data.columns:
        if column in id_columns:
            # Handle ID columns
            data[column] = data[column].astype(str).fillna("Unknown_ID")
        else:
            # Smart date detection
            if data[column].dtype == 'object' and any(kw in column for kw in ['date', 'time']):
                try:
                    temp_series = pd.to_datetime(data[column], errors='coerce')
                    if temp_series.notna().mean() > 0.5:
                        data[column] = temp_series
                except:
                    pass

    # Handle missing values
    for column in data.columns:
        if column in id_columns:
            continue
            
        if data[column].isnull().sum() > 0:
            if data[column].dtype in ['float64', 'int64']:
                data[column].fillna(data[column].median(), inplace=True)
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                data[column].fillna(data[column].mode()[0], inplace=True)
            elif data[column].dtype == 'object':
                data[column].fillna("Unknown", inplace=True)

    # Remove duplicates
    data = data.drop_duplicates()
    
    return data

# Enhanced Feature Engineering Framework
def feature_engineering(data):
    # Preserve original index for time-series
    original_index = data.index
    
    # Standardize column names
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]

    # Date feature engineering
    date_columns = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
    for date_col in date_columns:
        try:
            data[f"{date_col}_year"] = data[date_col].dt.year
            data[f"{date_col}_month"] = data[date_col].dt.month
            data[f"{date_col}_day"] = data[date_col].dt.day
            data[f"{date_col}_dayofweek"] = data[date_col].dt.dayofweek
            data[f"{date_col}_is_weekend"] = data[date_col].dt.dayofweek >= 5
        except Exception as e:
            st.warning(f"Date processing warning: {e}")

    # Categorical encoding (1/0)
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    for cat_col in categorical_columns:
        if 1 < data[cat_col].nunique() <= 20:
            # Create integer-encoded dummy variables
            dummies = pd.get_dummies(data[cat_col], prefix=cat_col, dtype=int)
            data = pd.concat([data.drop(columns=[cat_col]), dummies], axis=1)

    # Time-series specific features
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for num_col in numeric_columns:
        # Lag features
        if 'target' in num_col.lower():
            for lag in [1, 7, 30]:  # Daily, weekly, monthly lags
                data[f'{num_col}_lag_{lag}'] = data[num_col].shift(lag)
        
        # Rolling features
        if 'value' in num_col.lower():
            data[f'{num_col}_rolling_7d_mean'] = data[num_col].rolling(7).mean()
            data[f'{num_col}_rolling_30d_std'] = data[num_col].rolling(30).std()

    # Scale numeric features
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_columns:
        scaler = MinMaxScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Cleanup
    data = data.dropna()
    data = data.loc[:, ~data.columns.duplicated()]
    
    return data

# Streamlit App Interface
st.title("Procode Modeling Platform")

uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read data
        if uploaded_file.name.endswith('.csv'):
            raw_data = pd.read_csv(uploaded_file)
        else:
            raw_data = pd.read_excel(uploaded_file)

        # Show raw data
        st.subheader("Raw Data Preview")
        st.dataframe(raw_data.head())

        # Validation
        validation_summary = validate_columns_and_values(raw_data.copy())
        display_validation_summary(validation_summary)

        # Data Cleaning
        cleaned_data = clean_data(raw_data.copy())
        st.subheader("Cleaned Data")
        st.dataframe(cleaned_data.head())
        
        # Download cleaned data
        st.download_button(
            label="Download Cleaned Data",
            data=cleaned_data.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

        # Feature Engineering
        if not cleaned_data.empty:
            with st.spinner("Creating advanced features..."):
                feature_data = feature_engineering(cleaned_data.copy())
                
                st.subheader("Feature Engineered Data")
                st.dataframe(feature_data.head())
                
                # Download features
                st.download_button(
                    label="Download Feature-Engineered Data",
                    data=feature_data.to_csv(index=False),
                    file_name="feature_engineered_data.csv",
                    mime="text/csv"
                )
                
                # Model readiness check
                st.success(f"Data preparation complete! Final shape: {feature_data.shape}")
                st.write("✅ Suitable for time-series analysis and ML modeling")
        else:
            st.error("Cleaning resulted in empty dataset")

    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
        st.stop()