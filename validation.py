import numpy as np
import pandas as pd

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