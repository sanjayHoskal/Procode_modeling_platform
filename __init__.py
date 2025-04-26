# sales_prediction_project/__init__.py
# This file can be empty, or you can add package-level initialization code.
# It can be left empty or used for package-level imports or settings.

# Example: Importing key functions directly at the package level
from .validation import validate_columns_and_values
from .cleaning import clean_data
from .feature_engineering import feature_engineering
from .forecasting import prepare_dataset, train_prophet_model, generate_forecast, auto_detect_columns
