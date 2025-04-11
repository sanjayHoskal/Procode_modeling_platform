import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def prepare_dataset(data, date_column, value_column):
    """Prepare the dataset for Prophet modeling."""
    # Rename columns for Prophet compatibility
    prophet_data = data.rename(columns={date_column: "ds", value_column: "y"})
    # Ensure date column is in datetime format
    prophet_data["ds"] = pd.to_datetime(prophet_data["ds"], errors='coerce')
    # Drop rows with missing or invalid values
    prophet_data = prophet_data.dropna(subset=["ds", "y"])
    return prophet_data

def train_prophet_model(data):
    """Train the Prophet model."""
    # Initialize the Prophet model
    model = Prophet()
    # Fit the model with the prepared data
    model.fit(data)
    return model

def generate_forecast(model, periods=365):
    """Generate future sales forecast."""
    # Create a future dataframe for prediction
    future = model.make_future_dataframe(periods=periods)
    # Generate predictions
    forecast = model.predict(future)
    return forecast

def visualize_forecast(model, forecast):
    """Visualize the forecast."""
    fig = model.plot(forecast)
    plt.title("Sales Forecast")
    plt.show()

def visualize_components(model, forecast):
    """Visualize forecast components (trends, seasonality)."""
    fig = model.plot_components(forecast)
    plt.show()
