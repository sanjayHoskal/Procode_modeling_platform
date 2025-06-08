from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import pandas as pd

def train_prophet_model(data, seasonality_mode='additive', changepoint_prior_scale=0.05, periods=30, return_model=False):
    df = data.copy()
    df = df.rename(columns={"ds": "ds", "y": "y"})
    model = Prophet(seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior_scale)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    if return_model:
        return (forecast, model)
    else:
        return forecast

def train_arima_model(data, order=(1,1,1), forecast_periods=365, return_model=False):
    df = data.copy()
    df = df.groupby('ds', as_index=False).agg({'y': 'sum'})
    df = df.set_index('ds').asfreq('D')
    df['y'].interpolate(inplace=True)
    model = ARIMA(df['y'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_periods)
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
    forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast.values})
    forecast_df['yhat_lower'] = forecast_df['yhat'] * 0.95
    forecast_df['yhat_upper'] = forecast_df['yhat'] * 1.05
    if return_model:
        return (forecast_df, model_fit)
    else:
        return forecast_df

def train_sarima_model(data, order=(1,1,1), seasonal_order=(1,1,1,12), forecast_periods=365, return_model=False):
    df = data.copy()
    df = df.groupby('ds', as_index=False).agg({'y': 'sum'})
    df = df.set_index('ds').asfreq('D')
    df['y'].interpolate(inplace=True)
    model = SARIMAX(df['y'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=forecast_periods)
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
    forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast.values})
    forecast_df['yhat_lower'] = forecast_df['yhat'] * 0.95
    forecast_df['yhat_upper'] = forecast_df['yhat'] * 1.05
    if return_model:
        return (forecast_df, model_fit)
    else:
        return forecast_df

def train_xgb_model(data, forecast_periods=365, return_model=False):
    df = data.copy()
    df = df.sort_values("ds")
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['y_lag1'] = df['y'].shift(1)
    df = df.dropna()
    X = df[['year', 'month', 'day', 'dayofweek', 'y_lag1']]
    y = df['y']

    model = xgb.XGBRegressor()
    model.fit(X, y)

    last_row = df.iloc[-1].copy()
    future_rows = []
    for i in range(forecast_periods):
        next_date = last_row['ds'] + pd.Timedelta(days=1)
        features = {
            'year': next_date.year,
            'month': next_date.month,
            'day': next_date.day,
            'dayofweek': next_date.dayofweek,
            'y_lag1': last_row['y']
        }
        X_pred = pd.DataFrame([features])
        y_pred = model.predict(X_pred)[0]
        future_rows.append({'ds': next_date, 'yhat': y_pred})
        last_row = pd.Series({'ds': next_date, 'y': y_pred, **features})

    forecast_df = pd.DataFrame(future_rows)
    forecast_df['yhat_lower'] = forecast_df['yhat'] * 0.95
    forecast_df['yhat_upper'] = forecast_df['yhat'] * 1.05
    if return_model:
        return (forecast_df, model)
    else:
        return forecast_df

def prepare_dataset(df, date_col, target_col):
    df = df[[date_col, target_col]].copy()
    df = df.rename(columns={date_col: "ds", target_col: "y"})
    df = df.sort_values("ds")
    df = df.dropna()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df[["ds", "y"]]
    return df

def get_train_test_split(data, test_size=30):
    train = data.iloc[:-test_size]
    test = data.iloc[-test_size:]
    return train, test

def auto_detect_columns(df):
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower()]
    num_cols = df.select_dtypes(include='number').columns
    date_col = date_cols[0] if date_cols else df.columns[0]
    target_col = num_cols[0] if len(num_cols) else df.columns[-1]
    return date_col, target_col
