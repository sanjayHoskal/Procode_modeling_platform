import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import chardet
import io
import kaleido
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
from prophet import Prophet

from validation import validate_columns_and_values
from cleaning import clean_data
from feature_engineering import feature_engineering
from forecasting import prepare_dataset , train_prophet_model, auto_detect_columns, generate_forecast
from pdf_gen import generate_pdf_report


# ---- Streamlit App ----
st.set_page_config(page_title="Forecast Dashboard", layout="wide")
st.title("📈 Procode Modeling Platform")
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Upload", "Raw Data", "Cleaned Data", "Feature Engineered Data",
    "Forecast Table", "Forecast Chart", "YoY Growth", "Summary Dashboard",
    "Download Reports"
])

if 'ran_forecast' not in st.session_state:
    st.session_state["ran_forecast"] = False

if section == "Upload":
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            raw_bytes = uploaded_file.read()
            encoding = chardet.detect(raw_bytes)['encoding']
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, encoding=encoding)
        else:
            data = pd.read_excel(uploaded_file)

        st.session_state["raw_data"] = data.copy()

        # Basic cleaning
        cleaned_data = data.drop_duplicates().copy()
        for col in cleaned_data.select_dtypes(include='object').columns:
            cleaned_data[col] = cleaned_data[col].replace(["Not Available", "NULL", "Missing", "Unknown", " "], np.nan)
        for col in cleaned_data.columns:
            if cleaned_data[col].isnull().sum() > 0:
                if cleaned_data[col].dtype in ['float64', 'int64']:
                    cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
                elif pd.api.types.is_datetime64_any_dtype(cleaned_data[col]):
                    cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
                elif cleaned_data[col].dtype == 'object':
                    cleaned_data[col].fillna("Unknown", inplace=True)
        st.session_state["cleaned_data"] = cleaned_data.copy()

        # Feature engineering
        fe_data = cleaned_data.copy()
        date_cols = [col for col in fe_data.columns if pd.api.types.is_datetime64_any_dtype(fe_data[col]) or 'date' in col.lower()]
        for col in date_cols:
            fe_data[col] = pd.to_datetime(fe_data[col], errors='coerce')
            fe_data[f"{col}_year"] = fe_data[col].dt.year
            fe_data[f"{col}_month"] = fe_data[col].dt.month
            fe_data[f"{col}_day"] = fe_data[col].dt.day
            fe_data[f"{col}_dayofweek"] = fe_data[col].dt.dayofweek
        st.session_state["fe_data"] = fe_data.copy()

        st.success("Data uploaded and processed. Please proceed to individual sections for detailed view or run forecast.")

        st.subheader("📄 Raw Data Preview")
        st.dataframe(data.head())

        st.subheader("🧼 Cleaned Data Preview")
        st.dataframe(cleaned_data.head())

        st.subheader("🛠️ Feature Engineered Data Preview")
        st.dataframe(fe_data.head())

        # Auto detect ds and y columns
        auto_date_col, auto_target_col = auto_detect_columns(fe_data)

        st.write("### Select Columns for Forecasting")
        date_col = st.selectbox("Select Date Column (ds)", fe_data.columns, index=fe_data.columns.get_loc(auto_date_col))
        target_col = st.selectbox("Select Target Column (y)", fe_data.select_dtypes(include=np.number).columns, index=list(fe_data.select_dtypes(include=np.number).columns).index(auto_target_col))

        if st.button("Run Forecast"):
            forecast_data = prepare_dataset(fe_data, date_col, target_col)
            model = train_prophet_model(forecast_data)
            forecast = generate_forecast(model)

            st.session_state["forecast"] = forecast
            st.session_state["model"] = model
            st.session_state["ran_forecast"] = True
            st.session_state["date_col"] = date_col
            st.session_state["target_col"] = target_col

            st.subheader("Forecast Table")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

            st.subheader("Forecast Chart")
            fig = px.line(forecast, x="ds", y="yhat", title=f"Forecasted Values for {target_col}")
            fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode='lines', name='Upper Bound')
            fig.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode='lines', name='Lower Bound')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Year-over-Year Growth")
            forecast["year"] = forecast["ds"].dt.year
            yearly = forecast.groupby("year")["yhat"].sum().reset_index()
            yearly["YoY Growth %"] = yearly["yhat"].pct_change() * 100
            fig2 = px.bar(yearly, x="year", y="YoY Growth %", title=f"YoY Forecast Growth % for {target_col}")
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            st.info("✅ Forecast complete! Please visit the navigation panel for individual graph analysis.")

if section == "Raw Data" and "raw_data" in st.session_state:
    st.subheader("📄 Raw Data")
    st.dataframe(st.session_state["raw_data"].head())

if section == "Cleaned Data" and "cleaned_data" in st.session_state:
    st.subheader("🧼 Cleaned Data")
    st.dataframe(st.session_state["cleaned_data"].head())

if section == "Feature Engineered Data" and "fe_data" in st.session_state:
    st.subheader("🛠️ Feature Engineered Data")
    st.write("Feature engineered data includes additional columns like year, month, day, and day of week extracted from date columns to help improve forecasting accuracy. These features help the model learn seasonal and periodic patterns that occur across time.")
    st.dataframe(st.session_state["fe_data"].head())

if section in ["Forecast Table", "Forecast Chart", "YoY Growth", "Summary Dashboard", "Download Reports"] and not st.session_state.get("ran_forecast"):
    st.warning("Please upload data and run a forecast first.")

if section == "Forecast Table" and st.session_state.get("ran_forecast"):
    forecast = st.session_state["forecast"]
    st.subheader("📋 Forecast Table")
    st.write("This table provides numerical forecast results.\n\nEach row contains the predicted value (\"yhat\") for a specific date, along with the lower and upper bounds of the confidence interval (\"yhat_lower\" and \"yhat_upper\"). This helps you understand both the expected values and potential range of fluctuation.")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

if section == "Forecast Chart" and st.session_state.get("ran_forecast"):
    forecast = st.session_state["forecast"]
    st.subheader("📈 Forecast Visualization")
    st.write("Visualize forecasted values over time with options to show trend lines, confidence intervals, and time-based aggregation.")

    show_trend = st.checkbox("Show Trend Line", True)
    show_bounds = st.checkbox("Show Confidence Interval", True)
    aggregate_option = st.selectbox("Aggregate Forecast", ["None", "Monthly", "Weekly", "Quarterly"], index=0)

    fig = go.Figure()

    if aggregate_option != "None":
        forecast_copy = forecast.copy()

        if aggregate_option == "Monthly":
            forecast_copy["period"] = forecast_copy["ds"].dt.to_period("M").astype(str)
        elif aggregate_option == "Weekly":
            forecast_copy["period"] = forecast_copy["ds"].dt.to_period("W").astype(str)
        elif aggregate_option == "Quarterly":
            forecast_copy["period"] = forecast_copy["ds"].dt.to_period("Q").astype(str)

        agg_df = forecast_copy.groupby("period")["yhat"].sum().reset_index()
        fig.add_trace(go.Scatter(x=agg_df["period"], y=agg_df["yhat"], mode='lines+markers', name=f'{aggregate_option} Forecast'))
        fig.update_layout(title=f"{aggregate_option} Aggregated Forecast", xaxis_title=aggregate_option, yaxis_title=st.session_state['target_col'])

    else:
        if show_trend:
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode='lines', name='Forecast'))
        if show_bounds:
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode='lines', name='Upper Bound', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode='lines', name='Lower Bound', line=dict(dash='dot')))
        fig.update_layout(title="Forecast with Trend and Confidence Intervals", xaxis_title="Date", yaxis_title=st.session_state['target_col'])

    st.plotly_chart(fig, use_container_width=True)

if section == "YoY Growth" and st.session_state.get("ran_forecast"):
    forecast = st.session_state["forecast"]
    forecast["year"] = forecast["ds"].dt.year
    yearly = forecast.groupby("year")["yhat"].sum().reset_index()
    yearly["YoY Growth %"] = yearly["yhat"].pct_change() * 100

    st.subheader("📊 Year-over-Year Growth Analysis")
    st.write("This section helps you understand how the forecasted values change from one year to the next.\n\nA positive YoY (Year-over-Year) growth percentage suggests increasing performance, while a negative value indicates decline. This can help in strategic planning and goal setting.")

    show_yoy = st.checkbox("Show YoY % Growth Chart", True)
    show_total = st.checkbox("Show Total Forecast Chart", False)

    col1, col2 = st.columns(2)

    if show_yoy:
        with col1:
            fig1 = px.bar(yearly, x="year", y="YoY Growth %", title="Year-over-Year Growth %")
            st.plotly_chart(fig1, use_container_width=True)

    if show_total:
        with col2:
            fig2 = px.line(yearly, x="year", y="yhat", title="Total Forecast by Year")
            st.plotly_chart(fig2, use_container_width=True)

if section == "Summary Dashboard" and st.session_state.get("ran_forecast"):
    st.subheader("📊 Summary Dashboard")
    st.write("This dashboard summarizes the key outcomes from the forecast.\n\nYou can quickly view the latest forecast value, total forecasted revenue/sales, and the average YoY growth across years to gain high-level insights.")
    forecast = st.session_state["forecast"]
    latest_forecast = forecast["yhat"].iloc[-1]
    total_forecast = forecast["yhat"].sum()
    yearly = forecast.copy()
    yearly["year"] = yearly["ds"].dt.year
    yoy_df = yearly.groupby("year")["yhat"].sum().pct_change().reset_index()
    avg_yoy = yoy_df["yhat"].mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Forecast Value", f"{latest_forecast:,.2f}")
    col2.metric("Total Forecast Value", f"{total_forecast:,.2f}")
    col3.metric("Average YoY Growth", f"{avg_yoy:.2f}%")

if section == "Download Reports" and st.session_state.get("ran_forecast"):
    st.subheader("📥 Download Reports")
    st.write("""
    This section allows you to download important results for offline analysis or presentation:

    - **Cleaned Data:** Raw uploaded data after duplicates, missing values, and invalid entries are corrected.
    - **Feature Engineered Data:** Includes additional features like year, month, day, and day-of-week extracted from date columns.
    - **Forecast Results:** Final prediction output with date-wise forecasts and confidence bounds.
    """)

    # Download Cleaned Data
    cleaned_csv = st.session_state["cleaned_data"].to_csv(index=False)
    st.download_button(
        label="📄 Download Cleaned Data",
        data=cleaned_csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    # Download Feature Engineered Data
    fe_csv = st.session_state["fe_data"].to_csv(index=False)
    st.download_button(
        label="📄 Download Feature Engineered Data",
        data=fe_csv,
        file_name="feature_engineered_data.csv",
        mime="text/csv"
    )

        # Download Forecast Results
    forecast_csv = st.session_state["forecast"].to_csv(index=False)
    st.download_button(
        label="📄 Download Forecast Results (CSV)",
        data=forecast_csv,
        file_name="forecast_results.csv",
        mime="text/csv"
    )

    # Generate Metrics for Summary Section
    forecast = st.session_state["forecast"]
    metrics = {
        "total_forecast": forecast["yhat"].sum(),
        "latest_forecast": forecast["yhat"].iloc[-1],
        "average_yoy": (forecast.groupby(forecast["ds"].dt.year)["yhat"].sum().pct_change().mean()) * 100
    }

    # Create Forecast Chart
    fig_forecast = px.line(forecast, x="ds", y="yhat", title=f"Forecast for {st.session_state['target_col']}")
    fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode='lines', name='Upper Bound')
    fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode='lines', name='Lower Bound')

    # Create YoY Growth Chart
    forecast["year"] = forecast["ds"].dt.year
    yearly = forecast.groupby("year")["yhat"].sum().reset_index()
    yearly["YoY Growth %"] = yearly["yhat"].pct_change() * 100
    yoy_fig = px.bar(yearly, x="year", y="YoY Growth %", title="Year-over-Year Growth")

    # Generate PDF Report
    pdf_buffer = generate_pdf_report(
    forecast=forecast,
    metrics=metrics,
    # forecast_image_path=forecast_image_path,
    # yoy_image_path=yoy_image_path
)

    # Download PDF Report
    st.download_button(
        label="📄 Download Comprehensive PDF Report",
        data=pdf_buffer,
        file_name="forecast_report.pdf",
        mime="application/pdf"
    )