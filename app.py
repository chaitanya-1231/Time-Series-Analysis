import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO

import plotly.express as px
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

st.title("📈 Stock Forecast Dashboard")

df = pd.read_csv("Stock_Predictions.csv")
if len(df.columns) >= 3:
    df.columns = ["Date"] + list(df.columns[1:])
else:
    st.error(f"Unexpected column count in CSV. Found {len(df.columns)} columns.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

all_models = df.columns.tolist()
selected_models = st.sidebar.multiselect("Choose Models to Display", all_models, default=all_models)

if selected_models:
    fig = px.line(df, x=df.index, y=selected_models, title="Model Forecasts")
    st.plotly_chart(fig)
else:
    st.warning("Please select at least one model to display.")

with st.expander("📄 View Forecast Data"):
    st.dataframe(df)

try:
    eval_df = pd.read_csv("model_evaluation.csv")
    st.subheader("📊 Model Performance (RMSE & MAE)")
    st.dataframe(eval_df)
except:
    st.warning("Evaluation file not found or improperly formatted.")

@st.cache_data
def load_actual_data():
    date_rng = pd.date_range(end=datetime.today(), periods=180, freq='D')
    data = pd.DataFrame({
        'ds': date_rng,
        'y': np.random.randn(len(date_rng)).cumsum() + 100
    })
    data.set_index('ds', inplace=True)
    return data

data = load_actual_data()

st.subheader("📆 Actual Data (Last 30 Days)")
min_date = data.index.min().date()
max_date = data.index.max().date()
start_date, end_date = st.date_input("Select actual data range", [max_date - timedelta(days=30), max_date], min_value=min_date, max_value=max_date)
filtered_data = data.loc[start_date:end_date]
st.line_chart(filtered_data)

n_days = st.slider("Forecast horizon (days)", min_value=7, max_value=60, value=30)

def forecast_arima(data, n_days):
    model = ARIMA(data, order=(5,1,0)).fit()
    forecast = model.forecast(steps=n_days)
    return forecast

def forecast_sarima(data, n_days):
    model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
    forecast = model.forecast(steps=n_days)
    return forecast

def forecast_prophet(data, n_days):
    df = data.reset_index().rename(columns={"ds": "ds", "y": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=n_days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].set_index('ds').iloc[-n_days:]['yhat']

def combine_forecasts(f1, f2, f3):
    return (f1 + f2 + f3) / 3

st.subheader("🔮 Forecasts")

try:
    arima_forecast = forecast_arima(data['y'], n_days)
    sarima_forecast = forecast_sarima(data['y'], n_days)
    prophet_forecast = forecast_prophet(data, n_days)
    combined_forecast = combine_forecasts(arima_forecast, sarima_forecast, prophet_forecast)

    forecast_df = pd.DataFrame({
        'ARIMA': arima_forecast.values,
        'SARIMA': sarima_forecast.values,
        'Prophet': prophet_forecast.values,
        'Combined': combined_forecast.values
    }, index=pd.date_range(start=data.index[-1] + timedelta(days=1), periods=n_days))

    st.line_chart(forecast_df)

    st.subheader("📊 Evaluation Table (RMSE)")

    actual = data['y'][-n_days:]

    def evaluate(model_forecast):
        aligned_actual = actual[:len(model_forecast)]
        return np.sqrt(mean_squared_error(aligned_actual, model_forecast[:len(aligned_actual)]))

    eval_data = {
        'Model': ['ARIMA', 'SARIMA', 'Prophet', 'Combined'],
        'RMSE': [
            evaluate(arima_forecast),
            evaluate(sarima_forecast),
            evaluate(prophet_forecast),
            evaluate(combined_forecast)
        ]
    }

    eval_df = pd.DataFrame(eval_data)
    st.dataframe(eval_df)

    st.subheader("⬇️ Download Forecast")

    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=True, sheet_name='Forecast')
        processed_data = output.getvalue()
        return processed_data

    excel_data = to_excel(forecast_df)

    st.download_button(
        label="Download forecast as Excel",
        data=excel_data,
        file_name="forecast_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception as e:
    st.error(f"An error occurred during forecasting: {e}")
