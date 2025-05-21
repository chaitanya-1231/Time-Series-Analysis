import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import numpy as np

df = pd.read_csv("Stock_data.csv", parse_dates=['Date'], index_col='Date')
df = df.asfreq('D')
df['y'] = df['Close'].interpolate()
df['ds'] = df.index

train = df.iloc[:-30].copy()
test = df.iloc[-30:].copy()

arima_model = ARIMA(train['y'], order=(5, 1, 0))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=30)

sarima_model = SARIMAX(train['y'], order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit()
sarima_forecast = sarima_fit.forecast(steps=30)

prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(train[['ds', 'y']])
future = prophet_model.make_future_dataframe(periods=30)
prophet_forecast = prophet_model.predict(future)
prophet_forecast_30 = prophet_forecast.set_index('ds').loc[test.index, 'yhat']

comparison_df = pd.DataFrame({
    "Date": test.index,
    "Actual": test['y'].values,
    "ARIMA": arima_forecast.values,
    "SARIMA": sarima_forecast.values,
    "PROPHET": prophet_forecast_30.values
})
comparison_df.set_index("Date", inplace=True)

def evaluate_model(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae

arima_rmse, arima_mae = evaluate_model(comparison_df["Actual"], comparison_df["ARIMA"])
sarima_rmse, sarima_mae = evaluate_model(comparison_df["Actual"], comparison_df["SARIMA"])
prophet_rmse, prophet_mae = evaluate_model(comparison_df["Actual"], comparison_df["PROPHET"])

metrics_df = pd.DataFrame({
    "Model": ["ARIMA", "SARIMA", "PROPHET"],
    "RMSE": [arima_rmse, sarima_rmse, prophet_rmse],
    "MAE": [arima_mae, sarima_mae, prophet_mae]
})

comparison_df.to_csv("model_comparison_forecast.csv")
metrics_df.to_csv("model_evaluation.csv", index=False)

plt.figure(figsize=(12, 6))
plt.plot(comparison_df.index, comparison_df["Actual"], label="Actual", color="blue", linewidth=2)
plt.plot(comparison_df.index, comparison_df["ARIMA"], label="ARIMA", linestyle="dashed", color="orange")
plt.plot(comparison_df.index, comparison_df["SARIMA"], label="SARIMA", linestyle="dotted", color="purple")
plt.plot(comparison_df.index, comparison_df["PROPHET"], label="Prophet", linestyle="dashdot", color="green")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Model Forecasts vs. Actual Prices")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
