import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("Stock_data.csv", parse_dates=['Date'], index_col='Date')
df = df.asfreq('D')
df['y'] = df['Close'].interpolate()

arima_model = ARIMA(df['y'], order=(7, 1, 2))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=30)

sarima_model = SARIMAX(df['y'], order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit()
sarima_forecast = sarima_fit.forecast(steps=30)

df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'y': 'y'})
prophet_model = Prophet(seasonality_mode="multiplicative")
prophet_model.fit(df_prophet)
future = prophet_model.make_future_dataframe(periods=30, freq='D')
prophet_forecast = prophet_model.predict(future)[['ds', 'yhat']][-30:]

scaling_factor = df['Close'].max() / max(arima_forecast.max(), prophet_forecast['yhat'].max(), sarima_forecast.max())

merged_forecast = pd.DataFrame({
    'Date': pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D'),
    'ARIMA': arima_forecast.values * scaling_factor,
    'Prophet': prophet_forecast['yhat'].values * scaling_factor,
    'SARIMA': sarima_forecast.values * scaling_factor
})

merged_forecast['Combined'] = merged_forecast[['ARIMA', 'Prophet', 'SARIMA']].mean(axis=1)
merged_forecast.to_csv("Stock_Predictions.csv", index=False)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label="Actual Price")
plt.plot(merged_forecast['Date'], merged_forecast['ARIMA'], label="ARIMA", linestyle="dashed")
plt.plot(merged_forecast['Date'], merged_forecast['Prophet'], label="Prophet", linestyle="dotted")
plt.plot(merged_forecast['Date'], merged_forecast['SARIMA'], label="SARIMA", linestyle="dashdot")
plt.plot(merged_forecast['Date'], merged_forecast['Combined'], label="Combined", linestyle="solid")
plt.legend()
plt.show()
