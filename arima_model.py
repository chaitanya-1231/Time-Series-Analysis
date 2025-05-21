import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("Stock_data.csv", parse_dates=['Date'], index_col='Date')
df = df.asfreq('D')
df['y'] = df['Close'].interpolate()

p, d, q = 5, 1, 0  

arima_model = ARIMA(df['y'], order=(p, d, q))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=30)

def arima_forecast_function():
    return arima_forecast

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label="Actual Price", color='blue', linewidth=2)
plt.plot(pd.date_range(start=df.index[-1], periods=30, freq='D'), arima_forecast, 
         label="ARIMA Predictions", linestyle="dashed", color='orange', linewidth=2)

plt.xlabel("Date", fontsize=12)
plt.ylabel("Stock Price", fontsize=12)
plt.title("Stock Price Forecasting using ARIMA", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.xticks(rotation=45)

plt.show()