import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("Stock_data.csv", parse_dates=['Date'], index_col='Date')
df = df.asfreq('D')
df['y'] = df['Close'].interpolate()


p, d, q = 2, 1, 2  
P, D, Q, m = 1, 1, 1, 12  

sarima_model = SARIMAX(df['y'], order=(p, d, q), seasonal_order=(P, D, Q, m))
sarima_fit = sarima_model.fit()
sarima_forecast = sarima_fit.forecast(steps=30)


plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label="Actual Price", color='blue')
plt.plot(pd.date_range(start=df.index[-1], periods=30, freq='D'), sarima_forecast, 
         label="SARIMA Predictions", linestyle="dashed", color='purple')

plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Forecasting using SARIMA")
plt.legend()
plt.grid()
plt.show()