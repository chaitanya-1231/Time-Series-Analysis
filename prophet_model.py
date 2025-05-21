from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Stock_data.csv", parse_dates=['Date'])
df = df.set_index('Date').asfreq('D')
df['y'] = df['Close'].interpolate()
df['ds'] = df.index

train = df.iloc[:-30].reset_index(drop=True)

model = Prophet(changepoint_prior_scale=0.8, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(train[['ds', 'y']])

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()

model.plot_components(forecast)
plt.show()
