import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('Stock_data.csv', parse_dates=['Date'], index_col='Date')
print("Data Head:\n", data.head())


data['Close'].plot(figsize=(12, 6), title='Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid()
plt.show()


data['MA_30'] = data['Close'].rolling(window=30).mean()
data[['Close', 'MA_30']].plot(figsize=(12, 6), title='30-Day Moving Average')
plt.show()
