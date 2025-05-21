import pandas as pd

df = pd.read_csv("Stock_Predictions.csv")

expected_columns = ["ARIMA", "SARIMA"]

missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    print(f"Error: Missing columns {missing_columns}. Available columns: {df.columns}")
else:
    df.index = pd.to_datetime(df.iloc[:, 0])
    df.drop(df.columns[0], axis=1, inplace=True)
    print("Successfully validated and formatted the dataset!")
    print(df.head())