import yfinance as yf

# Define the ticker symbol for Bitcoin
ticker = "BTC-USD"

# Define the start and end dates
start_date = "2015-01-01"
end_date = "2025-03-19"

# Fetch historical OHLC data
df = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Reorder columns to match the requested format
df = df[['Close', 'High', 'Low', 'Open', 'Volume']]

# Reset index to move Date from index to a column
df.reset_index(inplace=True)

# Rename columns for consistency
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Round numerical values to 2 decimal places
df[['Close', 'High', 'Low', 'Open']] = df[['Close', 'High', 'Low', 'Open']].round(2)

# Save data to a CSV file
df.to_csv("BTC-USD.csv", index=False)

# Display the first few rows
print(df.head())
