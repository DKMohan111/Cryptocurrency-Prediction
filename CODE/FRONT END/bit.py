import yfinance as yf
import pandas as pd
from datetime import datetime

# List of cryptocurrency ticker symbols
crypto_tickers = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "BNB-USD",  # Binance Coin
    "SOL-USD",  # Solana
    "XRP-USD",  # XRP
    "ADA-USD",  # Cardano
    "AVAX-USD", # Avalanche
    "DOGE-USD", # Dogecoin
    "DOT-USD",  # Polkadot
    "TON11419-USD"  # Toncoin (unofficial ticker, may need validation)
]

# Today's date
end_date = datetime.today().strftime('%Y-%m-%d')

# Download and save data for each crypto
for ticker in crypto_tickers:
    try:
        print(f"📥 Downloading {ticker} from beginning to {end_date}...")
        df = yf.download(ticker, start="2010-01-01", end=end_date, interval="1d")

        if df.empty:
            print(f"⚠️ No data found for {ticker}. Skipping...")
            continue

        # Reformat and clean
        df = df[['Close', 'High', 'Low', 'Open', 'Volume']]
        df.reset_index(inplace=True)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df[['Close', 'High', 'Low', 'Open']] = df[['Close', 'High', 'Low', 'Open']].round(2)

        # Save as TICKER.csv
        df.to_csv(f"{ticker}.csv", index=False)
        print(f"✅ Saved: {ticker}.csv")

    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")

print("\n🏁 All downloads completed.")
