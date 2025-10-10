import yfinance as yf

data = yf.download("GOOG MSFT", start="2025-01-01", end="2025-10-01")['Close']
print(data.head())