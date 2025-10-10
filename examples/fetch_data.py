import yfinance as yf
data = yf.download("AAPL MSFT", start="2022-01-01", end="2023-01-01")['Close']
print(data.head())