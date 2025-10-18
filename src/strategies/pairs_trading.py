import yfinance as yf
import matplotlib.pyplot as plt

##############################
# Fetch historical closing prices for Google (GOOG) and Microsoft (MSFT)
data = yf.download(['GOOGL', 'GOOG'], start="2022-10-01", end="2025-10-01")
#print(data.head())
##############################

close = data['Close']
daily_returns = close.pct_change().dropna()
large_moves = (daily_returns.abs() > 0.05).any(axis=1)
days_with_large_moves = daily_returns[large_moves]
num_days_large_moves = len(days_with_large_moves)
days_in_df = len(daily_returns)
percentage_large_moves = (num_days_large_moves / days_in_df) * 100 if days_in_df > 0 else 0


print(daily_returns.head())
print("Days in df:", days_in_df)
print("Days with large moves:", num_days_large_moves)
print(f"Percentage of days with large moves: {percentage_large_moves:.2f}%")


close.plot(figsize=(12,6), title="GOOGL vs GOOG Closing Prices")
plt.ylabel("Price ($)")
#plt.show()

#correlation calcs 
correlation = daily_returns.corr()
print("Correlation between GOOGL and GOOG daily returns:", correlation.loc['GOOGL', 'GOOG'])

# cointegration test
from statsmodels.tsa.stattools import coint
coint_t, p_value, _ = coint(close['GOOGL'], close['GOOG'])
print(f"Cointegration test p-value between GOOGL and GOOG: {p_value:.4f}")