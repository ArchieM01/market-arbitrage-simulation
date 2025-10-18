import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def calc_zscore(spread, window=21):
    """
    Calculate the z-score of the spread over a rolling window.
    21 trading days = approx. 1 month
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore

def backtest_pair(prices_1, prices_2, hedge_ratio, window=21, entry_z=2.0, exit_z=0.5):
    """
    Backtest a pair trading strategy based on z-score thresholds.
    
    Parameters:
    - prices_1: pd.Series of prices for asset 1
    - prices_2: pd.Series of prices for asset 2
    - hedge_ratio: float, the hedge ratio between asset 1 and asset 2
    - entry_z: float, z-score threshold to enter a trade
    - exit_z: float, z-score threshold to exit a trade
    
    Returns:
    - trades: list of dicts with trade details
    - total_return: float, total return from the strategy
    """
    spread = prices_1 - hedge_ratio * prices_2
    zscore = calc_zscore(spread)
    
    pnl = pd.Series(index=spread.index, data=0.0) # initialise P&L series
    
    position = 0 # initialise position as no position
    # 0 = No position
    # 1 = Long position
    # -1 = Short position

    #interate by day
    for i in range(window, len(spread)): # skips first 'window' days so z-score is valid

        # Short signal
        if zscore.iloc[i] > entry_z and position == 0:
            position = -1
            entry_price_1 = prices_1.iloc[i]
            entry_price_2 = prices_2.iloc[i]
            
        # Long signal
        elif zscore.iloc[i] < -entry_z and position == 0:
            position = 1
            entry_price_1 = prices_1.iloc[i]
            entry_price_2 = prices_2.iloc[i]

        # Exit signal for short position
        elif position == -1 and zscore.iloc[i] < exit_z:
            # Calculate return and record trade
            pnl_1 = entry_price_1 - prices_1.iloc[i] # entry price - current price
            pnl_2 = (prices_2.iloc[i] - entry_price_2) * hedge_ratio # current price - entry price * hedge ratio
            pnl.iloc[i] = pnl_1 + pnl_2

            position = 0 # exit position

        # Exit signal for long position
        elif position == 1 and zscore.iloc[i] > -exit_z:
            # Calculate return and record trade
            pnl_1 = prices_1.iloc[i] - entry_price_1 # current price - entry price
            pnl_2 = (entry_price_2 - prices_2.iloc[i]) * hedge_ratio # entry price - current price * hedge ratio
            pnl.iloc[i] = pnl_1 + pnl_2

            position = 0 # exit position

            # P&L for open positions
        # Open short position
        elif position == -1:
            pnl_1 = entry_price_1 - prices_1.iloc[i]
            pnl_2 = (prices_2.iloc[i] - entry_price_2) * hedge_ratio
            pnl.iloc[i] = pnl_1 + pnl_2
            
        # Open long position
        elif position == 1:
            pnl_1 = prices_1.iloc[i] - entry_price_1
            pnl_2 = (entry_price_2 - prices_2.iloc[i]) * hedge_ratio
            pnl.iloc[i] = pnl_1 + pnl_2
            
    return pnl.cumsum() # return cumulative P&L for time series

# paths
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
data_dir = os.path.join(project_root, 'data')

tradeable_pairs_path = os.path.join(data_dir, 'tradeable_pairs.csv')
price_data_path = os.path.join(data_dir, 'coint_pair_prices.csv')

# load data
try:
    tradeable_pairs = pd.read_csv(tradeable_pairs_path)
    prices = pd.read_csv(price_data_path, index_col=0, parse_dates=True)
    prices.dropna(inplace=True) # Drops rows with any NaNs
except FileNotFoundError as e:
    print("Error, file not found:", e)
    exit()

if tradeable_pairs.empty:
    print("No tradeable pairs found.")
    exit()

print(f"Backtesting {len(tradeable_pairs)} tradeable pairs...")

# run backtest for each pair
total_pnl = pd.Series(index=prices.index, dtype=float)

for index, row in tradeable_pairs.iterrows():
    t1 = row['Ticker 1']
    t2 = row['Ticker 2']
    hedge_ratio = row['Hedge Ratio']

    if t1 not in prices.columns or t2 not in prices.columns:
        print(f"Skipping pair {t1}, {t2} due to missing price data.")
        continue
    
    print(f"Backtesting pair: {t1}, {t2}")
    pair_pnl = backtest_pair(prices[t1], prices[t2], hedge_ratio, window=21)
    total_pnl = total_pnl.add(pair_pnl, fill_value=0) # aggregate P&L for all pairs

# results
print("backtest complete.")

final_pnl = total_pnl.dropna().iloc[-1] if not total_pnl.dropna().empty else 0.0
total_days_traded = (total_pnl.diff() != 0).sum() # any change in P&L indicates a day with a trade

print("Total days with trades executed:", total_days_traded)
print(f"Final cumulative P&L from all pairs: ${final_pnl:.2f}")

# plot cumulative P&L
print("Plotting cumulative P&L...")
plt.figure(figsize=(12,6))
total_pnl.plot()
plt.title('Cumulative P&L for all Tradeable Pairs')
plt.xlabel('Date')
plt.ylabel('Cumulative P&L ($)')
plt.grid()

plt_path = os.path.join(data_dir, 'cumulative_pnl.png')
plt.savefig(plt_path)
print(f"Cumulative P&L plot saved to {plt_path}")



    



    