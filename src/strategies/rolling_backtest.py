import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from tqdm import tqdm

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

def calc_rolling_hedge_ratio(series_1, series_2):
    """
    Calculate rolling hedge ratio using OLS regression over a specified window.
    252 trading days = approx. 1 year
    """
    series_2_with_const = sm.add_constant(series_2)
    model = sm.OLS(series_1, series_2_with_const).fit()
    hedge_ratio = model.params.iloc[1]  # Slope coefficient
    return hedge_ratio

# paths
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
data_dir = os.path.join(project_root, 'data')

tradeable_pairs_path = os.path.join(data_dir, 'tradeable_pairs.csv')
#price_data_path = os.path.join(data_dir, 'coint_pair_prices.csv')

Training_start = "2022-09-30"
Training_end = "2024-09-30"
Testing_start = "2024-10-01"
Testing_end = "2025-10-01"

Download_start = Training_start
Download_end = Testing_end

averaging_window = 252 # 252 trading days = approx. 1 year
zscore_window = 21 # 21 trading days = approx. 1 month
entry_z = 2.0
exit_z = 0.1

# load data
tradeable_pairs_path = os.path.join(data_dir, 'tradeable_pairs_TRAINING.csv')
try:
    tradeable_pairs = pd.read_csv(tradeable_pairs_path)
except FileNotFoundError:
    print("Error: File not found at", tradeable_pairs_path)
    exit() # Stop the script if the file is missing

if tradeable_pairs.empty:
    print("No tradeable pairs found in the CSV.")
    exit()

unique_tickers = pd.unique(tradeable_pairs[['Ticker 1', 'Ticker 2']].values.ravel('K'))

print(f"Fetching data for {len(unique_tickers)} tickers...")
prices_data = yf.download(list(unique_tickers), start=Download_start, end=Download_end)
prices = prices_data['Close']

prices.dropna(axis='columns', how='all', inplace=True) # Drops columns with all NaNs
prices.dropna(inplace=True) # Drops rows with any NaNs

print(f"Backtesting {len(tradeable_pairs)} tradeable pairs...")

# run backtest for each pair
print(f"Starting rolling backtest on {len(tradeable_pairs)} pairs...")
total_pnl = pd.Series(index=prices.loc[Testing_start:Testing_end].index, dtype=float)

for index, row in tqdm(tradeable_pairs.iterrows(), total=len(tradeable_pairs), desc="Backtesting pairs"):
    t1 = row['Ticker 1']
    t2 = row['Ticker 2']

    if t1 not in prices.columns or t2 not in prices.columns:
        print(f"Skipping pair {t1}, {t2} due to missing price data.")
        continue


    prices_1 = prices[t1]
    prices_2 = prices[t2]

    pair_pnl = pd.Series(index=total_pnl.index, data=0.0)
    position = 0 # initialise position as no position

    # iterate through testing period
    for i in range(len(total_pnl)):
        current_date = total_pnl.index[i] # get current date
        current_date_loc = prices.index.get_loc(current_date) # get current date location in prices index
    
        if current_date_loc < averaging_window:
            continue # skip until we have enough data for rolling hedge ratio

        averaging_start_loc = current_date_loc - averaging_window
        averaging_end_loc = current_date_loc - 1 # up to day before current date

        # averaged prices for hedge ratio calculation
        averaged_1 = prices_1.iloc[averaging_start_loc:averaging_end_loc]
        averaged_2 = prices_2.iloc[averaging_start_loc:averaging_end_loc]

        # calculate rolling hedge ratio
        hedge_ratio = calc_rolling_hedge_ratio(averaged_1, averaged_2)
        averaged_spread = averaged_1 - hedge_ratio * averaged_2

        averaged_zscore = (averaged_spread.iloc[-zscore_window:])
        
        rolling_mean = averaged_zscore.mean()
        rolling_std = averaged_zscore.std()

        current_spread = prices_1.iloc[current_date_loc] - hedge_ratio * prices_2.iloc[current_date_loc]
        current_zscore = (current_spread - rolling_mean) / rolling_std

        prev_date_loc = current_date_loc - 1
        prev_price_1 = prices_1.iloc[prev_date_loc]
        prev_price_2 = prices_2.iloc[prev_date_loc]

        # positions
        # Short signal
        if position == -1:
            pnl_1 = prev_price_1 - prices_1.iloc[current_date_loc]
            pnl_2 = (prices_2.iloc[current_date_loc] - prev_price_2) * hedge_ratio
            pair_pnl.iloc[i] = pnl_1 + pnl_2
        elif position == 1:
            pnl_1 = prices_1.iloc[current_date_loc] - prev_price_1
            pnl_2 = (prev_price_2 - prices_2.iloc[current_date_loc]) * hedge_ratio
            pair_pnl.iloc[i] = pnl_1 + pnl_2

        if current_zscore > entry_z and position == 0:
            position = -1 # enter short position
        elif current_zscore < -entry_z and position == 0:
            position = 1 # enter long position
        elif current_zscore < exit_z and position == -1:
            position = 0 # exit short position
        elif current_zscore > -exit_z and position == 1:
            position = 0 # exit long position

    total_pnl = total_pnl.add(pair_pnl.cumsum(), fill_value=0) # aggregate P&L for all pairs

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

plt_path = os.path.join(data_dir, 'cumulative_pnl_TESTING.png')
plt.savefig(plt_path)
print(f"Cumulative P&L plot saved to {plt_path}")