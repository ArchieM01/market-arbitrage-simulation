import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import os

def check_stationarity(series):
    """Perform Augmented Dickey-Fuller test to check stationarity."""
    
    # autolag='AIC' lets the function choose the best lag based on Akaike Information Criterion
    result = adfuller(series, autolag='AIC')
    p_value = result[1] # p-value is the second element in the result tuple
    
    return p_value < 0.05  # Returns True if series is stationary (p < 0.05)

def find_hedge_ratio(series1, series2):
    """Calculate hedge ratio using OLS regression."""

    # Add constant to independent variable
    series2_with_const = sm.add_constant(series2)

    # Fit OLS regression model
    model = sm.OLS(series1, series2_with_const).fit()

    hedge_ratio = model.params[1]  # Slope coefficient
    spread = series1 - hedge_ratio * series2 # spread is the residuals from the regression

    return hedge_ratio, spread 

# paths
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

# ins and outs
pairs_csv_path = os.path.join(script_dir, 'cointegrated_pairs.csv')
data_dir = os.path.join(project_root, 'data') # Store data in new data folder
tradeable_pairs_path = os.path.join(data_dir, 'tradeable_pairs.csv')
price_data_path = os.path.join(data_dir, 'coint_pair_prices.csv')

# Read cointegrated pairs
try:
    pairs_df = pd.read_csv(pairs_csv_path)
except FileNotFoundError:
    print("Error: File not found at", pairs_csv_path)
    exit() # Stop the script if the file is missing

if pairs_df.empty:
    print("No cointegrated pairs found in the CSV.")
    exit()

os.makedirs(data_dir, exist_ok=True) # Ensure data directory exists

# Fetch historical price data for all tickers in the cointegrated pairs
unique_tickers = pd.unique(pairs_df[['Ticker 1', 'Ticker 2']].values.ravel('K'))

print(f"Fetching data for {len(unique_tickers)} tickers...")

# Download price data
prices_data = yf.download(list(unique_tickers), start="2024-10-01", end="2025-10-01")
prices = prices_data['Close']
prices.dropna(axis='columns', inplace=True) # Drops columns with any NaNs

#save price data for future reference
prices.to_csv(price_data_path)

# Find tradeable pairs based on spread stationarity
stationary_pairs = []

for index, row in pairs_df.iterrows():
    t1 = row['Ticker 1']
    t2 = row['Ticker 2']

    if t1 not in prices.columns or t2 not in prices.columns:
        print(f"Skipping pair {t1}, {t2} due to missing price data.")
        continue
    
    series1 = prices[t1]
    series2 = prices[t2]

    try:
        hedge_ratio, spread = find_hedge_ratio(series1, series2) # gets hedge ratio and spread

        if check_stationarity(spread):
            stationary_pairs.append({
                'Ticker 1': t1,
                'Ticker 2': t2,
                'Cointegration P-Value': row['P-Value'],
                'Hedge Ratio': hedge_ratio
            })


            print(f"Pair {t1}, {t2} is tradeable with hedge ratio {hedge_ratio:.4f}.")
        else:
            print(f"Pair {t1}, {t2} spread is not stationary.")
    
    except Exception as e:
        print(f"Error processing pair {t1} and {t2}: {e}")

# Save tradeable pairs to CSV
if not stationary_pairs:
    print("No tradeable pairs found.")

else:
    stationary_df = pd.DataFrame(stationary_pairs)
    stationary_df.to_csv(tradeable_pairs_path, index=False)
    print(f"{len(stationary_pairs)} tradeable pairs saved to {tradeable_pairs_path}")


