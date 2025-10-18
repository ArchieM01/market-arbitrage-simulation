import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
import itertools
import os


#Ins and outs
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
csv_path = os.path.join(project_root, 'data', 'spx_companies_list.csv')
output_path = os.path.join(project_root, 'data', 'cointegrated_pairs.csv')

try:
     spx_df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Error: File not found at", csv_path)
    exit() # Stop the script if the file is missing

#Read SPX tickers from CSV

tickers = spx_df.iloc[:,0].dropna().astype(str).str.strip().tolist() # Strip first column, remove NaNs, convert to str and strip whitespace

data = yf.download(tickers, start="2024-10-01", end="2025-10-01")

if data is None or data.empty:
    print("No valid data fetched for the provided tickers.")
    exit()

close = data['Close']
close.dropna(axis='columns', how='all', inplace=True) # Drops columns

close_prices = data['Close'].copy()
close_prices.dropna(axis='columns', how='all', inplace=True) # Drops columns with all NaNs
close_prices.dropna(axis='columns', inplace=True) # Drops columns with any NaNs

valid_tickers = close_prices.columns.tolist() #list of tickers with actual data


with open(output_path, 'w') as f:
    f.write("Ticker 1,Ticker 2,P-Value\n")

    for t1, t2 in itertools.combinations(valid_tickers, 2): #gets all possible pairs
        try:
            score, p_value, _ = coint(close_prices[t1], close_prices[t2])

            if p_value < 0.05:
                # print a counter of cointegrated pairs found
                num_coints = sum(1 for line in open(output_path)) - 1 #subtract header
                print(f"Coint pairs found: {num_coints + 1}", end='\r')
                f.write(f"{t1},{t2},{p_value:.4f}\n")

        except Exception as e: # e means exception object
            print(f"Error processing pair {t1} and {t2}: {e}")

print(f"\nCointegrated pairs saved to {output_path}")