import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
import itertools
import os
import requests


#Ins and outs
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
output_path = os.path.join(project_root, 'data', 'cointegrated_pairs_TRAINING.csv')

print("Fetching SPX tickers from Wikipedia...")

# use user-agent to avoid request blocks
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

try:
    response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
    response.raise_for_status()  # Raise an error for bad responses

    tables = pd.read_html(response.text)
    tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
    print(f"Fetched {len(tickers)} tickers from Wikipedia SPX list.")
except Exception as e:
    print("Error fetching SPX tickers:", e)
    exit() # Stop the script if the tickers can't be fetched

Training_start = "2022-09-30"
Training_end = "2024-09-30"
Testing_start = "2024-10-01"
Testing_end = "2025-10-01"

Download_start = Training_start
Download_end = Testing_end

data = yf.download(tickers, start=Download_start, end=Download_end)

if data is None or data.empty:
    print("No valid data fetched for the provided tickers.")
    exit()

#close = data['Close']
#close.dropna(axis='columns', how='all', inplace=True) # Drops columns

close_prices = data['Close'].copy()
close_prices.dropna(axis='columns', how='all', inplace=True) # Drops columns with all NaNs
close_prices.dropna(axis='columns', inplace=True) # Drops columns with any NaNs

print(f"Creating training set...")

training_data = close_prices.loc[Training_start:Training_end]

if training_data.empty:
    print("Training data is empty for the specified date range.")
    exit()

valid_tickers = training_data.columns.tolist() #list of tickers with actual data

print(f"Found {len(valid_tickers)} valid tickers for training set.")

with open(output_path, 'w') as f:
    f.write("Ticker 1,Ticker 2,P-Value\n")

    for t1, t2 in itertools.combinations(valid_tickers, 2): #gets all possible pairs
        try:
            score, p_value, _ = coint(training_data[t1], training_data[t2])

            if p_value < 0.05:
                # print a counter of cointegrated pairs found
                num_coints = sum(1 for line in open(output_path)) - 1 #subtract header
                print(f"Coint pairs found in training data: {num_coints + 1}", end='\r')
                f.write(f"{t1},{t2},{p_value:.4f}\n")

        except Exception as e: # e means exception object
            print(f"Error processing pair {t1} and {t2}: {e}")

print(f"\nCointegrated pairs in training data saved to {output_path}")