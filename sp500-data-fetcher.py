import os
import sys
import json
import time
import pickle
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

# Data cache paths
DATA_DIR = Path("data_cache")
SP500_LIST_FILE = DATA_DIR / "sp500_list.json"
SP500_DATA_DIR = DATA_DIR / "stocks"
METADATA_FILE = DATA_DIR / "metadata.json"

# Create necessary folders
def setup_folders():
    """Create necessary folders for data caching."""
    for folder in [DATA_DIR, SP500_DATA_DIR]:
        folder.mkdir(exist_ok=True, parents=True)
    print(f"Created folder structure at {os.getcwd()}")

# Test stocks to use - well-known, liquid stocks that should work reliably
TEST_STOCKS = [
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'AMZN',  # Amazon
    'GOOGL', # Alphabet (Google)
    'META',  # Meta (Facebook)
    'TSLA',  # Tesla
    'BRK-B', # Berkshire Hathaway
    'JPM',   # JPMorgan Chase
    'JNJ',   # Johnson & Johnson
    'V',     # Visa
]

def download_stock_data(ticker, start_date="2018-01-01", end_date=None, verbose=True):
    """
    Download stock data for a single ticker with detailed error reporting.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD), defaults to today
        verbose: Whether to print detailed messages
        
    Returns:
        data: OHLCV dataframe for the stock
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if verbose:
        print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    try:
        # First try direct history method
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            show_errors=False
        )
        
        if data.empty or len(data) < 50:
            if verbose:
                print(f"Direct download failed for {ticker}, trying Ticker object...")
            
            # Try with Ticker object if direct method fails
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
        
        # Check if we got valid data
        if data.empty:
            if verbose:
                print(f"❌ No data received for {ticker}")
            return pd.DataFrame()
        
        if len(data) < 50:
            if verbose:
                print(f"❌ Insufficient data for {ticker}: only {len(data)} rows")
            return pd.DataFrame()
        
        # Rename columns to lowercase for consistency with our project
        data.columns = [col.lower() for col in data.columns]
        
        if verbose:
            print(f"✓ Downloaded {len(data)} rows for {ticker} ({data.index[0]} to {data.index[-1]})")
        
        return data
    
    except Exception as e:
        if verbose:
            print(f"❌ Error downloading {ticker}: {str(e)}")
        return pd.DataFrame()

def save_to_cache(ticker, data):
    """Save stock data to cache."""
    cache_file = SP500_DATA_DIR / f"{ticker}.pkl"
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving {ticker} to cache: {e}")
        return False

def load_from_cache(ticker):
    """Load stock data from cache."""
    cache_file = SP500_DATA_DIR / f"{ticker}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Error loading {ticker} from cache: {e}")
    
    return None

def update_metadata(ticker, data):
    """Update metadata for a stock."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
        except Exception:
            metadata = {"last_update": "", "stocks": {}}
    else:
        metadata = {"last_update": "", "stocks": {}}
    
    today = datetime.now().strftime("%Y-%m-%d")
    metadata["last_update"] = today
    
    # Update stock metadata
    metadata.setdefault("stocks", {}).setdefault(ticker, {})
    metadata["stocks"][ticker]["last_updated"] = today
    metadata["stocks"][ticker]["rows"] = len(data)
    metadata["stocks"][ticker]["start_date"] = data.index[0].strftime("%Y-%m-%d")
    metadata["stocks"][ticker]["end_date"] = data.index[-1].strftime("%Y-%m-%d")
    metadata["stocks"][ticker]["avg_volume"] = float(data["volume"].mean()) if "volume" in data.columns else 0
    metadata["stocks"][ticker]["avg_price"] = float(data["close"].mean()) if "close" in data.columns else 0
    
    # Save metadata
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False

def test_data_fetching():
    """Test data fetching for a few well-known stocks."""
    setup_folders()
    
    print("\n=== Testing Data Fetching with Well-Known Stocks ===\n")
    
    success_count = 0
    cache_success = 0
    
    for ticker in TEST_STOCKS:
        print(f"\nTesting {ticker}:")
        
        # Try to load from cache first
        cached_data = load_from_cache(ticker)
        
        if cached_data is not None and not cached_data.empty:
            print(f"✓ Loaded {len(cached_data)} rows for {ticker} from cache")
            print(f"  Date range: {cached_data.index[0]} to {cached_data.index[-1]}")
            
            if 'close' in cached_data.columns:
                print(f"  Latest price: ${cached_data['close'].iloc[-1]:.2f}")
            
            success_count += 1
            continue
        
        # If not in cache, download
        print(f"Not found in cache, downloading...")
        
        # Start from 2020 (post-COVID) for faster testing
        data = download_stock_data(ticker, start_date="2020-01-01", verbose=True)
        
        if not data.empty:
            success_count += 1
            
            # Save to cache
            if save_to_cache(ticker, data):
                cache_success += 1
                print(f"✓ Saved {ticker} data to cache")
                
                # Update metadata
                if update_metadata(ticker, data):
                    print(f"✓ Updated metadata for {ticker}")
                else:
                    print(f"❌ Failed to update metadata for {ticker}")
            else:
                print(f"❌ Failed to save {ticker} data to cache")
        
        # Add delay to avoid rate limiting
        time.sleep(2)
    
    # Print summary
    print(f"\n=== Data Fetching Test Summary ===")
    print(f"Successful downloads/loads: {success_count}/{len(TEST_STOCKS)}")
    print(f"Successfully cached: {cache_success}")
    
    # If we couldn't get any data, there's likely a more serious issue
    if success_count == 0:
        print("\n⚠️ WARNING: Could not retrieve any stock data!")
        print("Possible issues:")
        print("1. Network connectivity problems")
        print("2. Yahoo Finance API changes")
        print("3. Rate limiting or IP blocking")
        print("4. Proxy settings interference")
        print("\nTry running this script on a different network or after waiting a while.")
    else:
        print(f"\n✓ Data fetching is working for {success_count}/{len(TEST_STOCKS)} test stocks.")
        print("You can proceed with training using cached data.")
    
    return success_count > 0

if __name__ == "__main__":
    try:
        test_data_fetching()
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)