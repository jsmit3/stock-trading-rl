import os
import sys
import json
import time
import pickle
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Data cache paths
DATA_DIR = Path("data_cache")
SP500_LIST_FILE = DATA_DIR / "sp500_list.json"
STOCK_DATA_DIR = DATA_DIR / "stocks"
METADATA_FILE = DATA_DIR / "metadata.json"

# Stock universe - S&P 500 top components
TOP_STOCKS = [
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'AMZN',  # Amazon
    'NVDA',  # NVIDIA
    'GOOGL', # Alphabet Class A
    'META',  # Meta Platforms
    'GOOG',  # Alphabet Class C
    'BRK-B', # Berkshire Hathaway
    'TSLA',  # Tesla
    'UNH',   # UnitedHealth
    'JPM',   # JPMorgan Chase
    'V',     # Visa
    'JNJ',   # Johnson & Johnson
    'PG',    # Procter & Gamble
    'MA',    # Mastercard
    'XOM',   # Exxon Mobil
    'HD',    # Home Depot
    'MRK',   # Merck
    'CVX',   # Chevron
    'AVGO',  # Broadcom
]

def setup_folders():
    """Create necessary folders for data caching."""
    for folder in [DATA_DIR, STOCK_DATA_DIR]:
        folder.mkdir(exist_ok=True, parents=True)
    print(f"Created folder structure at {os.getcwd()}")

def fetch_stock_data(ticker, period='max', interval='1d', verbose=True):
    """
    Fetch stock data using the yfinance Ticker approach.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period to fetch ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1d', '1wk', '1mo')
        verbose: Whether to print progress messages
        
    Returns:
        DataFrame with stock data or None if there was an error
    """
    if verbose:
        print(f"Fetching {ticker} data for period: {period}, interval: {interval}...")
    
    try:
        # Create ticker object
        ticker_obj = yf.Ticker(ticker)
        
        # Get historical data
        data = ticker_obj.history(period=period, interval=interval)
        
        # Check if we got valid data
        if data.empty:
            if verbose:
                print(f"No data received for {ticker}")
            return None
        
        # Rename columns to lowercase for consistency with your project
        data.columns = [col.lower() for col in data.columns]
        
        if verbose:
            print(f"✓ Downloaded {len(data)} rows for {ticker} ({data.index[0]} to {data.index[-1]})")
        
        return data
    
    except Exception as e:
        if verbose:
            print(f"Error fetching {ticker}: {str(e)}")
        return None

def fetch_longer_history(ticker, years=5, interval='1d', verbose=True):
    """
    Fetch longer historical data by trying longer periods first.
    
    Args:
        ticker: Stock ticker symbol
        years: Number of years of data to fetch
        interval: Data interval ('1d', '1wk', '1mo')
        verbose: Whether to print progress messages
        
    Returns:
        DataFrame with the longest available history
    """
    if verbose:
        print(f"Fetching {years} years of {ticker} data...")
    
    # Try periods from longest to shortest
    periods = ['max', '10y', '5y', '2y', '1y']
    
    best_data = None
    most_rows = 0
    
    # Try each period and keep the one with the most data
    for period in periods:
        if verbose:
            print(f"Trying period: {period}...")
            
        data = fetch_stock_data(ticker, period=period, interval=interval, verbose=False)
        
        if data is not None and not data.empty:
            rows = len(data)
            if verbose:
                print(f"✓ Got {rows} rows using period={period}")
            
            # Check if this is the most data so far
            if rows > most_rows:
                most_rows = rows
                best_data = data
                
                # If we have more than years*250 rows (approx trading days per year),
                # or if we're using 'max', we can stop
                if rows > years * 250 or period == 'max':
                    if verbose:
                        print(f"✓ Found sufficient data: {rows} rows")
                    break
    
    # If we have data from the period approach, return it
    if best_data is not None:
        if verbose:
            print(f"Using best data: {most_rows} rows")
        return best_data
    
    # If all period attempts failed, try the start date approach
    if verbose:
        print("Trying start date approach...")
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=start_date.strftime('%Y-%m-%d'), interval=interval)
        
        if not data.empty:
            if verbose:
                print(f"✓ Successfully fetched {len(data)} rows using date range")
            return data
    except Exception as e:
        if verbose:
            print(f"Start date approach failed: {str(e)}")
    
    if verbose:
        print(f"All attempts to fetch {ticker} data failed")
    return None

def fetch_by_date_chunks(ticker, years=5, chunk_size=2, interval='1d', verbose=True):
    """
    Fetch data in chunks of years and concatenate the results.
    
    Args:
        ticker: Stock ticker symbol
        years: Total years of data to fetch
        chunk_size: Years per chunk
        interval: Data interval ('1d', '1wk', '1mo')
        verbose: Whether to print progress messages
        
    Returns:
        DataFrame with concatenated stock data
    """
    if verbose:
        print(f"Fetching {years} years of {ticker} data in {chunk_size}-year chunks...")
    
    chunks = []
    end_date = datetime.now()
    
    # Create chunks from most recent to oldest
    for i in range(0, years, chunk_size):
        chunk_end = end_date - timedelta(days=i*365)
        chunk_start = chunk_end - timedelta(days=chunk_size*365)
        
        if verbose:
            print(f"Fetching chunk: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            chunk_data = ticker_obj.history(
                start=chunk_start.strftime('%Y-%m-%d'),
                end=chunk_end.strftime('%Y-%m-%d'),
                interval=interval
            )
            
            if not chunk_data.empty:
                if verbose:
                    print(f"✓ Got {len(chunk_data)} rows for chunk")
                chunks.append(chunk_data)
            else:
                if verbose:
                    print(f"No data for this chunk")
        except Exception as e:
            if verbose:
                print(f"Error fetching chunk: {str(e)}")
        
        # Add a small delay between chunks
        time.sleep(1)
    
    # Combine all chunks
    if chunks:
        combined_data = pd.concat(chunks).sort_index()
        
        # Remove duplicates
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
        
        if verbose:
            print(f"✓ Combined {len(combined_data)} rows from {len(chunks)} chunks")
        
        return combined_data
    
    return None

def save_to_cache(ticker, data):
    """Save stock data to cache."""
    cache_file = STOCK_DATA_DIR / f"{ticker}.pkl"
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {ticker} data to cache")
        return True
    except Exception as e:
        print(f"Error saving {ticker} to cache: {e}")
        return False

def load_from_cache(ticker):
    """Load stock data from cache."""
    cache_file = STOCK_DATA_DIR / f"{ticker}.pkl"
    
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
    
    if "volume" in data.columns:
        metadata["stocks"][ticker]["avg_volume"] = float(data["volume"].mean())
    if "close" in data.columns:
        metadata["stocks"][ticker]["avg_price"] = float(data["close"].mean())
        metadata["stocks"][ticker]["latest_price"] = float(data["close"].iloc[-1])
    
    # Save metadata
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False

def get_stock_data(ticker, force_download=False, plot=False):
    """
    Get stock data for a ticker, either from cache or by downloading.
    
    Args:
        ticker: Stock ticker symbol
        force_download: Whether to force a download even if cached data exists
        plot: Whether to plot the data
        
    Returns:
        DataFrame with stock data or None if not available
    """
    # Try to load from cache first
    if not force_download:
        cached_data = load_from_cache(ticker)
        if cached_data is not None and not cached_data.empty:
            print(f"Loaded {len(cached_data)} rows of {ticker} data from cache")
            print(f"Data range: {cached_data.index[0]} to {cached_data.index[-1]}")
            
            # Plot if requested
            if plot:
                plot_stock_data(ticker, cached_data)
                
            return cached_data
    
    # If not in cache or force_download, fetch from API
    print(f"Downloading {ticker} data...")
    
    # Try multiple methods to get the most data possible
    # Method 1: Try longest periods first
    data = fetch_longer_history(ticker, years=5)
    
    # Method 2: If that doesn't get enough data, try fetching in chunks
    if data is None or len(data) < 250 * 3:  # If less than ~3 years of data
        print("Trying chunk method...")
        data = fetch_by_date_chunks(ticker, years=5, chunk_size=2)
    
    if data is not None and not data.empty:
        print(f"Downloaded {len(data)} rows of data from {data.index[0]} to {data.index[-1]}")
        
        # Save to cache
        if save_to_cache(ticker, data):
            # Update metadata
            update_metadata(ticker, data)
            
            # Plot if requested
            if plot:
                plot_stock_data(ticker, data)
            
            return data
    
    print(f"Failed to get data for {ticker}")
    return None

def plot_stock_data(ticker, data):
    """Plot stock price and volume data."""
    if "close" not in data.columns:
        print("No price data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot price
    ax1.plot(data.index, data["close"])
    ax1.set_title(f"{ticker} Stock Price")
    ax1.set_ylabel("Price ($)")
    ax1.grid(True)
    
    # Plot volume if available
    if "volume" in data.columns:
        ax2.bar(data.index, data["volume"])
        ax2.set_title(f"{ticker} Trading Volume")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volume")
        ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(DATA_DIR / f"{ticker}_chart.png")
    print(f"Saved {ticker} chart to {DATA_DIR}/{ticker}_chart.png")
    
    # Display plot
    plt.show()

def get_all_stock_data(max_stocks=None, delay=1.5):
    """
    Get data for all stocks in the TOP_STOCKS list.
    
    Args:
        max_stocks: Maximum number of stocks to process (for testing)
        delay: Delay between API calls to avoid rate limiting
        
    Returns:
        Dictionary mapping ticker symbols to dataframes
    """
    setup_folders()
    
    # Determine which stocks to process
    stocks_to_process = TOP_STOCKS
    if max_stocks is not None:
        stocks_to_process = stocks_to_process[:max_stocks]
    
    # Process each stock
    stock_data = {}
    for i, ticker in enumerate(stocks_to_process):
        print(f"\n[{i+1}/{len(stocks_to_process)}] Processing {ticker}")
        
        data = get_stock_data(ticker)
        if data is not None:
            stock_data[ticker] = data
        
        # Add delay between stocks
        if i < len(stocks_to_process) - 1:
            time.sleep(delay)
    
    # Print summary
    success_count = len(stock_data)
    print(f"\nSuccessfully processed {success_count}/{len(stocks_to_process)} stocks")
    
    return stock_data

def get_sp500_list():
    """Get top S&P 500 stocks."""
    # For simplicity, we'll just return the TOP_STOCKS list
    return TOP_STOCKS

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch and cache stock data")
    parser.add_argument("--ticker", type=str, help="Specific ticker to fetch")
    parser.add_argument("--force", action="store_true", help="Force download even if cached")
    parser.add_argument("--plot", action="store_true", help="Plot the data")
    parser.add_argument("--all", action="store_true", help="Fetch all stocks")
    parser.add_argument("--max", type=int, help="Maximum number of stocks to fetch")
    
    args = parser.parse_args()
    
    setup_folders()
    
    if args.ticker:
        # Fetch a specific ticker
        get_stock_data(args.ticker, force_download=args.force, plot=args.plot)
    elif args.all:
        # Fetch all stocks
        get_all_stock_data(max_stocks=args.max)
    else:
        # Default: get AAPL data
        get_stock_data("AAPL", force_download=args.force, plot=args.plot)