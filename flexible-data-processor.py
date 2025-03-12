import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import time
from datetime import datetime
import traceback

# Path to the SQLite database
DB_PATH = Path("data_cache/alpaca_data.db")
OUTPUT_DIR = Path("data_cache/stocks")

def analyze_database():
    """Analyze the database structure and content"""
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get table information
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    print(f"Found {len(tables)} tables: {', '.join(tables)}")
    
    analysis = {
        'tables': tables,
        'timeframes': [],
        'symbols': [],
        'has_daily': False,
        'daily_timeframe': None
    }
    
    # Check for historical_data table
    if 'historical_data' in tables:
        # Check timeframes
        cursor.execute("SELECT DISTINCT timeframe FROM historical_data")
        timeframes = [row[0] for row in cursor.fetchall()]
        analysis['timeframes'] = timeframes
        
        print(f"\nFound {len(timeframes)} timeframes: {', '.join(timeframes)}")
        
        # Count rows per timeframe
        print("\nRow count by timeframe:")
        for tf in timeframes:
            cursor.execute("SELECT COUNT(*) FROM historical_data WHERE timeframe = ?", (tf,))
            count = cursor.fetchone()[0]
            print(f"  {tf}: {count:,} rows")
        
        # Check for daily timeframe
        daily_timeframes = [tf for tf in timeframes if tf.lower() in ['1day', 'd', 'day', 'daily']]
        if daily_timeframes:
            daily_tf = daily_timeframes[0]
            analysis['has_daily'] = True
            analysis['daily_timeframe'] = daily_tf
            
            print(f"\nFound daily timeframe: {daily_tf}")
            
            # Count symbols with daily data
            cursor.execute(f"SELECT DISTINCT symbol FROM historical_data WHERE timeframe = ?", (daily_tf,))
            daily_symbols = [row[0] for row in cursor.fetchall()]
            
            print(f"  {len(daily_symbols)} symbols have daily data")
            sample = daily_symbols[:10]
            print(f"  Sample symbols: {', '.join(sample)}")
            
            # Check date range
            cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM historical_data WHERE timeframe = ?", (daily_tf,))
            min_date, max_date = cursor.fetchone()
            print(f"  Date range: {min_date} to {max_date}")
        
        # Check all available symbols
        cursor.execute("SELECT DISTINCT symbol FROM historical_data")
        all_symbols = [row[0] for row in cursor.fetchall()]
        analysis['symbols'] = all_symbols
        
        print(f"\nFound {len(all_symbols)} total symbols in the database")
        print(f"  Sample: {', '.join(all_symbols[:10])}")
        
        # Check row count per symbol
        cursor.execute("""
            SELECT symbol, COUNT(*) as count 
            FROM historical_data 
            GROUP BY symbol 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_symbols = cursor.fetchall()
        
        print("\nTop 10 symbols by data count:")
        for symbol, count in top_symbols:
            print(f"  {symbol}: {count:,} rows")
    
    conn.close()
    return analysis

def get_available_symbols():
    """Get list of all available symbols in the database"""
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return []
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query for distinct symbols
    try:
        cursor.execute("SELECT DISTINCT symbol FROM historical_data")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols
    except Exception as e:
        print(f"Error getting symbols: {e}")
        conn.close()
        return []

def get_daily_data(symbol, timeframe='1Day'):
    """
    Get daily price data for a symbol directly from the database.
    
    Args:
        symbol: Ticker symbol
        timeframe: Daily timeframe identifier
        
    Returns:
        DataFrame with daily OHLCV data or None
    """
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Build query for daily data
    query = """
    SELECT 
        timestamp, symbol, open, high, low, close, volume
    FROM 
        historical_data
    WHERE 
        symbol = ? AND timeframe = ?
    ORDER BY 
        timestamp
    """
    
    try:
        # Execute query
        df = pd.read_sql_query(query, conn, params=[symbol, timeframe], parse_dates=['timestamp'])
        conn.close()
        
        # Check if we got data
        if len(df) == 0:
            print(f"No daily data found for {symbol} with timeframe {timeframe}")
            return None
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        print(f"Retrieved {len(df)} daily bars for {symbol}")
        return df
    
    except Exception as e:
        print(f"Error fetching daily data for {symbol}: {e}")
        conn.close()
        return None

def get_minute_data(symbol, timeframe='1Min'):
    """
    Get minute-level price data for a symbol.
    
    Args:
        symbol: Ticker symbol
        timeframe: Minute timeframe identifier
        
    Returns:
        DataFrame with minute-level OHLCV data or None
    """
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Build query for minute data
    query = """
    SELECT 
        timestamp, symbol, open, high, low, close, volume
    FROM 
        historical_data
    WHERE 
        symbol = ? AND timeframe = ?
    ORDER BY 
        timestamp
    """
    
    try:
        # Execute query
        df = pd.read_sql_query(query, conn, params=[symbol, timeframe], parse_dates=['timestamp'])
        conn.close()
        
        # Check if we got data
        if len(df) == 0:
            print(f"No minute data found for {symbol} with timeframe {timeframe}")
            return None
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        print(f"Retrieved {len(df):,} minute bars for {symbol}")
        return df
    
    except Exception as e:
        print(f"Error fetching minute data for {symbol}: {e}")
        conn.close()
        return None

def aggregate_to_daily(df):
    """
    Aggregate minute-level data to daily OHLCV bars.
    
    Args:
        df: DataFrame with minute-level OHLCV data
        
    Returns:
        DataFrame with daily OHLCV data
    """
    if df is None or len(df) == 0:
        return None
    
    try:
        # Resample to daily frequency
        daily_df = df.resample('D').agg({
            'open': 'first',        # First price of the day
            'high': 'max',          # Highest price of the day
            'low': 'min',           # Lowest price of the day
            'close': 'last',        # Last price of the day
            'volume': 'sum',        # Total volume for the day
            'symbol': 'first'       # Keep the symbol
        })
        
        # Filter out days with no data (weekends, holidays)
        daily_df = daily_df.dropna(subset=['open', 'high', 'low', 'close'])
        
        print(f"Aggregated {len(df):,} minute bars to {len(daily_df)} daily bars")
        return daily_df
    
    except Exception as e:
        print(f"Error aggregating to daily: {e}")
        return None

def clean_daily_data(df):
    """
    Clean and validate daily OHLCV data.
    
    Args:
        df: DataFrame with daily OHLCV data
        
    Returns:
        Cleaned DataFrame
    """
    if df is None or len(df) == 0:
        return None
    
    # Make a copy
    cleaned_df = df.copy()
    
    # Convert all price and volume columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Check for and fill NaN values
    if cleaned_df.isnull().any().any():
        print(f"Warning: {cleaned_df.isnull().sum().sum()} NaN values found - filling with forward and backward fill")
        cleaned_df = cleaned_df.fillna(method='ffill').fillna(method='bfill')
    
    # Check for negative or zero prices
    for col in ['open', 'high', 'low', 'close']:
        if (cleaned_df[col] <= 0).any():
            bad_count = (cleaned_df[col] <= 0).sum()
            print(f"Warning: {bad_count} non-positive values found in {col} - replacing with small positive value")
            cleaned_df.loc[cleaned_df[col] <= 0, col] = 0.01
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    inconsistent_high = (cleaned_df['high'] < cleaned_df[['open', 'close']].max(axis=1)).sum()
    inconsistent_low = (cleaned_df['low'] > cleaned_df[['open', 'close']].min(axis=1)).sum()
    
    if inconsistent_high > 0:
        print(f"Fixing {inconsistent_high} rows where high < max(open, close)")
        cleaned_df['high'] = cleaned_df[['high', 'open', 'close']].max(axis=1)
    
    if inconsistent_low > 0:
        print(f"Fixing {inconsistent_low} rows where low > min(open, close)")
        cleaned_df['low'] = cleaned_df[['low', 'open', 'close']].min(axis=1)
    
    # Make volume an integer
    if 'volume' in cleaned_df.columns:
        cleaned_df['volume'] = cleaned_df['volume'].fillna(0).astype(int)
    
    # Reset symbol as a column (not index)
    if 'symbol' in cleaned_df.columns:
        symbol_value = df['symbol'].iloc[0] if 'symbol' in df.columns else None
        cleaned_df['symbol'] = cleaned_df['symbol'].fillna(symbol_value)
    
    # Sort by index
    cleaned_df = cleaned_df.sort_index()
    
    return cleaned_df

def process_symbol_data(symbol, force=False, plot=False, analysis=None):
    """
    Process data for a symbol, using daily data if available or aggregating minute data.
    
    Args:
        symbol: Ticker symbol
        force: Whether to force reprocessing if file already exists
        plot: Whether to generate a plot
        analysis: Database analysis results
        
    Returns:
        DataFrame with processed daily data or None if error
    """
    # Check if output file already exists
    output_path = OUTPUT_DIR / f"{symbol}.csv"
    if os.path.exists(output_path) and not force:
        print(f"Output file already exists: {output_path}")
        
        # Load existing file
        try:
            daily_df = pd.read_csv(output_path, parse_dates=['timestamp'], index_col='timestamp')
            print(f"Loaded {len(daily_df)} days from existing file")
            
            # Generate plot if requested
            if plot:
                plot_daily_data(daily_df, symbol)
                
            return daily_df
        except Exception as e:
            print(f"Error loading existing file: {e}")
            # Continue with processing
    
    # Use database analysis to determine the best way to get data
    has_daily = False
    daily_timeframe = None
    
    if analysis is not None:
        has_daily = analysis.get('has_daily', False)
        daily_timeframe = analysis.get('daily_timeframe')
    
    # Try to get daily data directly if available
    daily_df = None
    if has_daily and daily_timeframe:
        print(f"Getting daily data directly for {symbol}...")
        daily_df = get_daily_data(symbol, timeframe=daily_timeframe)
    
    # If we couldn't get daily data directly, try aggregating minute data
    if daily_df is None or len(daily_df) == 0:
        print(f"No daily data found, trying to aggregate from minute data...")
        
        # Get minute data
        minute_df = get_minute_data(symbol)
        
        if minute_df is None or len(minute_df) == 0:
            print(f"No minute data found for {symbol}")
            return None
        
        # Aggregate to daily
        daily_df = aggregate_to_daily(minute_df)
        
        if daily_df is None or len(daily_df) == 0:
            print(f"Failed to aggregate daily data for {symbol}")
            return None
    
    # Clean the data
    print("Cleaning data...")
    daily_df = clean_daily_data(daily_df)
    
    if daily_df is None or len(daily_df) == 0:
        print(f"Failed to clean daily data for {symbol}")
        return None
    
    # Save to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Make sure timestamp is included as a column for saving
    df_to_save = daily_df.copy()
    df_to_save['timestamp'] = df_to_save.index
    
    df_to_save.to_csv(output_path, index=False)
    print(f"Saved {len(daily_df)} daily bars to {output_path}")
    
    # Generate plot if requested
    if plot:
        plot_daily_data(daily_df, symbol)
    
    return daily_df

def plot_daily_data(df, symbol):
    """
    Generate a plot of daily OHLCV data.
    
    Args:
        df: DataFrame with daily OHLCV data
        symbol: Ticker symbol
    """
    if df is None or len(df) == 0:
        print(f"No data to plot for {symbol}")
        return
    
    # Create plot directory
    os.makedirs("data_cache/plots", exist_ok=True)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price
    ax1.plot(df.index, df['close'], label='Close')
    ax1.set_title(f"{symbol} Daily Price")
    ax1.set_ylabel("Price ($)")
    ax1.grid(True)
    ax1.legend()
    
    # Add date range to title
    date_range = f"({df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')})"
    ax1.set_title(f"{symbol} Daily Price {date_range}")
    
    # Plot volume
    ax2.bar(df.index, df['volume'], label='Volume')
    ax2.set_title(f"{symbol} Daily Volume")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Volume")
    ax2.grid(True)
    ax2.legend()
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"data_cache/plots/{symbol}_daily.png", dpi=150)
    print(f"Saved plot to data_cache/plots/{symbol}_daily.png")
    
    # Show plot
    plt.show()

def process_multiple_symbols(symbols, force=False, plot=False, analysis=None):
    """
    Process multiple symbols.
    
    Args:
        symbols: List of symbols to process
        force: Whether to force reprocessing of existing files
        plot: Whether to generate plots
        analysis: Database analysis results
        
    Returns:
        Dictionary with results
    """
    print(f"Processing {len(symbols)} symbols...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each symbol
    results = {
        'success': 0,
        'error': 0,
        'total': len(symbols)
    }
    
    for i, symbol in enumerate(symbols):
        try:
            print(f"\n[{i+1}/{len(symbols)}] Processing {symbol}")
            
            # Process symbol
            df = process_symbol_data(symbol, force=force, plot=plot, analysis=analysis)
            
            if df is not None and len(df) > 0:
                results['success'] += 1
            else:
                results['error'] += 1
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            traceback.print_exc()
            results['error'] += 1
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successful: {results['success']}/{results['total']}")
    print(f"Failed: {results['error']}/{results['total']}")
    
    return results

def get_top_symbols_by_data_count(limit=20):
    """
    Get the top symbols by data count.
    
    Args:
        limit: Maximum number of symbols to return
        
    Returns:
        List of top symbol names
    """
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return []
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query for top symbols by data count
    try:
        cursor.execute("""
            SELECT symbol, COUNT(*) as count 
            FROM historical_data 
            GROUP BY symbol 
            ORDER BY count DESC 
            LIMIT ?
        """, (limit,))
        
        top_symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return top_symbols
    except Exception as e:
        print(f"Error getting top symbols: {e}")
        conn.close()
        return []

def main():
    parser = argparse.ArgumentParser(description="Process Alpaca Data for RL Training")
    parser.add_argument("--analyze", action="store_true", help="Analyze database")
    parser.add_argument("--symbol", type=str, help="Process a specific symbol")
    parser.add_argument("--symbols", type=str, nargs="+", help="Process multiple symbols")
    parser.add_argument("--all", action="store_true", help="Process all available symbols")
    parser.add_argument("--top", type=int, default=20, help="Process top N symbols by data count")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of existing files")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Analyze database first
    print("Analyzing database...")
    analysis = analyze_database()
    
    # Just show analysis if requested
    if args.analyze:
        return
    
    # Process symbols based on arguments
    if args.symbol:
        # Process a single symbol
        process_symbol_data(args.symbol, force=args.force, plot=args.plot, analysis=analysis)
    elif args.symbols:
        # Process specified symbols
        process_multiple_symbols(args.symbols, force=args.force, plot=args.plot, analysis=analysis)
    elif args.all:
        # Process all symbols
        symbols = get_available_symbols()
        process_multiple_symbols(symbols, force=args.force, plot=args.plot, analysis=analysis)
    elif args.top:
        # Process top symbols by data count
        top_symbols = get_top_symbols_by_data_count(limit=args.top)
        print(f"Processing top {len(top_symbols)} symbols by data count")
        process_multiple_symbols(top_symbols, force=args.force, plot=args.plot, analysis=analysis)
    else:
        # Nothing specified, show help
        parser.print_help()

if __name__ == "__main__":
    main()