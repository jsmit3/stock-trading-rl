import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import time
from datetime import datetime

# Path to the SQLite database
DB_PATH = Path("data_cache/alpaca_data.db")
OUTPUT_DIR = Path("data_cache/stocks")

def analyze_database_schema():
    """Print schema information about the database"""
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = {}
    schema_info['tables'] = [table[0] for table in tables]
    
    print(f"Found {len(tables)} tables: {', '.join(schema_info['tables'])}")
    
    # Look for a table containing price data
    for table_name in schema_info['tables']:
        print(f"\nAnalyzing table: {table_name}")
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        column_names = [col[1] for col in columns]
        print(f"Columns: {', '.join(column_names)}")
        
        # Check for row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"Row count: {row_count:,}")
        
        # Get a sample row
        if row_count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            sample_row = cursor.fetchone()
            print(f"Sample row: {sample_row}")
    
    # Close connection
    conn.close()
    return schema_info

def detect_price_table():
    """
    Detect which table contains price data and return schema information.
    
    Returns:
        Dictionary with schema information or None if not found
    """
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = {
        'tables': [table[0] for table in tables],
        'price_table': None,
        'symbol_column': None,
        'datetime_column': None,
        'column_mapping': {}
    }
    
    # Look for a table containing price data
    for table_name in schema_info['tables']:
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        column_names = [col[1].lower() for col in columns]
        
        # Check if this looks like a price table
        price_columns = ['open', 'high', 'low', 'close']
        price_columns_present = sum(col in column_names for col in price_columns)
        
        if price_columns_present >= 3:  # Allow for some missing columns
            schema_info['price_table'] = table_name
            
            # Find the symbol column
            for col in columns:
                col_id, name, dtype, notnull, default_val, pk = col
                name_lower = name.lower()
                
                # Look for symbol column
                if name_lower in ['symbol', 'ticker', 'stock', 'security']:
                    schema_info['symbol_column'] = name
                
                # Look for datetime column
                if name_lower in ['timestamp', 'time', 'date', 'datetime']:
                    schema_info['datetime_column'] = name
                
                # Map column names to our expected format
                for expected in ['open', 'high', 'low', 'close', 'volume']:
                    if expected in name_lower or name_lower == expected:
                        schema_info['column_mapping'][expected] = name
            
            # If we found a price table, break
            if schema_info['symbol_column'] and schema_info['datetime_column']:
                break
    
    conn.close()
    
    if schema_info['price_table'] and schema_info['symbol_column'] and schema_info['datetime_column']:
        return schema_info
    
    return None

def get_available_symbols(schema_info):
    """
    Get a list of available symbols in the database.
    
    Args:
        schema_info: Dictionary with schema information
        
    Returns:
        List of symbols or empty list if error
    """
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return []
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get list of symbols
    try:
        cursor.execute(f"SELECT DISTINCT {schema_info['symbol_column']} FROM {schema_info['price_table']}")
        symbols = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return symbols
    except Exception as e:
        print(f"Error getting symbols: {e}")
        conn.close()
        return []

def get_date_range(schema_info):
    """
    Get the full date range available in the database.
    
    Args:
        schema_info: Dictionary with schema information
        
    Returns:
        Tuple of (min_date, max_date) as strings or None if error
    """
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get min and max dates
    try:
        cursor.execute(f"SELECT MIN({schema_info['datetime_column']}), MAX({schema_info['datetime_column']}) FROM {schema_info['price_table']}")
        min_date, max_date = cursor.fetchone()
        conn.close()
        return min_date, max_date
    except Exception as e:
        print(f"Error getting date range: {e}")
        conn.close()
        return None

def count_rows_per_symbol(schema_info, limit=20):
    """
    Count the number of rows for each symbol.
    
    Args:
        schema_info: Dictionary with schema information
        limit: Maximum number of symbols to show
        
    Returns:
        DataFrame with counts
    """
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Get counts per symbol
    try:
        query = f"""
        SELECT {schema_info['symbol_column']} as symbol, COUNT(*) as row_count
        FROM {schema_info['price_table']}
        GROUP BY {schema_info['symbol_column']}
        ORDER BY row_count DESC
        LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error counting rows: {e}")
        conn.close()
        return None

def extract_price_data(schema_info, symbol, start_date=None, end_date=None):
    """
    Extract price data for a specific symbol.
    
    Args:
        schema_info: Dictionary with schema information
        symbol: Ticker symbol
        start_date: Start date or None for all data
        end_date: End date or None for all data
        
    Returns:
        DataFrame with OHLCV data or None if error
    """
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Build query based on schema
    select_columns = [
        f"{schema_info['datetime_column']} as timestamp",
        f"{schema_info['symbol_column']} as symbol"
    ]
    
    for expected, actual in schema_info['column_mapping'].items():
        select_columns.append(f"{actual} as {expected}")
    
    select_clause = ", ".join(select_columns)
    
    query = f"""
    SELECT {select_clause}
    FROM {schema_info['price_table']}
    WHERE {schema_info['symbol_column']} = ?
    """
    
    params = [symbol]
    
    if start_date:
        query += f" AND {schema_info['datetime_column']} >= ?"
        params.append(start_date)
    
    if end_date:
        query += f" AND {schema_info['datetime_column']} <= ?"
        params.append(end_date)
    
    query += f" ORDER BY {schema_info['datetime_column']}"
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if len(df) == 0:
            print(f"No data found for {symbol}")
            return None
        
        # Convert timestamp to datetime and set as index
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            except:
                print("Warning: Could not convert timestamp to datetime")
        
        return df
    
    except Exception as e:
        print(f"Error extracting data for {symbol}: {e}")
        conn.close()
        return None

def preprocess_data(df):
    """
    Preprocess the data for training.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Preprocessed DataFrame or None if error
    """
    if df is None or len(df) == 0:
        return None
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Check for required columns
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in processed_df.columns:
            print(f"Missing required column: {col}")
            return None
    
    # Add volume if missing
    if 'volume' not in processed_df.columns:
        processed_df['volume'] = 0
    
    # Ensure all price columns are numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Check for and clean NaN values
    if processed_df.isnull().any().any():
        print(f"Warning: {processed_df.isnull().sum().sum()} NaN values found - filling with forward and backward fill")
        processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
    
    # Check for negative or zero prices
    for col in ['open', 'high', 'low', 'close']:
        if (processed_df[col] <= 0).any():
            bad_count = (processed_df[col] <= 0).sum()
            print(f"Warning: {bad_count} non-positive values found in {col} - replacing with small positive value")
            processed_df.loc[processed_df[col] <= 0, col] = 0.01
    
    # Ensure correct OHLC relationships
    # - high should be >= max(open, close)
    # - low should be <= min(open, close)
    processed_df['high'] = processed_df[['high', 'open', 'close']].max(axis=1)
    processed_df['low'] = processed_df[['low', 'open', 'close']].min(axis=1)
    
    # Check for duplicate indices
    if processed_df.index.duplicated().any():
        print(f"Warning: {processed_df.index.duplicated().sum()} duplicate timestamps found - keeping first occurrence")
        processed_df = processed_df[~processed_df.index.duplicated(keep='first')]
    
    # Sort by index
    processed_df = processed_df.sort_index()
    
    return processed_df

def export_data(df, symbol, output_dir=OUTPUT_DIR):
    """
    Export data to CSV format.
    
    Args:
        df: DataFrame with processed OHLCV data
        symbol: Ticker symbol
        output_dir: Output directory
        
    Returns:
        Path to the exported file or None if error
    """
    if df is None or len(df) == 0:
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file path
    output_path = os.path.join(output_dir, f"{symbol}.csv")
    
    try:
        # Export to CSV
        df.to_csv(output_path)
        return output_path
    except Exception as e:
        print(f"Error exporting data for {symbol}: {e}")
        return None

def plot_data(df, symbol):
    """
    Plot OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Ticker symbol
    """
    if df is None or len(df) == 0:
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot price
    ax1.plot(df.index, df['close'], label='Close')
    ax1.set_title(f"{symbol} Stock Price")
    ax1.set_ylabel("Price")
    ax1.grid(True)
    ax1.legend()
    
    # Plot volume if available
    if 'volume' in df.columns and df['volume'].sum() > 0:
        ax2.bar(df.index, df['volume'], label='Volume')
        ax2.set_title(f"{symbol} Trading Volume")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volume")
        ax2.grid(True)
        ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("data_cache/plots", exist_ok=True)
    plt.savefig(f"data_cache/plots/{symbol}_plot.png")
    
    plt.close()

def process_symbols(schema_info, symbols, batch_size=10, force=False, plot=False):
    """
    Process a list of symbols.
    
    Args:
        schema_info: Dictionary with schema information
        symbols: List of symbols to process
        batch_size: Number of symbols to process in each batch
        force: Whether to force reprocessing even if file exists
        plot: Whether to generate plots
        
    Returns:
        Dictionary with results
    """
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {
        'success': 0,
        'error': 0,
        'skipped': 0,
        'total': len(symbols),
    }
    
    # Process in batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} ({len(batch)} symbols)")
        
        for j, symbol in enumerate(batch):
            # Check if output file already exists
            output_path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
            if os.path.exists(output_path) and not force:
                print(f"[{j+1}/{len(batch)}] {symbol}: Already processed - skipping")
                results['skipped'] += 1
                continue
            
            print(f"[{j+1}/{len(batch)}] Processing {symbol}...")
            
            # Extract data
            df = extract_price_data(schema_info, symbol)
            
            if df is None or len(df) == 0:
                print(f"  Error: No data found for {symbol}")
                results['error'] += 1
                continue
            
            print(f"  Retrieved {len(df)} rows from {df.index[0]} to {df.index[-1]}")
            
            # Preprocess data
            processed_df = preprocess_data(df)
            
            if processed_df is None:
                print(f"  Error: Failed to preprocess data for {symbol}")
                results['error'] += 1
                continue
            
            # Export data
            output_path = export_data(processed_df, symbol)
            
            if output_path is None:
                print(f"  Error: Failed to export data for {symbol}")
                results['error'] += 1
                continue
            
            print(f"  Exported data to {output_path}")
            results['success'] += 1
            
            # Generate plot if requested
            if plot:
                plot_data(processed_df, symbol)
                print(f"  Generated plot for {symbol}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Process Alpaca Data for Training")
    parser.add_argument("--analyze", action="store_true", help="Analyze database schema")
    parser.add_argument("--symbol", type=str, help="Process a specific symbol")
    parser.add_argument("--all", action="store_true", help="Process all symbols")
    parser.add_argument("--top", type=int, default=100, help="Process top N symbols by data count")
    parser.add_argument("--batch", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if file exists")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    
    args = parser.parse_args()
    
    print(f"Alpaca Data Processor")
    print(f"Database: {DB_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Detect schema
    print("\nDetecting database schema...")
    schema_info = detect_price_table()
    
    if schema_info is None:
        print("Could not detect price table in the database.")
        if args.analyze:
            print("\nFalling back to schema analysis...")
            analyze_database_schema()
        return
    
    print(f"Found price table: {schema_info['price_table']}")
    print(f"Symbol column: {schema_info['symbol_column']}")
    print(f"Datetime column: {schema_info['datetime_column']}")
    print(f"Column mapping: {schema_info['column_mapping']}")
    
    # Get available symbols
    symbols = get_available_symbols(schema_info)
    print(f"\nFound {len(symbols)} symbols in the database")
    
    if len(symbols) == 0:
        print("No symbols found in the database.")
        return
    
    # Get date range
    date_range = get_date_range(schema_info)
    if date_range:
        print(f"Date range: {date_range[0]} to {date_range[1]}")
    
    # Show counts per symbol
    print("\nData counts per symbol:")
    counts_df = count_rows_per_symbol(schema_info, limit=20)
    if counts_df is not None:
        print(counts_df)
    
    # Process data based on arguments
    if args.symbol:
        if args.symbol not in symbols:
            print(f"Symbol {args.symbol} not found in the database.")
            return
        
        print(f"\nProcessing symbol: {args.symbol}")
        df = extract_price_data(schema_info, args.symbol)
        
        if df is None:
            print(f"No data found for {args.symbol}")
            return
        
        print(f"Retrieved {len(df)} rows")
        
        processed_df = preprocess_data(df)
        if processed_df is None:
            print(f"Failed to preprocess data for {args.symbol}")
            return
        
        output_path = export_data(processed_df, args.symbol)
        if output_path:
            print(f"Exported data to {output_path}")
        
        if args.plot:
            plot_data(processed_df, args.symbol)
            print(f"Generated plot for {args.symbol}")
    
    elif args.all or args.top:
        # Process all symbols or top N symbols
        if args.top and args.top < len(symbols):
            # Get top N symbols by data count
            counts_df = count_rows_per_symbol(schema_info, limit=args.top)
            if counts_df is not None:
                symbols_to_process = counts_df['symbol'].tolist()
                print(f"\nProcessing top {len(symbols_to_process)} symbols by data count")
            else:
                # Fallback to first N symbols
                symbols_to_process = symbols[:args.top]
                print(f"\nProcessing first {len(symbols_to_process)} symbols")
        else:
            symbols_to_process = symbols
            print(f"\nProcessing all {len(symbols_to_process)} symbols")
        
        # Process symbols
        start_time = time.time()
        results = process_symbols(
            schema_info,
            symbols_to_process,
            batch_size=args.batch,
            force=args.force,
            plot=args.plot
        )
        end_time = time.time()
        
        # Print results
        print("\nProcessing completed!")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Success: {results['success']}/{results['total']}")
        print(f"Errors: {results['error']}/{results['total']}")
        print(f"Skipped: {results['skipped']}/{results['total']}")
        
        # Create report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'database': str(DB_PATH),
            'output_dir': str(OUTPUT_DIR),
            'total_symbols': len(symbols),
            'processed_symbols': len(symbols_to_process),
            'success': results['success'],
            'error': results['error'],
            'skipped': results['skipped'],
            'time_seconds': end_time - start_time
        }
        
        # Save report
        os.makedirs("data_cache/reports", exist_ok=True)
        report_path = f"data_cache/reports/processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {report_path}")
    
    elif args.analyze:
        # Just analyze the schema
        print("\nPerforming detailed schema analysis...")
        analyze_database_schema()

if __name__ == "__main__":
    main()