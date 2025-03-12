import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import datetime
import numpy as np

# Path to the SQLite database
DB_PATH = Path("data_cache/alpaca_data.db")

def analyze_database_schema():
    """Analyze the structure of the database"""
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return
    
    print(f"Analyzing database at {DB_PATH}")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"Found {len(tables)} tables in the database:")
    
    # Analyze each table
    for table in tables:
        table_name = table[0]
        print(f"\n{'='*20} TABLE: {table_name} {'='*20}")
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        print(f"Columns ({len(columns)}):")
        for col in columns:
            col_id, name, dtype, notnull, default_val, pk = col
            pk_str = "PRIMARY KEY" if pk else ""
            null_str = "NOT NULL" if notnull else ""
            print(f"  - {name} ({dtype}) {null_str} {pk_str}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"Row count: {row_count}")
        
        # Sample data (first 5 rows)
        if row_count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            sample_data = cursor.fetchall()
            
            print("Sample data (first 5 rows):")
            for row in sample_data:
                print(f"  {row}")
        
        # Get unique values for some key columns (if not too many)
        if row_count > 0:
            for col_info in columns:
                col_id, name, dtype, notnull, default_val, pk = col_info
                
                # Check if this might be a categorical column with few unique values
                if dtype in ['TEXT', 'VARCHAR'] and name.lower() in ['symbol', 'ticker', 'type', 'status', 'frequency']:
                    cursor.execute(f"SELECT COUNT(DISTINCT {name}) FROM {table_name}")
                    unique_count = cursor.fetchone()[0]
                    
                    if unique_count < 100:  # Only show if not too many
                        cursor.execute(f"SELECT DISTINCT {name} FROM {table_name}")
                        unique_values = cursor.fetchall()
                        print(f"Unique values for {name} ({unique_count}):")
                        print(f"  {[val[0] for val in unique_values]}")
            
            # Check for date/time columns and get the min/max values
            for col_info in columns:
                col_id, name, dtype, notnull, default_val, pk = col_info
                
                if dtype in ['TIMESTAMP', 'DATETIME', 'DATE', 'INTEGER'] and name.lower() in ['timestamp', 'time', 'date', 'datetime']:
                    cursor.execute(f"SELECT MIN({name}), MAX({name}) FROM {table_name}")
                    min_val, max_val = cursor.fetchone()
                    print(f"Time range for {name}: {min_val} to {max_val}")
    
    # Close connection
    conn.close()

def extract_price_data(symbol, start_date=None, end_date=None):
    """
    Extract price data for a specific symbol from the database.
    
    Args:
        symbol: The ticker symbol to extract data for
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with OHLCV data
    """
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Initialize SQL query - will be adjusted based on database schema
    # This is a placeholder that will be updated after analyzing the schema
    sql_query = """
    SELECT * FROM price_data 
    WHERE symbol = ? 
    """
    
    params = [symbol]
    
    if start_date:
        sql_query += " AND timestamp >= ?"
        params.append(start_date)
    
    if end_date:
        sql_query += " AND timestamp <= ?"
        params.append(end_date)
    
    sql_query += " ORDER BY timestamp"
    
    try:
        # This will be adjusted based on actual schema
        df = pd.read_sql_query(sql_query, conn, params=params)
        
        # Close connection
        conn.close()
        
        if len(df) == 0:
            print(f"No data found for symbol {symbol}")
            return None
        
        print(f"Retrieved {len(df)} rows for {symbol}")
        return df
    
    except Exception as e:
        print(f"Error extracting data: {e}")
        conn.close()
        return None

def get_available_symbols():
    """Get list of available symbols in the database"""
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return []
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # This will need to be adjusted based on actual schema
    try:
        cursor.execute("SELECT DISTINCT symbol FROM price_data")
        symbols = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return symbols
    except Exception as e:
        print(f"Error getting symbols: {e}")
        conn.close()
        
        # Fallback: try to find a column that might be the symbol
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(price_data)")
            columns = cursor.fetchall()
            
            # Look for likely symbol column names
            symbol_column = None
            for col in columns:
                col_id, name, dtype, notnull, default_val, pk = col
                if name.lower() in ['symbol', 'ticker', 'stock', 'security']:
                    symbol_column = name
                    break
            
            if symbol_column:
                cursor.execute(f"SELECT DISTINCT {symbol_column} FROM price_data")
                symbols = [row[0] for row in cursor.fetchall()]
                conn.close()
                return symbols
        except:
            pass
            
        return []

def adjust_query_for_schema():
    """Analyze schema and adjust the SQL query for the actual database structure"""
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None, None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    price_table = None
    symbol_column = None
    datetime_column = None
    column_mapping = {}
    
    # Look for a table containing price data
    for table in tables:
        table_name = table[0]
        
        # Check if this table has price-related columns
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        column_names = [col[1].lower() for col in columns]
        
        # Check if this looks like a price table
        price_columns = ['open', 'high', 'low', 'close']
        if all(col in column_names for col in price_columns):
            price_table = table_name
            
            # Find the symbol column
            for col in columns:
                col_id, name, dtype, notnull, default_val, pk = col
                name_lower = name.lower()
                
                # Look for symbol column
                if name_lower in ['symbol', 'ticker', 'stock', 'security']:
                    symbol_column = name
                
                # Look for datetime column
                if name_lower in ['timestamp', 'time', 'date', 'datetime']:
                    datetime_column = name
                
                # Map column names to our expected format
                if name_lower in ['open', 'high', 'low', 'close', 'volume']:
                    column_mapping[name_lower] = name
            
            # If we found a price table, break
            if price_table and symbol_column and datetime_column:
                break
    
    conn.close()
    
    if price_table and symbol_column and datetime_column:
        # Build SQL query based on actual schema
        sql_query = f"""
        SELECT 
            {datetime_column} as timestamp,
            {symbol_column} as symbol,
        """
        
        # Add price columns
        for expected, actual in column_mapping.items():
            sql_query += f"{actual} as {expected}, "
        
        # Remove trailing comma and space
        sql_query = sql_query.rstrip(', ')
        
        sql_query += f"""
        FROM {price_table}
        WHERE {symbol_column} = ?
        """
        
        # Return query template and table info
        return {
            'query': sql_query,
            'price_table': price_table,
            'symbol_column': symbol_column,
            'datetime_column': datetime_column,
            'column_mapping': column_mapping
        }
    
    return None

def extract_data_with_schema(symbol, start_date=None, end_date=None):
    """
    Extract price data using detected schema.
    
    Args:
        symbol: The ticker symbol to extract data for
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with OHLCV data
    """
    schema_info = adjust_query_for_schema()
    
    if not schema_info:
        print("Could not determine database schema")
        return None
    
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Start with base query
    sql_query = schema_info['query']
    params = [symbol]
    
    # Add date filters if provided
    if start_date:
        sql_query += f" AND {schema_info['datetime_column']} >= ?"
        params.append(start_date)
    
    if end_date:
        sql_query += f" AND {schema_info['datetime_column']} <= ?"
        params.append(end_date)
    
    sql_query += f" ORDER BY {schema_info['datetime_column']}"
    
    try:
        # Read data
        df = pd.read_sql_query(sql_query, conn, params=params)
        
        # Close connection
        conn.close()
        
        if len(df) == 0:
            print(f"No data found for symbol {symbol}")
            return None
        
        print(f"Retrieved {len(df)} rows for {symbol}")
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                pass
        
        # Set timestamp as index if it exists
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error extracting data: {e}")
        conn.close()
        return None

def get_available_symbols_with_schema():
    """Get list of available symbols using the detected schema"""
    schema_info = adjust_query_for_schema()
    
    if not schema_info:
        print("Could not determine database schema")
        return []
    
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return []
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"SELECT DISTINCT {schema_info['symbol_column']} FROM {schema_info['price_table']}")
        symbols = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return symbols
    except Exception as e:
        print(f"Error getting symbols: {e}")
        conn.close()
        return []

def plot_stock_data(df, symbol):
    """Plot stock price data"""
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot price
    ax1.plot(df.index, df['close'])
    ax1.set_title(f"{symbol} Stock Price")
    ax1.set_ylabel("Price")
    ax1.grid(True)
    
    # Plot volume if available
    if 'volume' in df.columns:
        ax2.bar(df.index, df['volume'])
        ax2.set_title(f"{symbol} Trading Volume")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volume")
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def prepare_for_training(df):
    """
    Prepare data for training with the RL environment.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame ready for use with the RL environment
    """
    if df is None or len(df) == 0:
        print("No data to prepare")
        return None
    
    # Make a copy of the dataframe
    train_df = df.copy()
    
    # Ensure we have all required columns (open, high, low, close, volume)
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None
    
    # Add volume column if missing (with zeros)
    if 'volume' not in train_df.columns:
        train_df['volume'] = 0
    
    # Ensure all columns are numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    
    # Check for NaN values
    if train_df.isnull().any().any():
        print("Warning: NaN values found - filling with forward and backward fill")
        train_df = train_df.fillna(method='ffill').fillna(method='bfill')
    
    # Check for negative prices
    for col in ['open', 'high', 'low', 'close']:
        if (train_df[col] <= 0).any():
            print(f"Warning: Non-positive values found in {col} column")
            # Replace with a small positive value
            train_df.loc[train_df[col] <= 0, col] = 0.01
    
    # Ensure high >= low
    mask = train_df['high'] < train_df['low']
    if mask.any():
        print(f"Warning: {mask.sum()} rows have high < low - fixing")
        # Swap high and low where necessary
        temp = train_df.loc[mask, 'high'].copy()
        train_df.loc[mask, 'high'] = train_df.loc[mask, 'low']
        train_df.loc[mask, 'low'] = temp
    
    # Ensure high >= open and high >= close
    train_df['high'] = train_df[['high', 'open', 'close']].max(axis=1)
    
    # Ensure low <= open and low <= close
    train_df['low'] = train_df[['low', 'open', 'close']].min(axis=1)
    
    return train_df

def save_to_csv(df, symbol, output_dir="data_cache/stocks"):
    """
    Save the preprocessed data to a CSV file.
    
    Args:
        df: DataFrame with processed OHLCV data
        symbol: Ticker symbol
        output_dir: Output directory
        
    Returns:
        Path to the saved file
    """
    if df is None or len(df) == 0:
        print("No data to save")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    filename = os.path.join(output_dir, f"{symbol}_daily.csv")
    
    # Save to CSV
    df.to_csv(filename)
    print(f"Saved to {filename}")
    
    return filename

def main():
    """Main function to analyze the database and extract sample data"""
    # Analyze database schema
    analyze_database_schema()
    
    # Get available symbols
    print("\n\nGetting available symbols...")
    symbols = get_available_symbols_with_schema()
    
    if symbols:
        print(f"Found {len(symbols)} symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        
        # Extract and process data for a sample symbol
        if len(symbols) > 0:
            sample_symbol = symbols[0]
            print(f"\n\nExtracting data for sample symbol: {sample_symbol}")
            
            df = extract_data_with_schema(sample_symbol)
            
            if df is not None:
                print("\nSample data:")
                print(df.head())
                
                print("\nData statistics:")
                print(df.describe())
                
                print("\nData info:")
                print(df.info())
                
                print("\nPreparing data for training...")
                train_df = prepare_for_training(df)
                
                if train_df is not None:
                    print("\nSaving processed data...")
                    save_to_csv(train_df, sample_symbol)
                    
                    print("\nPlotting sample data...")
                    plot_stock_data(train_df, sample_symbol)

if __name__ == "__main__":
    main()