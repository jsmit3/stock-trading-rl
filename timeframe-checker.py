import sqlite3
from pathlib import Path

# Path to the SQLite database
DB_PATH = Path("data_cache/alpaca_data.db")

def check_timeframes():
    """Check what timeframes are available in the database"""
    if not DB_PATH.exists():
        print(f"Database file not found at {DB_PATH}")
        return None
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query for distinct timeframes
    try:
        cursor.execute("SELECT DISTINCT timeframe FROM historical_data")
        timeframes = [row[0] for row in cursor.fetchall()]
        
        print(f"Found {len(timeframes)} timeframes: {', '.join(timeframes)}")
        
        # Count rows per timeframe
        print("\nRow count by timeframe:")
        for tf in timeframes:
            cursor.execute("SELECT COUNT(*) FROM historical_data WHERE timeframe = ?", (tf,))
            count = cursor.fetchone()[0]
            print(f"  {tf}: {count:,} rows")
            
        # Check for daily timeframe specifically
        if '1Day' in timeframes or 'D' in timeframes or 'day' in timeframes or 'Day' in timeframes or 'daily' in timeframes or 'Daily' in timeframes:
            # Find the exact daily timeframe name
            daily_tf = [tf for tf in timeframes if tf.lower() in ['1day', 'd', 'day', 'daily']][0]
            print(f"\nFound daily timeframe: {daily_tf}")
            
            # Check how many symbols have daily data
            cursor.execute(f"SELECT DISTINCT symbol FROM historical_data WHERE timeframe = ?", (daily_tf,))
            daily_symbols = [row[0] for row in cursor.fetchall()]
            print(f"  {len(daily_symbols)} symbols have daily data")
            
            # Sample of symbols with daily data
            sample_limit = min(10, len(daily_symbols))
            print(f"  Sample symbols: {', '.join(daily_symbols[:sample_limit])}")
            
            # Check date range for daily data
            cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM historical_data WHERE timeframe = ?", (daily_tf,))
            min_date, max_date = cursor.fetchone()
            print(f"  Date range: {min_date} to {max_date}")
        
        conn.close()
        return timeframes
    
    except Exception as e:
        print(f"Error checking timeframes: {e}")
        conn.close()
        return None

if __name__ == "__main__":
    check_timeframes()