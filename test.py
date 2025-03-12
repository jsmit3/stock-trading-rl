import pandas as pd
import os
from pathlib import Path

# Create necessary directories
Path('data_cache').mkdir(exist_ok=True)

# Fetch S&P 500 companies from Wikipedia
print('Downloading S&P 500 list from Wikipedia...')
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(url)
df = tables[0]

# Print column names for debugging
print("Columns in the table:", df.columns.tolist())

# Just take the first 100 companies (typically ordered by importance)
top_100 = df['Symbol'].head(100).str.replace('.', '-').tolist()

# Save to file
with open('data_cache/top_100_sp500.txt', 'w') as f:
    f.write('\n'.join(top_100))

print(f'Saved top 100 S&P 500 companies to data_cache/top_100_sp500.txt')
print(f'First 10 companies: {", ".join(top_100[:10])}')