import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('NIFTY_part_1.csv')

print(f"Total rows: {len(df)}")
print(f"\nFirst few rows of date column:")
print(df['date'].head())

print(f"\nFirst few rows of time column:")
print(df['time'].head())

print(f"\nFirst few rows of spot column:")
print(df['spot'].head())

# Clean Excel formula format
df['date'] = df['date'].astype(str).str.replace(r'^=\"|\"$', '', regex=True)
df['time'] = df['time'].astype(str).str.replace(r'^=\"|\"$', '', regex=True)

print(f"\nAfter cleaning date:")
print(df['date'].head())

print(f"\nAfter cleaning time:")
print(df['time'].head())

# Try combining timestamp
df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%y %H:%M:%S')

print(f"\nTimestamp created successfully:")
print(df['timestamp'].head())

print(f"\nSpot price stats:")
print(f"Min: {df['spot'].min()}")
print(f"Max: {df['spot'].max()}")
print(f"Mean: {df['spot'].mean()}")

# Test strike calculation
strike_interval = 50
df['strike'] = np.round(df['spot'] / strike_interval) * strike_interval
df['strike'] = df['strike'].apply(lambda x: max(x, strike_interval))

print(f"\nStrike price stats:")
print(f"Min: {df['strike'].min()}")
print(f"Max: {df['strike'].max()}")
print(f"Any zeros: {(df['strike'] == 0).sum()}")

print("\n✅ CSV data looks good!")
