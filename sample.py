from Fetch_data import Fetching_data
from datetime import datetime , timedelta

import plotly.graph_objects as go
import pandas as pd

fetch_data = Fetching_data()

fetch_data.load_from_csv(data_type="options_screener")
df = fetch_data.options_screener

def remove_expired_trades(df):
        """
        Remove rows from the DataFrame where the 'Expiration Date' is before the current date.
        If the current time is 8:00 UTC, also remove trades that expired yesterday.
        Print the number of expired trades removed and return the filtered DataFrame.
        """
        # Convert 'Expiration Date' to datetime if it's not already
        
        df['Expiration Date'] = pd.to_datetime(df['Expiration Date'], format='%d-%b-%y', errors='coerce')

        # Get the current date and time in UTC
        current_utc_datetime = datetime.utcnow() 
        print("currecnt date:" , current_utc_datetime)
        # Determine the cutoff date for expiration
        if current_utc_datetime.hour >= 8:
            # If it's 8:00 UTC or more, consider trades expired if they expired yesterday
            cutoff_date = pd.to_datetime((current_utc_datetime + timedelta(days=2)).date()).replace(hour=8, minute=0, second=0)
        else:
            # Otherwise, consider trades expired if they expired before today
            cutoff_date = pd.to_datetime(current_utc_datetime.date()).replace(hour=8, minute=0, second=0)
        # Filter out expired trades
        print(cutoff_date)
        expired_trades = df[df['Expiration Date'] < cutoff_date]
        num_expired_trades = expired_trades.shape[0]
        print(expired_trades)
        # Drop expired trades from the DataFrame
        df = df[df['Expiration Date'] >= cutoff_date]

        # Print the number of expired trades
        print(f"{num_expired_trades} trades are expired.")

        return df
# Drop rows with duplicate Trade IDs, keeping only the first occurrence
# Create a DataFrame with 'Price' and 'Expiration Date' columns
data = {
    'Price': [100 + i for i in range(20)],  # Example prices from 100 to 119
    'Expiration Date': [datetime.now() + timedelta(days=i) for i in range(20)]  # Expiration dates from today onwards
}

df_generated = pd.DataFrame(data)
print(df_generated.head(10))  # Print the first 10 rows of the generated DataFrame

copy_df = remove_expired_trades(df_generated )
print(copy_df.head(10))

