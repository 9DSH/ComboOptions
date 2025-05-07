from Fetch_data import Fetching_data
import pandas as pd

fetch =Fetching_data()
options_ = fetch.fetch_option_data()
raw_df = fetch.load_market_trades(filter=None, drop = None, show_24h_public_trades=False)
print(raw_df.shape)
def validate_and_convert_df(df, data_type="Public"):
    # Define the desired data types for each column based on data_type
    if data_type == "Public":
        desired_dtypes = {
            'Side': 'object',
            'Instrument': 'object',
            'Price (BTC)': 'float64',
            'Price (USD)': 'float64',
            'Mark Price (BTC)': 'float64',
            'IV (%)': 'float64',
            'Size': 'float64',
            'Entry Value': 'float64',
            'Underlying Price': 'float64',
            'Expiration Date': 'object',
            'Strike Price': 'float64',
            'Option Type': 'object',
            'Entry Date': 'datetime64[ns]',
            'BlockTrade IDs': 'object',
            'BlockTrade Count': 'float64',
            'Combo ID': 'object',
            'ComboTrade IDs': 'float64',
            'Trade ID': 'float64'
        }
    elif data_type == "Trade":
        desired_dtypes = {
            'Instrument': 'object',
            'Option Type': 'object',
            'Strike Price': 'float64',
            'Expiration Date': 'object',
            'Last Price (USD)': 'float64',
            'Bid Price (USD)': 'float64',
            'Ask Price (USD)': 'float64',
            'Bid IV': 'float64',
            'Ask IV': 'float64',
            'Delta': 'float64',
            'Gamma': 'float64',
            'Theta': 'float64',
            'Vega': 'float64',
            'open_interest': 'float64',
            'total traded volume': 'float64',
            'monetary volume': 'float64'
        }
    else:
        raise ValueError("Invalid data_type. Expected 'Public' or 'Trade'.")

    # Iterate over the columns and convert data types
    for column, dtype in desired_dtypes.items():
        if column in df.columns:
            # Convert the column to the desired data type
            try:
                df[column] = df[column].astype(dtype)
            except ValueError:
                print(f"Warning: Could not convert column {column} to {dtype}.")
        
        # Check for None or NaN values
        if df[column].isnull().any():
            print(f"Warning: Column {column} contains None or NaN values.")
    
    # Drop rows with incorrect data types
    for column, dtype in desired_dtypes.items():
        if column in df.columns:
            # Check each value in the column
            for index, value in df[column].items():
                try:
                    # Attempt to convert the value to the desired type
                    if dtype == 'object':
                        str(value)  # Ensure it can be converted to string
                    elif dtype == 'datetime64[ns]':
                        pd.to_datetime(value)  # Ensure it can be converted to datetime
                    else:
                        float(value)  # Ensure it can be converted to float
                except (ValueError, TypeError):
                    # Drop the row if conversion fails
                    df.drop(index, inplace=True)
                    print(f"Dropped row {index} due to invalid data type in column {column}.")
    
    return df

filter_df = validate_and_convert_df(raw_df)
print(filter_df.shape)
print(filter_df)