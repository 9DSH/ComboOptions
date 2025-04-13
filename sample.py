from Fetch_data import Fetching_data
import pandas as pd

fetch_data = Fetching_data()

fetch_data.load_from_csv(data_type="options_screener")
df = fetch_data.options_screener

# Drop rows with duplicate Trade IDs, keeping only the first occurrence
df = df.drop_duplicates(subset=['Trade ID'])

combined_df = pd.concat([df , df ])
print(combined_df .shape)
        # Drop duplicates based on specified columns
public_trades_total = combined_df.drop_duplicates(subset=['Trade ID'])

print(public_trades_total .shape)

