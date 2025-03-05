from Calculations import get_most_traded_instruments
from Fetch_data import Fetching_data

fetch_data = Fetching_data()

market_screener_df  = fetch_data.load_market_trades()
most_traded,  top_options_chains= get_most_traded_instruments(market_screener_df)

print(top_options_chains )
print(most_traded )