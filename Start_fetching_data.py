from Deribit import DeribitAPI
import streamlit as st
from datetime import date, datetime , timedelta
import time

# Example usage
CLIENT_ID = 'B2uxTcMr'
CLIENT_SECRET = '4DFWS6LcbQPBGU6xglFytGI0Bu8or0kRh-a8C-IjtGk'
deribit_api = DeribitAPI(CLIENT_ID, CLIENT_SECRET)



def start_fetching_data_from_api():
    while True:
        try:

            # Clear the price cache before fetching new data
            deribit_api.clear_price_cache()
            
            current_utc_date = datetime.utcnow().date() 
            start_date = current_utc_date - timedelta(days=7)
            end_date = current_utc_date  

            # Convert to datetime for timestamps
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.min.time()).replace(hour=23, minute=59)  # End of the day

            # Start fetching data for the specified date range
            deribit_api.execute_data_fetch(currency='BTC', start_date=start_datetime, end_date=end_datetime)

            # Optional: Sleep for a specified amount of time before the next execution
            time.sleep(10)  # Wait for 10 seconds before fetching again

        except KeyboardInterrupt:
            print("Process interrupted by the user.")
            break  # Exit the loop if the user interrupts

        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally, you could implement a wait time before retrying on failure
            time.sleep(30)  # Wait for 30 seconds before trying again in case of an error

def get_btcusd_price():
    price = deribit_api.fetch_btc_to_usd()
    highest_price , lowest_price = deribit_api.fetch_today_high_low()
    return price , highest_price, lowest_price
