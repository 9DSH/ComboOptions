from datetime import datetime, timedelta
import pandas as pd
import asyncio
import aiohttp
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DeribitAPI:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://www.deribit.com/api/v2"
        self.session = requests.Session()

    async def fetch_public_trades_async(self, session, instrument_name, start_timestamp, end_timestamp):
        """Fetch public trade data asynchronously for a given instrument."""
        url = "https://www.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time"
        params = {
            "instrument_name": instrument_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "count": 1000
        }

        for attempt in range(5):  # Retry up to 5 times
            async with session.get(url, params=params) as response:
                if response.status == 429:  # Too Many Requests
                    wait_time = 16  # fixed wait time for rate limit reached
                    await asyncio.sleep(wait_time)  # Wait before retrying
                    continue  # Retry the request
                response.raise_for_status()
                data = await response.json()
                return data

        logging.error(f"Failed to fetch trades for {instrument_name} after multiple attempts.")
        return None  # Return None if all attempts fail

    def get_all_instruments(self, currency='BTC'):
        """Fetch all available options for a given currency."""
        url = f"{self.base_url}/public/get_instruments"
        params = {'currency': currency, 'kind': 'option', 'expired': 'false'}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        instruments = response.json().get('result', [])
        return [instrument['instrument_name'] for instrument in instruments]

async def fetch_all_trades(deribit_api, instrument_names, start_timestamp, end_timestamp):
    """Fetch all trades for a list of instrument names."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            deribit_api.fetch_public_trades_async(session, name, start_timestamp, end_timestamp)
            for name in instrument_names
        ]
        results = await asyncio.gather(*tasks)
        all_trades = []
        for result in results:
            if result and 'result' in result and 'trades' in result['result']:
                all_trades.extend(result['result']['trades'])
        df = pd.DataFrame(all_trades)
        if not df.empty:
            # Convert the 'timestamp' column from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

# Example usage
CLIENT_ID = 'i_kH-t6n'
CLIENT_SECRET = 'sHrn7n_7RE4vVEzhMkm3n4S8giSl5gK9L9qLcXFtDTk'
deribit_api = DeribitAPI(CLIENT_ID, CLIENT_SECRET)

# Fetch all instruments
instrument_names = deribit_api.get_all_instruments()

# Determine the earliest available date
earliest_available_date = datetime.utcnow().date() - timedelta(days=365)  # Adjust as needed
start_datetime = datetime.combine(earliest_available_date, datetime.min.time())
end_datetime = datetime.combine(datetime.utcnow().date(), datetime.min.time()).replace(hour=23, minute=59)

start_timestamp = int(start_datetime.timestamp() * 1000)  # Convert to milliseconds
end_timestamp = int(end_datetime.timestamp() * 1000)  # Convert to milliseconds

# Fetch all trades for all instruments
df_trades = asyncio.run(fetch_all_trades(deribit_api, instrument_names, start_timestamp, end_timestamp))
print(df_trades)