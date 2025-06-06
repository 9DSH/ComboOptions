import requests
import logging
from datetime import datetime, date, timedelta
import pandas as pd
import asyncio
import aiohttp
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.utils import resample
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.utils.validation')
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DeribitAPI:
    def __init__(self, 
                 client_id: str, 
                 client_secret: str, 
                 options_data_csv: str = "options_data.csv", 
                 options_screener_csv: str = "options_screener.csv",
                 public_trades_24h_csv: str = "public_trades_24h.csv"):
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.btc_usd_price = None
        self.session = requests.Session()
        self.options_data_csv = options_data_csv  # Path for CSV storage for options data
        self.options_screener_csv = options_screener_csv  # Path for CSV storage for options screener
        self.public_trades_24h_csv =  public_trades_24h_csv
        
        # Initialize empty DataFrames for options data and screener
        self.options_data = pd.DataFrame()
        self.options_screener = pd.DataFrame()        
        self.public_trades_24h = pd.DataFrame()

    def authenticate(self):
        if self.access_token:
            return self.access_token
        
        auth_url = 'https://www.deribit.com/api/v2/public/auth'
        auth_params = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'client_credentials'
        }
        
        try:
            response = self.session.get(auth_url, params=auth_params)
            response.raise_for_status()
            auth_data = response.json()
            self.access_token = auth_data.get('result', {}).get('access_token')
            if not self.access_token:
                raise ValueError("Authentication failed: Invalid credentials or response format")
            return self.access_token
        except requests.RequestException as e:
            logging.error(f"Authentication error: {e}")
            return None
        
    def fetch_btc_to_usd(self):
        """Fetch current BTC to USD conversion rate.""" 
        
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            self.btc_usd_price = data.get('bitcoin', {}).get('usd', 0)
            return self.btc_usd_price
        except requests.RequestException as e:
            logging.error(f"Error fetching BTC price: {e}")
            return 0
    
    def fetch_order_book(self, option_symbol):
        """Fetch the order book for a specific option."""
        order_book_url = 'https://www.deribit.com/api/v2/public/get_order_book'
        params = {'instrument_name': option_symbol}
        try:
            response = self.session.get(order_book_url, params=params)
            response.raise_for_status()
            return response.json().get('result', {})
        except requests.RequestException as e:
            logging.error(f"Failed to fetch order book for {option_symbol}: {e}")
            return {}

    def save_to_csv(self, df, data_type):
        """Save options data or options screener data to a CSV file based on data_type."""
        if data_type == "options_screener":
            # Remove duplicates based on 'Trade ID' column
            if 'Trade ID' in df.columns:
                df = df.drop_duplicates(subset=['Trade ID'], keep='first')
        
        if data_type == "options_data":  # Save options data
            df.to_csv(self.options_data_csv, index=False)
            logging.info(f"Saved options data to {self.options_data_csv}")
        elif data_type == "options_screener":  # Save options screener data
            df.to_csv(self.options_screener_csv, index=False)
            logging.info(f"Saved options screener data to {self.options_screener_csv}")
        elif data_type == "public_trades_24h":  # Save public trades data
            df.to_csv(self.public_trades_24h_csv, index=False)
            logging.info(f"Saved public_trades_24h options screener data to {self.public_trades_24h_csv}")

    def refresh_options_data(self, currency='BTC'):
        """Refresh options data for the given currency."""
        logging.info(f"Refreshing options data for {currency}")

        access_token = self.authenticate()
        if not access_token:
            logging.error("Failed to authenticate.")
            return pd.DataFrame()
        
        btc_to_usd = self.fetch_btc_to_usd()

        options_data = []
        option_chains_url = 'https://www.deribit.com/api/v2/public/get_instruments'
        params = {'currency': currency, 'kind': 'option', 'expired': 'false'}
        headers = {'Authorization': f'Bearer {access_token}'}

        try:
            response = self.session.get(option_chains_url, params=params, headers=headers)
            response.raise_for_status()
            option_chains = response.json().get('result', [])
            logging.info(f"Fetched {len(option_chains)} option chains.")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch option chains for {currency}: {e}")
            return pd.DataFrame()

        for instrument in option_chains:
            option_symbol = instrument['instrument_name']
            order_book = self.fetch_order_book(option_symbol)

            last_price_btc = order_book.get('last_price', 0) or 0
            bid_price_btc = order_book.get('best_bid_price', 0) or 0
            ask_price_btc = order_book.get('best_ask_price', 0) or 0
            bid_iv = order_book.get('bid_iv', 0) or 0.8  # Default IV if not available
            ask_iv = order_book.get('ask_iv', 0) or 0.8  # Default IV if not available

            delta = order_book.get('greeks', {}).get('delta')
            gamma = order_book.get('greeks', {}).get('gamma')
            theta = order_book.get('greeks', {}).get('theta')
            vega = order_book.get('greeks', {}).get('vega')

            volume = order_book.get('stats', {}).get('volume')
            volume_usd = order_book.get('stats', {}).get('volume_usd')
            open_interest = order_book.get('open_interest')

            # Check if bid or ask price is missing before adding to options_data
            if bid_price_btc <= 0 and ask_price_btc <= 0:
                continue

            # Construct the option details
            options_data.append({
                'Instrument': option_symbol,
                'Option Type': instrument['option_type'],
                'Strike Price': instrument.get('strike', 0),
                'Expiration Date': datetime.utcfromtimestamp(instrument['expiration_timestamp'] / 1000).date(),
                'Last Price (USD)': last_price_btc * btc_to_usd,
                'Bid Price (USD)': bid_price_btc * btc_to_usd,
                'Ask Price (USD)': ask_price_btc * btc_to_usd,
                'Bid IV': bid_iv ,
                'Ask IV': ask_iv ,
                'Delta': delta,
                'Gamma': gamma,
                'Theta': theta,
                'Vega': vega,
                'open_interest': open_interest,
                'total traded volume': volume,
                'monetary volume': volume_usd
            })

        if options_data:
            self.options_data = pd.DataFrame(options_data)
            option_df_copy = self.options_data.copy()
            probabilities_df = self.option_probabilities_with_greeks(option_df_copy)
            logging.info(f"Exercise Probabilities calculated for {len(probabilities_df)} options.")
            # Merge only the Probability (%) column
            self.options_data = self.options_data.merge(probabilities_df[['Instrument', 'Exercise_Probability (%)']], on='Instrument', how='left')

            # Rename the column for clarity
            self.options_data.rename(columns={'Exercise_Probability (%)': 'Probability (%)'}, inplace=True)

            self.save_to_csv(self.options_data, data_type="options_data")
        else:
            logging.warning("No options data fetched.")

        return self.options_data
    

    def option_probabilities_with_greeks(self, df):
        df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])
        df['Option Type'] = LabelEncoder().fit_transform(df['Option Type'])
        df['Time to Expiration'] = (df['Expiration Date'] - pd.to_datetime('today')).dt.days

        delta_threshold = 0.5
        df['Exercise'] = (df['Delta'] > delta_threshold).astype(int)

        features = ['Option Type', 'Strike Price', 'Bid IV', 'Ask IV', 'Gamma', 'Theta', 'Vega', 'Time to Expiration']
        target = 'Exercise'
        
        X = df[features]
        y = df[target]

        df_majority = df[df.Exercise == 0]
        df_minority = df[df.Exercise == 1]
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
        df_balanced = pd.concat([df_majority, df_minority_upsampled])

        X_balanced = df_balanced[features]
        y_balanced = df_balanced[target]

        X_train, X_valid, y_train, y_valid = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        # Convert to DataFrame to maintain feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
        X_valid_scaled = pd.DataFrame(X_valid_scaled, columns=features)

        xgb_search = BayesSearchCV(
            xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, verbosity=0),
            {'n_estimators': (50, 200), 'learning_rate': (0.01, 0.1), 'max_depth': (3, 10)},
            n_iter=20,
            cv=3,
            scoring='accuracy'
        )
        xgb_search.fit(X_train_scaled, y_train)
        best_xgb_model = xgb_search.best_estimator_

        lgb_model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', verbose=-1)
        lgb_search = BayesSearchCV(
            lgb_model,
            {'num_leaves': (31, 100), 'learning_rate': (0.01, 0.1), 'n_estimators': (50, 200)},
            n_iter=20,
            cv=3,
            scoring='accuracy'
        )

        lgb_search.fit(X_train_scaled, y_train)
        best_lgb_model = lgb_search.best_estimator_

        voting_model = VotingClassifier(estimators=[('xgb', best_xgb_model), ('lgbm', best_lgb_model)], voting='soft')
        voting_model.fit(X_train_scaled, y_train)

        y_proba = voting_model.predict_proba(X_valid_scaled)[:, 1]

        results_df = df.iloc[y_valid.index].copy()
        results_df['Exercise_Probability (%)'] = y_proba * 100

        # Modify the Exercise Probability
        results_df['Exercise_Probability (%)'] = results_df['Exercise_Probability (%)'].apply(lambda x: int(round((x - int(x)) * 1000)) / 10)

        return results_df[['Instrument', 'Exercise_Probability (%)']]
    

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
    
    def extract_option_details(self, option_symbol: str):
        match = re.search(r'-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-(C|P)$', option_symbol)
        if not match:
            return None, None, None  # Ensure it returns a tuple with None values if no match

        day, month, year, strike_price, option_type = match.groups()
        expiration_date = f"{day}-{month.capitalize()}-{year}"
        strike_price = int(strike_price)
        option_type = "Call" if option_type == "C" else "Put"

        return expiration_date, strike_price, option_type  # Return as a tuple

    async def fetch_all_public_trades_async(self, instrument_names, start_timestamp, end_timestamp):
        """Fetch public trades for all instruments concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for instrument_name in instrument_names:
                tasks.append(self.fetch_public_trades_async(session, instrument_name, start_timestamp, end_timestamp))
            results = await asyncio.gather(*tasks)

            combined_trades = []
            for result in results:
                if result is None:  # Skip processing if result is None
                    continue
                if 'result' in result and 'trades' in result['result']:
                    combined_trades.extend(result['result']['trades'])
                else:
                    logging.error(f"Unexpected response format for instrument: {instrument_name}. Result: {result}")

            return combined_trades
        
    def validate_screener_data(self, df, required_columns):
        """Check if each row in the DataFrame has values in the required columns, otherwise skip the row."""
        # Check for missing columns in the DataFrame
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns in the DataFrame: {missing_columns}")
            return False

        # Check for missing or malformed data in each required column
        for column in required_columns:
            if df[column].isnull().any():
                logging.error(f"Missing data in column: {column}")
                return False

        # Additional checks for specific columns can be added here
        # For example, ensure numeric columns have valid numbers
        numeric_columns = ['iv', 'price', 'index_price', 'amount']
        for column in numeric_columns:
            if column in df.columns and not pd.api.types.is_numeric_dtype(df[column]):
                logging.error(f"Non-numeric data found in numeric column: {column}")
                return False

        return True
    
    def validate_df_mismatch(self, df):
        print("Starting validate_df_mismatch...")
        expected_fields = 18

        # Check for valid rows by comparing the number of columns
        valid_rows = df[df.apply(lambda row: len(row) == expected_fields, axis=1)].reset_index(drop=True)

        print(f"Number of valid rows: {valid_rows.shape[0]} out of {df.shape[0]}")
        print("validate_df_mismatch complete.")
        return valid_rows
        
    def process_screener_data(self, public_trades_df):
        """Process the public trades DataFrame by performing calculations, renaming columns, and saving it.
        raw_columns = [
                            'timestamp', 'iv', 'price', 'direction', 'index_price', 'instrument_name', 
                            'trade_seq', 'mark_price', 'amount', 'tick_direction', 'contracts', 'trade_id', 
                            'block_trade_id', 'block_rfq_id', 'combo_id', 'block_trade_leg_count', 'liquidation', 'combo_trade_id'
                    ]
        """

        logging.info("Processing public trades data...")

        required_columns = ['timestamp', 'iv', 'price', 'direction', 'index_price', 'instrument_name', 
                            'trade_seq', 'mark_price', 'amount', 'tick_direction', 'contracts', 'trade_id']

        if not self.validate_screener_data(public_trades_df, required_columns):
            return

        # Convert the 'timestamp' from trades to a readable date
        public_trades_df['timestamp'] = pd.to_datetime(public_trades_df['timestamp'], unit='ms')  # Convert ms to datetime
        
        # Format the timestamp: remove seconds and milliseconds
        public_trades_df['timestamp'] = public_trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        public_trades_df['direction'] = public_trades_df['direction'].str.upper()

        # Ensure the extraction function is available and retrieve details
        for index, row in public_trades_df.iterrows():
            try:
                expiration_date, strike_price, option_type = self.extract_option_details(row['instrument_name'])
                public_trades_df.at[index, 'Expiration Date'] = expiration_date
                public_trades_df.at[index, 'Strike Price'] = strike_price
                public_trades_df.at[index, 'Option Type'] = option_type
            except Exception as e:
                logging.error(f"Error extracting details for row {index}: {e}")
            # Optionally, you can choose to skip this row or fill with default values

        # Calculate additional columns
        public_trades_df['Price (USD)'] = (public_trades_df['price'] * public_trades_df['index_price']).round(2)
        public_trades_df['Entry Value'] = (public_trades_df['amount'] * public_trades_df['Price (USD)']).round(2)
        
        # Rename the columns as per the requirements
        columns = {
            'timestamp': 'Entry Date',
            'iv': 'IV (%)',
            'price': 'Price (BTC)',
            'direction': 'Side',
            'index_price': 'Underlying Price',
            'instrument_name': 'Instrument',
            'amount': 'Size',
            'mark_price': 'Mark Price (BTC)',
            'block_trade_id': 'BlockTrade IDs',
            'combo_id' : 'Combo ID', 
            'block_trade_leg_count' : 'BlockTrade Count', 
            'combo_trade_id': 'ComboTrade IDs',
            'liquidation' : 'Liquidation',
            'trade_id': 'Trade ID'
        }
        
        public_trades_df.rename(columns=columns, inplace=True)
        

        # Ensure correct order of columns
        new_order = [
            'Side', 'Instrument', 'Price (BTC)', 'Price (USD)', 'Mark Price (BTC)',
            'IV (%)', 'Size', 'Entry Value', 'Underlying Price',
            'Expiration Date', 'Strike Price', 'Option Type', 'Entry Date','BlockTrade IDs',
            'BlockTrade Count', 'Combo ID', 'ComboTrade IDs' , 'Trade ID'
        ]
        
        public_trades_df = public_trades_df[new_order]

        # Read existing data from CSV
        try:
            existing_df = pd.read_csv(self.options_screener_csv, on_bad_lines='skip') if os.path.exists(self.options_screener_csv) else pd.DataFrame(columns=new_order)
        except FileNotFoundError:
            existing_df = pd.DataFrame(columns=new_order)
        
        expired_trades_removed = self.remove_expired_trades(existing_df)
        # Concatenate the new data with the existing data
        combined_df = pd.concat([expired_trades_removed, public_trades_df])
        # Identify and remove all rows with duplicate Trade IDs (keep none)
        mask = combined_df.duplicated(subset=['Trade ID', 'Price (BTC)' , 'Underlying Price'], keep=False)
        public_trades_total = combined_df[~mask]

        processed_df = self.validate_df_mismatch(public_trades_total)
        # Save the processed DataFrame to CSV using the existing method
        self.save_to_csv(processed_df, data_type="options_screener")
        logging.info(f"Updated options screener data saved to {self.options_screener_csv}")

        self.save_to_csv(public_trades_df, data_type="public_trades_24h")
        logging.info("public_trades_24h Options data CSV saved.")

    def remove_expired_trades(self, data_df):
        """
        Remove rows from the DataFrame where the 'Expiration Date' is before the current date.
        If the current time is 8:00 UTC, also remove trades that expired yesterday.
        Print the number of expired trades removed and return the filtered DataFrame.
        The returned DataFrame will have the same format as the input.
        """
        df = data_df.copy()
        # Create a temporary parsed column for comparison
        parsed_col = 'Expiration Date Parsed'
        df[parsed_col] = pd.to_datetime(df['Expiration Date'], format='%d-%b-%y', errors='coerce')

        # Get the current date and time in UTC
        current_utc_datetime = datetime.utcnow()
        # Determine the cutoff date for expiration
        if current_utc_datetime.hour >= 8:
            cutoff_date = pd.to_datetime(current_utc_datetime.date())
            print("cutoff date expiry:" , cutoff_date)
        else:
            cutoff_date = pd.to_datetime((current_utc_datetime - timedelta(days=1)).date()).replace(hour=8, minute=0, second=0)
            print("cutoff date:" , cutoff_date)

        # Filter out expired trades using the parsed column
        expired_trades = df[df[parsed_col] < cutoff_date]
        num_expired_trades = expired_trades.shape[0]
        df = df[df[parsed_col] > cutoff_date]

        # Print the number of expired trades
        print(f"{num_expired_trades} trades are going to be expired.")

        # Drop the temporary column before returning
        df = df.drop(columns=[parsed_col])

        # Ensure the returned DataFrame has the same columns and order as the input
        df = df[data_df.columns]

        return df
    
    def execute_data_fetch(self, currency='BTC', start_date=None, end_date=None):
        """Fetch and save options and public trades data."""
        logging.info("Starting the data fetching process...")

        # Step 1: Authenticate and fetch options data
        options_data = self.refresh_options_data(currency)
        if options_data.empty:
            logging.warning("No options data fetched.")
            return

        # Step 2: Fetch public trades if start_date is provided
        if start_date and end_date:
            instrument_names = options_data['Instrument'].tolist()
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000) + 86400000  # End of the day
            
            asyncio.run(asyncio.sleep(5))  # Wait 5 seconds before starting

            # Run the async fetching
            public_trades = asyncio.run(self.fetch_all_public_trades_async(instrument_names, start_timestamp, end_timestamp))
            if public_trades:
                public_trades_df = pd.DataFrame(public_trades)
                self.process_screener_data(public_trades_df )  # Process and save the public trades data
            else:
                logging.warning("No public trades fetched.")


        logging.info("Data fetching process completed.")

    def clear_price_cache(self):
        """Clear the cached BTC price to force a fresh fetch"""
        self.btc_usd_price = None

    def fetch_today_high_low(self):
        # Using get_book_summary_by_currency which reliably returns high/low for BTC
        url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
        params = {
            'currency': 'BTC',
            'kind': 'future'  # For perpetual swaps and futures
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('result'):
                print("Error: No results in getting highest and lowest price response")
                return None, None
                
            # Find BTC-PERPETUAL in results
            for instrument in data['result']:
                if instrument['instrument_name'] == 'BTC-PERPETUAL':
                    highest_price = int(float(instrument['high']))
                    lowest_price = int(float(instrument['low']))
                    return highest_price, lowest_price
            
            return None, None
            
        except requests.exceptions.RequestException as e:
            print(f"API Request Failed: {e}")
            return None, None
        except (KeyError, ValueError) as e:
            print(f"Data Parsing Error: {e}")
            return None, None