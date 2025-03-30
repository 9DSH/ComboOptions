from datetime import datetime, timedelta
import asyncio
import aiohttp
from Deribit import DeribitAPI
import json

# Use existing credentials from your configuration
CLIENT_ID = 'i_kH-t6n'
CLIENT_SECRET = 'sHrn7n_7RE4vVEzhMkm3n4S8giSl5gK9L9qLcXFtDTk'

async def test_fetch_trades():
    # Initialize the DeribitAPI with proper credentials
    api = DeribitAPI(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    
    # Use a specific instrument for current month/quarter
    instrument_name = "BTC-1APR25-84000-P" # Example of a near-term option
    print(f"Testing with instrument: {instrument_name}")
    
    # Set up time range for last 7 days
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    
    # Convert to milliseconds timestamp
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    print(f"Attempting to fetch trades from {start_time} to {end_time}")
    
    async with aiohttp.ClientSession() as session:
        result = await api.fetch_public_trades_async(
            session, 
            instrument_name, 
            start_timestamp, 
            end_timestamp
        )
        
        if result and 'result' in result:
            trades = result['result']['trades']
            if trades:
                # Sort trades by timestamp in descending order
                trades.sort(key=lambda x: x['timestamp'], reverse=True)
                
                # Get time range of available trades
                latest_trade = trades[0]
                oldest_trade = trades[-1]
                latest_time = datetime.fromtimestamp(latest_trade['timestamp'] / 1000)
                oldest_time = datetime.fromtimestamp(oldest_trade['timestamp'] / 1000)
                
                # Extract option details
                expiration_date, strike_price, option_type = api.extract_option_details(instrument_name)
                
                print("\nInstrument Details:")
                print(f"Strike Price: {strike_price}")
                print(f"Option Type: {option_type}")
                print(f"Expiration: {expiration_date}")
                
                print("\nTrade History Summary:")
                print(f"Total number of trades: {len(trades)}")
                print(f"Oldest trade: {oldest_time}")
                print(f"Latest trade: {latest_time}")
                print(f"Time span: {latest_time - oldest_time}")
                
                # Calculate some basic statistics
                volumes = [trade['amount'] for trade in trades]
                prices = [trade['price'] for trade in trades]
                directions = [trade.get('direction', 'N/A') for trade in trades]
                buy_count = directions.count('buy')
                sell_count = directions.count('sell')
                
                print("\nTrading Statistics:")
                print(f"Total volume traded: {sum(volumes):.4f}")
                print(f"Average trade size: {sum(volumes)/len(volumes):.4f}")
                print(f"Highest price: {max(prices):.8f}")
                print(f"Lowest price: {min(prices):.8f}")
                print(f"Buy trades: {buy_count}")
                print(f"Sell trades: {sell_count}")
                
                # Show the 5 most recent trades
                print("\nMost recent 5 trades:")
                for i, trade in enumerate(trades[:5], 1):
                    trade_time = datetime.fromtimestamp(trade['timestamp'] / 1000)
                    print(f"\nTrade {i}:")
                    print(f"Time: {trade_time}")
                    print(f"Price: {trade['price']:.8f}")
                    print(f"Amount: {trade['amount']}")
                    print(f"Direction: {trade.get('direction', 'N/A')}")
                    print(f"Index Price: {trade.get('index_price', 'N/A')}")
            else:
                print(f"No trades found for {instrument_name} in the specified time period")
                print("Try using a different strike price or expiration date")
        else:
            print("Error fetching trades")

if __name__ == "__main__":
    asyncio.run(test_fetch_trades())