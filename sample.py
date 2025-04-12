import requests

def fetch_today_high_low():
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

# Test
max_price, min_price = fetch_today_high_low()
if max_price is not None:
    print(f"Today's High: {max_price}")
    print(f"Today's Low: {min_price}")
else:
    print("Failed to fetch prices")