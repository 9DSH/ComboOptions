import logging
import pandas as pd
from Calculations import calculate_option_profit, black_scholes
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from datetime import datetime, timezone
from Fetch_data import Fetching_data

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

fetch_data = Fetching_data()


def calculate_raw_profits(option_details, days_ahead_slider, position_side):
    
    risk_free_rate = 0.0
    change_in_iv  = 0.0
    profit_results = []
    index_price_range = np.arange(70000, 90000, 1000)
    now_utc = datetime.now(timezone.utc).date()

    strike_price = option_details['strike_price'].values[0]
    option_type = option_details['option_type'].values[0]
    bid_price = option_details['bid_price_usd'].values[0]
    ask_price = option_details['ask_price_usd'].values[0]
    bid_iv = option_details['bid_iv'].values[0]
    ask_iv = option_details['ask_iv'].values[0] 
    expiration_date_str = option_details['expiration_date'].values[0]

    expiration_date = pd.to_datetime(expiration_date_str).date()  

    time_to_expiration_days = max((expiration_date - now_utc).days, 1)
    remaining_days = time_to_expiration_days - days_ahead_slider
    time_to_expiration_future = max(remaining_days / 365.0, 0.0001)

    future_iv = ask_iv / 100 + (change_in_iv / 100.0) if position_side == "BUY" else bid_iv / 100 + (change_in_iv / 100.0)
    position_value = ask_price if position_side == "BUY" else bid_price

    for u_price in index_price_range:
        mtm_price = black_scholes(u_price, strike_price, time_to_expiration_future, risk_free_rate, future_iv, option_type)

        estimated_profit = (mtm_price - position_value) if position_side == "BUY" else (position_value - mtm_price)

        profit_results.append({
            'Underlying Price': u_price,
            f'Day {days_ahead_slider} Profit ({position_side})': estimated_profit,
        })

    results_df = pd.DataFrame(profit_results)

    return results_df

def get_optimized_quantities(base_quantity_X, base_quantity_Y, profits_main, profits_combo, max_quantity=2):
    quantity_X = base_quantity_X
    quantity_Y = base_quantity_Y
    new_quantity_X = 0
    new_quantity_Y = 0

    adjustment_factor = 0.1
    logging.info("Starting quantity optimization...")

    main_option_min = profits_main.min() 
    combo_option_min = profits_combo.min() 

    mean_x_profit = (main_option_min * quantity_X)  
    mean_y_profit = (combo_option_min * quantity_Y)  
    total_profit = mean_x_profit + mean_y_profit
    optimized_total_profit = total_profit

    iteration_count = 0
    max_iterations = 100  # Prevent infinite loop

    profit_combinations = {}
    
    while (quantity_X != new_quantity_X or quantity_Y != new_quantity_Y) and iteration_count < max_iterations:
        # Log current quantities and profits
        logging.info(f"Current quantities => X: {quantity_X}, Y: {quantity_Y}")
        logging.info(f"Current profits => Mean X: {mean_x_profit}, Mean Y: {mean_y_profit}, Total Profit: {total_profit}")

        
        new_quantity_X = max(0.1, quantity_X - adjustment_factor)
        new_quantity_X = min(new_quantity_X, max_quantity)
        new_quantity_Y = min(quantity_Y + adjustment_factor, max_quantity)

        # Calculate new profits with new quantities
        new_mean_x_profit = (main_option_min * new_quantity_X)  
        new_mean_y_profit = (combo_option_min * new_quantity_Y)  
        new_total_profit = new_mean_x_profit + new_mean_y_profit
        
        # Log updated potential quantities and profits
        logging.info(f"Proposed new quantities => X: {new_quantity_X}, Y: {new_quantity_Y}")
        logging.info(f"Proposed new profits => New Mean X: {new_mean_x_profit}, New Mean Y: {new_mean_y_profit}, New Total Profit: {new_total_profit}")

        if new_total_profit > optimized_total_profit:
            logging.info("No further optimization possible; exiting optimization loop.")
            profit_combinations[(quantity_X, quantity_Y)] = new_total_profit
            quantity_X = new_quantity_X
            quantity_Y = new_quantity_Y
            optimized_total_profit = new_total_profit
            continue
        else:
            # If no beneficial changes are found, we break to avoid infinite loop
            if abs(mean_x_profit - new_mean_x_profit) < 1e-5 and abs(mean_y_profit - new_mean_y_profit) < 1e-5:
                logging.info("No changes to profits detected; exiting optimization loop.")
                break

            # Update means and quantities
            mean_x_profit = new_mean_x_profit
            mean_y_profit = new_mean_y_profit
            quantity_X = new_quantity_X
            quantity_Y = new_quantity_Y

        # Log the updated quantities and profits
        logging.info(f"Updated quantities after adjustment => X: {quantity_X}, Y: {quantity_Y}")
        logging.info(f"Updated profits => Mean X: {mean_x_profit}, Mean Y: {mean_y_profit}")

        iteration_count += 1
    
    return optimized_total_profit, quantity_X, quantity_Y

if __name__ == '__main__':
    
    days_ahead_slider = 1
    initial_quantity = 1

    # Fetch all options with details
    all_options_with_details = fetch_data.get_all_options(filter=None, type='data')
    print(all_options_with_details.shape[0])

    main_option = 'BTC-2MAR25-83000-P' 
    combo_option = 'BTC-2MAR25-80000-C'

    main_option_details, _ = fetch_data.fetch_option_data(main_option)
    combo_option_details, _ = fetch_data.fetch_option_data(combo_option)

    main_profit_buy_raw = calculate_raw_profits(main_option_details, days_ahead_slider, "BUY")
    combo_profit_buy_raw = calculate_raw_profits(combo_option_details, days_ahead_slider, "BUY")
    combo_profit_sell_raw = calculate_raw_profits(combo_option_details, days_ahead_slider, "SELL")

    main_option_profits = main_profit_buy_raw['Day 1 Profit (BUY)']
    combo_option_profits = combo_profit_buy_raw['Day 1 Profit (BUY)']
    print("main:", main_option_profits)
    print("combo:", combo_option_profits)

    BUY_optimized_total_profit, quantity_X, quantity_Y = get_optimized_quantities(initial_quantity, initial_quantity, main_option_profits, combo_option_profits)

    print(BUY_optimized_total_profit)
    print(quantity_X)
    print(quantity_Y)


    total_pro =  (main_option_profits * quantity_X) + ( combo_option_profits * quantity_Y)
    print(total_pro.describe())

    total_proc =  (main_option_profits) + ( combo_option_profits)
    print(total_proc.describe())