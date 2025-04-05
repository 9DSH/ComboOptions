from Fetch_data import Fetching_data
from Analytics import Analytic_processing
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import streamlit as st
import plotly.graph_objects as go

analytics = Analytic_processing()
fetch_data = Fetching_data()

filtered_df = fetch_data.load_market_trades()

target_columns = ['BlockTrade IDs', 'BlockTrade Count', 'Combo ID', 'ComboTrade IDs']
strategy_trades_df = filtered_df[~filtered_df[target_columns].isna().all(axis=1)]

strategy_groups = strategy_trades_df.groupby(['BlockTrade IDs', 'Combo ID'])
strategy_df = analytics.Identify_combo_strategies(strategy_groups)

sorted_strategies = []

for (block_id, combo_id), group in strategy_groups:
    total_premium = group['Entry Value'].sum()
    strategy_type = strategy_df[strategy_df['Strategy ID'] == combo_id]['Strategy Type'].iloc[0]
    sorted_strategies.append((block_id, combo_id, group, total_premium, strategy_type))

# Sort by total premium in descending order
sorted_strategies.sort(key=lambda x: x[3], reverse=True)

# Function to calculate profits and create a plot
def calculate_and_plot_all_days_profits(group):
    # Calculate the minimum and maximum time to expiration in days for existing positions
    expiration_dates = [
        (pd.to_datetime(position['Expiration Date'], utc=True) - datetime.now(timezone.utc)).days 
        for i, position in group.iterrows()
    ]
    min_time_to_expiration_days = min(expiration_dates)
    max_time_to_expiration_days = max(expiration_dates)
    if max_time_to_expiration_days <= 1:
        min_time_to_expiration_days = 0
        max_time_to_expiration_days = 1
    
    # Calculate profits for each day until expiration
    days_to_expiration = np.arange(min_time_to_expiration_days, max_time_to_expiration_days + 1)
    profit_over_days = {day: [] for day in days_to_expiration}

    for day in days_to_expiration:
        daily_profits = []
        for i, position in group.iterrows():
            # Calculate profits for the current day
            expiration_date = pd.to_datetime(position['Expiration Date'], utc=True)
            now_utc = datetime.now(timezone.utc)
            time_to_expiration_days = expiration_date - now_utc - timedelta(days=int(day))  # Convert to int
            
            time_to_expiration_years = max(time_to_expiration_days.total_seconds() / (365 * 24 * 3600), 0.0001)
            
            future_iv = position['IV (%)'] / 100
            risk_free_rate = 0.0  # Example risk-free rate
            
            profits = analytics.calculate_public_profits(
                (np.arange(60000, 120000, 500), position['Side'], position['Strike Price'], position['Price (USD)'], 
                 position['Size'], time_to_expiration_years, risk_free_rate, future_iv, position['Option Type'].lower())
            )
            
            daily_profits.append(profits)
        
        # Sum profits for the current day
        total_daily_profit = np.sum(daily_profits, axis=0)
        profit_over_days[day] = total_daily_profit

    # Create a Plotly line chart for profits over all days
    fig_profit = go.Figure()

    for day in days_to_expiration:
        fig_profit.add_trace(go.Scatter(
            x=np.arange(60000, 120000, 500),
            y=profit_over_days[day],
            mode='lines+markers',
            name=f'Profit for {day} Days to Expiration'
        ))

    # Update layout for the profit chart
    fig_profit.update_layout(
        title='Profit for Each Day to Expiration',
        xaxis_title='Underlying Price',
        yaxis_title='Profit',
        showlegend=True
    )

    return fig_profit

# Streamlit app
st.title("Strategy Profits Visualization")

for block_id, combo_id, group, total_premium, strategy_type in sorted_strategies:
    strategy_label = f"${total_premium:,.0f}  ---  {strategy_type}"
    
    with st.expander(f"Block ID: {block_id}, Combo ID: {combo_id} - {strategy_label}"):
        st.write(group)
        
        # Call the function to calculate profits and create the plot
        fig_profit = calculate_and_plot_all_days_profits(group)

        # Display the profit chart in Streamlit
        st.plotly_chart(fig_profit)