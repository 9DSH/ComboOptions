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

# Streamlit app
st.title("Strategy Profits Visualization")

for block_id, combo_id, group, total_premium, strategy_type in sorted_strategies:
    strategy_label = f"${total_premium:,.0f}  ---  {strategy_type}"
    
    with st.expander(f"Block ID: {block_id}, Combo ID: {combo_id} - {strategy_label}"):
        st.write(group)
        
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
        
        # Slider for days ahead to expiration
        days_ahead = st.slider("Days ahead to expiration", min_value=min_time_to_expiration_days, max_value=max_time_to_expiration_days, value=0, step=1, key=f"days_ahead_slider_{block_id}_{combo_id}")
        
        # Prepare data for heatmap
        index_price_range = np.arange(60000, 120000, 500)  # Example range
        profit_matrix = []
        position_labels = []

        for i, position in group.iterrows():
            # Extract necessary data for profit calculation
            position_side = position['Side']
            strike_price = position['Strike Price']
            position_value = position['Price (USD)']
            position_size = position['Size']
            position_type = position['Option Type'].lower()
            expiration_date_str = position['Expiration Date']
            entry_price = position['Underlying Price']  # Assuming 'Underlying Price' is the column name for the entry price
             
            position_label = f"{int(strike_price)} - {position_type.upper()} - {position_side.upper()}"
            position_labels.append(position_label)  # Add to the list of labels
    
            # Convert expiration date to datetime
            expiration_date = pd.to_datetime(expiration_date_str, utc=True)
            now_utc = datetime.now(timezone.utc)
            time_to_expiration_days = expiration_date - now_utc - timedelta(days=days_ahead)
            time_to_expiration_years = max(time_to_expiration_days.total_seconds() / (365 * 24 * 3600), 0.0001)
            
            # Calculate profits using the calculate_public_profits function
            future_iv = position['IV (%)'] / 100
            risk_free_rate = 0.0  # Example risk-free rate
            
            profits = analytics.calculate_public_profits(
                (index_price_range, position_side, strike_price, position_value, position_size, time_to_expiration_years, risk_free_rate, future_iv, position_type)
            )
            
            # Append profits to the matrix
            profit_matrix.append(profits)
        
        # Convert profit matrix to a numpy array for plotting
        profit_matrix = np.array(profit_matrix)
        
        # Calculate the sum of profits for each underlying price
        total_profit_row = np.sum(profit_matrix, axis=0)
        
        # Append the total profit row to the profit matrix
        profit_matrix = np.vstack([profit_matrix, total_profit_row])
        
        # Create a Plotly figure
        fig = go.Figure()

        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=profit_matrix,
            x=index_price_range,
            y=np.arange(len(group) + 1),  # Adjust for the new total profit row
            colorscale=[(0, 'red'), (0.5, 'yellow'), (1, 'green')], 
            colorbar=dict(title='Profit'),
            hovertemplate=(
                "Underlying Price: %{x:.0f}K<br>"  # Format x-axis value as K
                "Profit: %{z:,.0f}<br>"  # Format z-axis value with commas
                "<extra></extra>"  # Suppress default hover info
            )
        ))

        # Update layout
        fig.update_layout(
            title='Profit Heatmap',
            xaxis_title='Underlying Price',
            yaxis=dict(tickvals=np.arange(len(group) + 1), ticktext=[*position_labels,  'Total Profit']),
            showlegend=False
        )
        
        # Display the plot in Streamlit
        st.plotly_chart(fig)