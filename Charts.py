import plotly.graph_objects as go
import pandas as pd
from datetime import datetime , timedelta , timezone
import numpy as np
from Analytics import Analytic_processing
from Calculations import calculate_profit

from concurrent.futures import ThreadPoolExecutor

analytics = Analytic_processing()

def plot_option_profit(results_df, 
                       combo_df, 
                       selected_option_name, 
                       combo_option_name, 
                       days_ahead_slider,
                       combo_days_ahead_slider, 
                       selected_option_side,
                       combo_option_side, 
                       breakeven_buy,
                       breakeven_sell, 
                       combo_breakeven_buy, 
                       combo_breakeven_sell):
    
    fig = go.Figure()

    # Initialize maximum profit values
    max_profit_buy = 0
    max_profit_sell = 0
    min_profit_buy =0
    min_profit_sell = 0
    
    combo_max_profit_buy = 0
    combo_max_profit_sell = 0
    combo_min_profit_buy = 0
    combo_min_profit_sell = 0
    
    # Validate the presence of results_df and combo_df
    if results_df is not None and not results_df.empty:
        if f'Day {days_ahead_slider} Profit (BUY)' in results_df.columns:
            max_profit_buy = max(results_df[f'Day {days_ahead_slider} Profit (BUY)'].max(), 0)
            min_profit_buy = min(results_df[f'Day {days_ahead_slider} Profit (BUY)'].min(), 0)
        if f'Day {days_ahead_slider} Profit (SELL)' in results_df.columns:
            max_profit_sell = max(results_df[f'Day {days_ahead_slider} Profit (SELL)'].max(), 0)
            min_profit_sell = min(results_df[f'Day {days_ahead_slider} Profit (SELL)'].min(), 0)

    if combo_df is not None and not combo_df.empty:
        if f'Day {combo_days_ahead_slider} Profit (BUY)' in combo_df.columns:
            combo_max_profit_buy = max(combo_df[f'Day {combo_days_ahead_slider} Profit (BUY)'].max(), 0)
            combo_min_profit_buy = min(combo_df[f'Day {combo_days_ahead_slider} Profit (BUY)'].min(), 0)
        if f'Day {combo_days_ahead_slider} Profit (SELL)' in combo_df.columns:
            combo_max_profit_sell = max(combo_df[f'Day {combo_days_ahead_slider} Profit (SELL)'].max(), 0)
            combo_min_profit_sell = min(combo_df[f'Day {combo_days_ahead_slider} Profit (SELL)'].min(), 0)

    # Determine the maximum Y-values for annotations
    max_buy_value = max(max_profit_buy, combo_max_profit_buy)
    max_sell_value = max(max_profit_sell, combo_max_profit_sell)

    # Determine the minimum Y-values for annotations
    min_buy_value = min(min_profit_buy, combo_min_profit_buy)
    min_sell_value = min(min_profit_sell, combo_min_profit_sell)

    # Create a common function to add traces with appropriate labels
    def add_traces(df, is_combo, option_side, days_ahead):
        # Determine the color
        color = 'yellow' if is_combo else 'red'  # Combo lines in yellow; results_df lines in red

        if option_side == 'BUY':
            # PnL for BUY
            fig.add_trace(go.Scatter(
                x=df['Underlying Price'], 
                y=df[f'Day {days_ahead} Profit (BUY)'],
                mode='lines',
                name=f'PnL {selected_option_name} (BUY)' if not is_combo else f'PnL {combo_option_name} (BUY)',
                line=dict(color=color, width=2),  # Solid line
                hovertemplate=(f'{selected_option_name} (BUY)' if not is_combo else f'{combo_option_name} (BUY)') + '<br>' +
                              '<b>PnL</b>: %{y:.2f}<br>' +
                              '<extra></extra>'
            ))

            # Expiration PnL for BUY
            fig.add_trace(go.Scatter(
                x=df['Underlying Price'], 
                y=df['Expiration Profit (BUY)'],
                mode='lines',
                name=f'Expiry PnL {selected_option_name} (BUY)' if not is_combo else f'Expiry PnL {combo_option_name} (BUY)',
                line=dict(color=color, dash='dash'),  # Dashed line
                hovertemplate=(f'{selected_option_name} (BUY)' if not is_combo else f'{combo_option_name} (BUY)') + '<br>' +
                              '<b>Expiry PnL</b>: %{y:.2f}<br>' +
                              '<extra></extra>'
            ))

            # Breakeven line for BUY from results_df if it's not None
            if breakeven_buy is not None:
                fig.add_shape(
                    type="line",
                    x0=breakeven_buy,
                    y0=min_buy_value,
                    x1=breakeven_buy,
                    y1=max_buy_value,  # Positioning up to the maximum Y-value
                    line=dict(color='rgba(255, 0, 0, 0.7)' , width=2, dash="dot")
                )
                # Add label for breakeven of results_df
                fig.add_annotation(
                    x=breakeven_buy,
                    y=max_buy_value ,  # Offset above the max Y-value
                    text=f'Breakeven: {breakeven_buy:.2f}',
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-10,
                    font=dict(size=12),  # Set font size to 12 for better visibility
                    bgcolor='rgba(255, 255, 255, 0)',  # Transparent background
                    bordercolor='black',
                    borderwidth=1
                )

            # Breakeven line for BUY from combo_df if it's not None
            if combo_breakeven_buy is not None:
                fig.add_shape(
                    type="line",
                    x0=combo_breakeven_buy,
                    y0=min_buy_value,
                    x1=combo_breakeven_buy,
                    y1=max_buy_value,  # Positioning up to the maximum Y-value
                    line=dict(color="rgba(255, 255, 0, 0.7)", width=2, dash="dot")  # Yellow for combo breakeven
                )
                # Add label for breakeven of combo_df
                fig.add_annotation(
                    x=combo_breakeven_buy,
                    y=max_buy_value * 1.10,  # Offset above the max Y-value
                    text=f'Breakeven: {combo_breakeven_buy:.2f}',
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-10,
                    font=dict(size=12),  # Set font size to 12 for better visibility
                    bgcolor='rgba(255, 255, 255, 0)',  # Transparent background
                    bordercolor='black',
                    borderwidth=1
                )

        elif option_side == 'SELL':
            # PnL for SELL
            fig.add_trace(go.Scatter(
                x=df['Underlying Price'], 
                y=df[f'Day {days_ahead} Profit (SELL)'],
                mode='lines',
                name=f'PnL {selected_option_name} (SELL)' if not is_combo else f'PnL {combo_option_name} (SELL)',
                line=dict(color=color, width=2),  # Solid line
                hovertemplate=(f'{selected_option_name} (SELL)' if not is_combo else f'{combo_option_name} (SELL)') + '<br>' +
                              '<b>PnL</b>: %{y:.2f}<br>' +
                              '<extra></extra>'
            ))

            # Expiration PnL for SELL
            fig.add_trace(go.Scatter(
                x=df['Underlying Price'], 
                y=df['Expiration Profit (SELL)'],
                mode='lines',
                name=f'Expiry PnL {selected_option_name} (SELL)' if not is_combo else f'Expiry PnL {combo_option_name} (SELL)',
                line=dict(color=color, dash='dash'),  # Dashed line
                hovertemplate=(f'{selected_option_name} (SELL)' if not is_combo else f'{combo_option_name} (SELL)') + '<br>' +
                              '<b>Expiry PnL</b>: %{y:.2f}<br>' +
                              '<extra></extra>'
            ))

            # Breakeven line for SELL from results_df if it's not None
            if breakeven_sell is not None:
                fig.add_shape(
                    type="line",
                    x0=breakeven_sell,
                    y0=min_sell_value,
                    x1=breakeven_sell,
                    y1=max_sell_value,  # Positioning up to the maximum Y-value
                    line=dict(color='rgba(255, 0, 0, 0.7)' , width=2, dash="dot")
                )
                # Add label for breakeven of results_df
                fig.add_annotation(
                    x=breakeven_sell,
                    y=min_sell_value * 0.95,  # Offset above the max Y-value
                    text=f'Breakeven: {breakeven_sell:.2f}',
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-10,
                    font=dict(size=12),  # Set font size to 12 for better visibility
                    bgcolor='rgba(255, 255, 255, 0)',  # Transparent background
                    bordercolor='black',
                    borderwidth=1
                )

            # Breakeven line for SELL from combo_df if it's not None
            if combo_breakeven_sell is not None:
                fig.add_shape(
                    type="line",
                    x0=combo_breakeven_sell,
                    y0=min_sell_value,
                    x1=combo_breakeven_sell,
                    y1=max_sell_value,  # Positioning up to the maximum Y-value
                    line=dict(color="rgba(255, 255, 0, 0.7)", width=2, dash="dot")
                )
                # Add label for breakeven of combo_df
                fig.add_annotation(
                    x=combo_breakeven_sell,
                    y=max_sell_value * 0.95,  # Offset above the max Y-value
                    text=f'Breakeven: {combo_breakeven_sell:.2f}',
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-10,
                    font=dict(size=12),  # Set font size to 12 for better visibility
                    bgcolor='rgba(255, 255, 255, 0)',  # Transparent background
                    bordercolor='black',
                    borderwidth=1
                )

    # Add traces for Buy Options from the combo_df only if combo_df is not empty
    if combo_option_side == "BUY" and combo_df is not None and not combo_df.empty:
        add_traces(combo_df, True, 'BUY', combo_days_ahead_slider)

    # Add traces for Sell Options from the combo_df only if combo_df is not empty
    if combo_option_side == "SELL" and combo_df is not None and not combo_df.empty:
        add_traces(combo_df, True, 'SELL', combo_days_ahead_slider)

    # Add traces for Buy Options from the results_df only if results_df is not empty
    if selected_option_side == "BUY" and results_df is not None and not results_df.empty:
        add_traces(results_df, False, 'BUY', days_ahead_slider)

    # Add traces for Sell Options from the results_df only if results_df is not empty
    if selected_option_side == "SELL" and results_df is not None and not results_df.empty:
        add_traces(results_df, False, 'SELL', days_ahead_slider)

    # Update layout for the figure
    fig.update_layout(
        xaxis_title='Underlying Price',
        yaxis_title='Profit',
        legend_title='Options',
        hovermode='x unified',  # Aligns hover information across multiple traces
    )

    return fig
def plot_stacked_calls_puts(df):
    """
    Plot a stacked column chart of total Calls and total Puts against Strike Price,
    separating Buy and Sell transactions.

    Parameters:
        df (pd.DataFrame): DataFrame containing options data with 'Strike Price', 
                           'Option Type', and 'Side' columns.
    """
    # Create a DataFrame to hold counts of Calls and Puts by Strike Price
    plot_data = df.copy()

    # Group by Strike Price and Option Type
    grouped_data = plot_data.groupby(['Strike Price', 'Option Type']).agg(
        Buy_Total=('Side', lambda x: (x == 'BUY').sum()),   # Total Buys
        Sell_Total=('Side', lambda x: (x == 'SELL').sum())  # Total Sells
    ).unstack(fill_value=0)  # Unstack the DataFrame for clarity

    # Remove the MultiIndex created by unstack and flatten the DataFrame
    grouped_data.columns = ['_'.join(col).strip() for col in grouped_data.columns.values]

    # Calculate totals for individual puts and calls
    grouped_data['Total_Puts'] = grouped_data['Buy_Total_Put'] + grouped_data['Sell_Total_Put']
    grouped_data['Total_Calls'] = grouped_data['Buy_Total_Call'] + grouped_data['Sell_Total_Call']

    # Create custom data for hover
    customdata = []
    for index, row in grouped_data.iterrows():
        buy_calls_total = row['Buy_Total_Call']
        sell_calls_total = row['Sell_Total_Call']
        buy_puts_total = row['Buy_Total_Put']
        sell_puts_total = row['Sell_Total_Put']

        customdata.append([
            buy_calls_total, 
            sell_calls_total, 
            buy_puts_total, 
            sell_puts_total,
            (buy_calls_total / row['Total_Calls'] * 100) if row['Total_Calls'] > 0 else 0,
            (sell_calls_total / row['Total_Calls'] * 100) if row['Total_Calls'] > 0 else 0,
            (buy_puts_total / row['Total_Puts'] * 100) if row['Total_Puts'] > 0 else 0,
            (sell_puts_total / row['Total_Puts'] * 100) if row['Total_Puts'] > 0 else 0,
        ])

    customdata = pd.DataFrame(customdata).values

    # Create an interactive stacked bar chart
    fig = go.Figure()

    # Define the order for stacking (reversed order)
    stack_order = [ ('Sell', 'Call') , ('Buy', 'Call'), ('Sell', 'Put'), ('Buy', 'Put')]

    # Add traces in reverse order to change the stacking
    for opt_type in stack_order:
        action, option = opt_type
        y_value = grouped_data[f'{action}_Total_{option}']
        
        fig.add_trace(go.Bar(
            x=grouped_data.index,
            y=y_value,
            name=f'{action} {option}s',
            marker=dict(
                color='green' if action == 'Buy' and option == 'Put' 
                      else 'red' if action == 'Sell' and option == 'Put' 
                      else 'teal' if action == 'Buy' and option == 'Call' 
                      else 'orange',
                line=dict(width=0)  # Remove the white border
            ),
            hovertemplate=(
                "Strike Price: %{x}<br>" +
                "Buy Calls: %{customdata[0]} (%{customdata[4]:.2f}%)<br>" +
                "Sell Calls: %{customdata[1]} (%{customdata[5]:.2f}%)<br>" +
                "Buy Puts: %{customdata[2]} (%{customdata[6]:.2f}%)<br>" +
                "Sell Puts: %{customdata[3]} (%{customdata[7]:.2f}%)<br>"
            ),
            customdata=customdata
        ))

    # Update layout settings
    fig.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Total Number',
        barmode='stack',
        template="plotly_white"  # Clean background
    )

    return fig

def plot_strike_price_vs_entry_value(filtered_df):
    fig = go.Figure()

    # Create hover text in a vectorized manner
    filtered_df['hover_text'] = (
        "Strike Price: " + filtered_df['Strike Price'].apply(lambda x: f"{int(x/1000)}k" if x >= 1000 else f"{int(x):,}").astype(str) + "<br>" +
        "Expiry Date: " + filtered_df['Expiration Date'].astype(str) + "<br>" +
        "Entry Value: " + filtered_df['Entry Value'].apply(lambda x: f"{int(x):,}").astype(str) + "<br>" +
        "Type: " + filtered_df['Option Type'] +"<br>" +
        "Size: " + filtered_df['Size'].astype(str) + "<br>" +
        "Side: " + filtered_df['Side'] + "<br>" +
        "Underlying Price: " + filtered_df['Underlying Price'].apply(lambda x: f"{int(x):,}").astype(str) + "<br>"  +
        "Entry Date: " + filtered_df['Entry Date'].astype(str) + "<br>" 
    )

    # Add traces for Buy Put options (downward green triangle)
    fig.add_trace(go.Scatter(
        x=filtered_df.loc[(filtered_df['Option Type'] == 'Put') & (filtered_df['Side'] == 'BUY'), 'Strike Price'],
        y=filtered_df.loc[(filtered_df['Option Type'] == 'Put') & (filtered_df['Side'] == 'BUY'), 'Entry Value'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='green', size=10, opacity=1, line=dict(color='black', width=0.5)),  # Downward green triangle for BUY with black border
        name='Buy Puts',
        hoverinfo='text', 
        hovertext=filtered_df.loc[(filtered_df['Option Type'] == 'Put') & (filtered_df['Side'] == 'BUY'), 'hover_text']
    ))

    # Add traces for Sell Put options (downward red triangle)
    fig.add_trace(go.Scatter(
        x=filtered_df.loc[(filtered_df['Option Type'] == 'Put') & (filtered_df['Side'] == 'SELL'), 'Strike Price'],
        y=filtered_df.loc[(filtered_df['Option Type'] == 'Put') & (filtered_df['Side'] == 'SELL'), 'Entry Value'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=10, opacity=1, line=dict(color='black', width=0.5)),  # Downward red triangle for SELL with black border
        name='Sell Puts',
        hoverinfo='text', 
        hovertext=filtered_df.loc[(filtered_df['Option Type'] == 'Put') & (filtered_df['Side'] == 'SELL'), 'hover_text']
    ))

    # Add traces for Buy Call options (upward green triangle)
    fig.add_trace(go.Scatter(
        x=filtered_df.loc[(filtered_df['Option Type'] == 'Call') & (filtered_df['Side'] == 'BUY'), 'Strike Price'],
        y=filtered_df.loc[(filtered_df['Option Type'] == 'Call') & (filtered_df['Side'] == 'BUY'), 'Entry Value'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='teal', size=10, opacity=1, line=dict(color='black', width=0.5)),  # Upward teal triangle for BUY with black border
        name='Buy Calls',
        hoverinfo='text', 
        hovertext=filtered_df.loc[(filtered_df['Option Type'] == 'Call') & (filtered_df['Side'] == 'BUY'), 'hover_text']
    ))

    # Add traces for Sell Call options (upward red triangle)
    fig.add_trace(go.Scatter(
        x=filtered_df.loc[(filtered_df['Option Type'] == 'Call') & (filtered_df['Side'] == 'SELL'), 'Strike Price'],
        y=filtered_df.loc[(filtered_df['Option Type'] == 'Call') & (filtered_df['Side'] == 'SELL'), 'Entry Value'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='darkorange', size=10, opacity=1, line=dict(color='black', width=0.5)),  # Upward dark orange triangle for SELL with black border
        name='Sell Calls',
        hoverinfo='text', 
        hovertext=filtered_df.loc[(filtered_df['Option Type'] == 'Call') & (filtered_df['Side'] == 'SELL'), 'hover_text']
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Entry Value',
        template="plotly_white",  # Use a clean white template
        showlegend=True
    )

    return fig

def plot_radar_chart(df_options_for_strike):
    # Check if required columns exist
    if 'Expiration Date' not in df_options_for_strike.columns or 'open_interest' not in df_options_for_strike.columns:
        print("DataFrame must contain 'expiration_date' and 'open_interest' columns.")
        return

    # Convert 'expiration_date' to datetime
    exp_dates = pd.to_datetime(df_options_for_strike['Expiration Date'])

    # Prepare labels:
    # 'categories' for plotting (in original format)
    categories = exp_dates.dt.strftime('%m/%d/%Y').tolist()
    # 'formatted_categories' for display (e.g., '4 July 25')
    formatted_categories = exp_dates.dt.strftime('%#d %B').tolist()  # Use '%-d' on Unix-based systems

    # Extract values for the radar chart
    values = df_options_for_strike['open_interest'].tolist()

    # Close the radar chart by repeating the first value
    values += values[:1]
    categories += categories[:1]
    formatted_categories += formatted_categories[:1]

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Open Interest',
        line=dict(color='red', width=2),  # Change color of the line to red
        hovertemplate='Open Interest: %{r}<br>Expiration Date: %{theta}<extra></extra>'  # Add hover labels
    ))

    # Update the layout of the radar chart
    fig.update_layout(
        title='Open Interest by Expiration Date',
        polar=dict(
            bgcolor='rgba(0,0,0,0)',  # Set plot background to black
            radialaxis=dict(
                visible=True,
                 range=[0, max(values) + 10],  # Adjust the range as necessary
            ),
            angularaxis=dict(
                tickcolor='white',  # Color of the angular ticks
                tickfont=dict(color='white'),  # Tick font color
                tickvals=categories,       # Original values for proper plotting
                ticktext=formatted_categories  # Formatted text for display
            )
        ),
        showlegend=True
    )

    return fig

def plot_public_profit_sums(summed_df):
    """
    Plot the Underlying Price against the Sum of Profits using Plotly.
    
    Parameters:
        summed_df (pd.DataFrame): DataFrame containing 'Underlying Price' and 'Sum of Profits'.
    """
    fig = go.Figure()

    # Add a line trace
    fig.add_trace(go.Scatter(
        x=summed_df['Underlying Price'],
        y=summed_df['Sum of Profits'],
        mode='lines+markers',
        name='Sum of Profits',
        line=dict(shape='linear'),  # Change line shape to linear
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Underlying Price vs. Sum of Profits',
        xaxis_title='Underlying Price',
        yaxis_title='Sum of Profits',
        template='plotly_white'
    )

    return fig
def plot_most_traded_instruments(most_traded):
    """
    Plots a pie chart of the most traded instruments with hover info showing
    total size, buy, and sell counts, each clearly displayed.
    """
    # Create hover text to display buy and sell portions clearly
    most_traded['hover_text'] = (
        "Instrument: " + most_traded['Instrument'] + "<br>" +
        "Total Size: " + most_traded['Size'].astype(int).astype(str) + "<br>" +  # No decimal
        "Buy Contracts: " + most_traded['BUY'].astype(str) + " - " + ((most_traded['BUY'] / (most_traded['SELL']+most_traded['BUY'] )) * 100).round(2).astype(str) + "%" + "<br>" +
        "Sell Contracts: " + most_traded['SELL'].astype(str) + " - " + ((most_traded['SELL'] / (most_traded['SELL']+most_traded['BUY'] )) * 100).round(2).astype(str) + "%" 
    )

    # Create a pie chart using Plotly
    fig = go.Figure(data=[go.Pie(
        labels=most_traded['Instrument'],
        values=most_traded['Size'],
        hole=0.6,
        hoverinfo='text',
        textinfo='value+percent+label',
        text=most_traded['hover_text']
    )])

    # Update the layout of the chart
    fig.update_layout(
        title_text='Top 10 Most Traded Instruments by Contracts',
        title_font_size=24
    )
    
    return fig

def plot_underlying_price_vs_entry_value(df, custom_price=None, custom_entry_value=None):
    # Helper function to format date with error handling
    def format_entry_date(date_obj):
        if isinstance(date_obj, datetime):
            return date_obj.strftime('%Y-%m-%d %H:%M:%S')
        return 'Invalid Date'

    # Helper function to format large numbers
    def format_large_number(value):
        if abs(value) >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.1f}k"
        else:
            return f"{value:.1f}"

    # Create a scatter plot of Entry Value against Underlying Price
    fig = go.Figure()

    # Separate data for BUY and SELL with corresponding colors
    df_sell = df[df['Side'] == 'SELL']
    df_buy = df[df['Side'] == 'BUY']
    # Avoid SettingWithCopyWarning by assigning to a new DataFrame
    df_sell = df_sell.assign(Formatted_Entry_Date=df_sell['Entry Date'].apply(format_entry_date))
    df_buy = df_buy.assign(Formatted_Entry_Date=df_buy['Entry Date'].apply(format_entry_date))

    # Add SELL data points (red)
    fig.add_trace(go.Scatter(
        x=df_sell['Entry Value'],
        y=df_sell['Underlying Price'],
        mode='markers',
        marker=dict(size=10, 
                    color='red', 
                    opacity=0.6,       
                    line=dict(        
                        color='black', # Color of the border
                        width=1         # Width of the border
                )),
        name='SELL',
        hovertemplate=(  
            '<b>Underlying Price:</b> %{y:.1f}<br>'  
            '<b>Entry Value:</b> %{customdata[0]}<br>' 
            '<b>Size:</b> %{customdata[2]}<br>'   
            '<b>Entry Date:</b> %{customdata[1]}<br>' 
            '<extra></extra>'  
        ),
        customdata=np.array([
            [format_large_number(x), date, size]
            for x, date, size in zip(df_sell['Entry Value'], df_sell['Formatted_Entry_Date'], df_sell['Size'])
        ])
    ))

    # Add BUY data points (white)
    fig.add_trace(go.Scatter(
        x=df_buy['Entry Value'],
        y=df_buy['Underlying Price'],
        mode='markers',
        marker=dict(
            size=10, 
            color='white', 
            opacity=0.6,       
            line=dict(        
                color='black', # Color of the border
                width=1         # Width of the border
        )),
        name='BUY',
        hovertemplate=(  
            '<b>Underlying Price:</b> %{y:.1f}<br>'  
            '<b>Entry Value:</b> %{customdata[0]}<br>'
            '<b>Size:</b> %{customdata[2]}<br>'  
            '<b>Entry Date:</b> %{customdata[1]}<br>'  
            '<extra></extra>'  
        ),
        customdata=np.array([
            [format_large_number(x), date, size]
            for x, date, size in zip(df_buy['Entry Value'], df_buy['Formatted_Entry_Date'], df_buy['Size'])
        ])
    ))

    # Custom point if provided
    if custom_price is not None and custom_entry_value is not None:
        fig.add_trace(go.Scatter(
            x=[custom_entry_value],
            y=[custom_price],
            mode='markers',
            marker=dict(size=15, 
                        color='green', 
                        symbol='circle',       
                        line=dict(        
                            color='black', # Color of the border
                            width=1         # Width of the border
                    )),
            name='Your Option',
            hovertemplate='<b>Your Entry Price:</b> ' + format_large_number(custom_price)  + '<br>' +
                          '<b>Your Premium:</b> ' + format_large_number(custom_entry_value) + '<br>' +
                          '<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title='Underlying Price vs Entry Value',
        xaxis_title='Entry Value',    
        yaxis_title='Underlying Price',       
        template='plotly_dark',
        hoverlabel=dict(bgcolor='black', font_color='white')
    )

    return fig

def plot_price_vs_entry_date(df):
    # Ensure 'Entry Date' is in datetime format
    df['Entry Date'] = pd.to_datetime(df['Entry Date'])


    # Create traces for BUY and SELL
    buy_df = df[df['Side'] == 'BUY']
    sell_df = df[df['Side'] == 'SELL']

    fig = go.Figure()

    # Add BUY trace with transparency
    fig.add_trace(go.Scatter(
        x=buy_df['Entry Date'],
        y=buy_df['Price (USD)'],  # Use 'Price (USD)' here
        mode='lines',  # Use lines
        name='BUY',
        line=dict(color='white', width=2),  # White line
        marker=dict(size=8),
        hovertext=(
            'Underlying Price: ' + buy_df['Underlying Price'].map('{:.1f}'.format).astype(str) + '<br>' +
            'Price (USD): ' + buy_df['Price (USD)'].map('{:.1f}'.format) + '<br>' +  # Updated here
            'Size: ' + buy_df['Size'].astype(str) + '<br>' +
            'Side: ' + buy_df['Side'] + '<br>'  +
            'Entry Date: ' + buy_df['Entry Date'].dt.strftime('%Y-%m-%d %H:%M:%S') + '<br>'  # Include time
        ),
        hoverinfo='text',
        hoverlabel=dict(bgcolor='black', font=dict(color='white')),  # Set hover background to black
        opacity=0.6  # Set trace transparency here
    ))

    # Add SELL trace with transparency
    fig.add_trace(go.Scatter(
        x=sell_df['Entry Date'],
        y=sell_df['Price (USD)'],  # Use 'Price (USD)' here
        mode='lines',  # Use lines
        name='SELL',
        line=dict(color='red', width=2),  # Red line
        marker=dict(size=8),
        hovertext=(
            'Underlying Price: ' + sell_df['Underlying Price'].map('{:.1f}'.format).astype(str) + '<br>' +
            'Price (USD): ' + sell_df['Price (USD)'].map('{:.1f}'.format) + '<br>' +  # Updated here
            'Size: ' + sell_df['Size'].astype(str) + '<br>' +
            'Side: ' + sell_df['Side'] + '<br>' +
            'Entry Date: ' + sell_df['Entry Date'].dt.strftime('%Y-%m-%d %H:%M:%S') + '<br>'   # Include time
        ),
        hoverinfo='text',
        hoverlabel=dict(bgcolor='black', font=dict(color='white')),  # Set hover background to black
        opacity=0.6  # Set trace transparency here
    ))

    # Update layout with transparent background
    fig.update_layout(
        title='Option Price vs Entry Date',
        xaxis_title='Entry Date',
        yaxis_title='Price (USD)',  # Updated y-axis label
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the entire figure
        font=dict(color='white'),  # White font for labels
        xaxis=dict(
            showgrid=False,
            title_standoff=10,
            zerolinecolor='gray'  # Gray zero line
        ),
        yaxis=dict(
            showgrid=False,
            title_standoff=10,
            zerolinecolor='gray'  # Gray zero line
        ),
        hovermode='closest'
    )

    # Show the plot
    return fig

def plot_identified_whale_trades(df, min_size=5, max_size=50, min_opacity=0.3, max_opacity=1.0, entry_value_threshold=None, filter_type=None):
    
    if filter_type == "Entry Value":
        filter_type_str = 'Entry Value'
    else:
        filter_type_str = 'Size'

    def scale_marker_size_and_opacity(entry_values):
        normalized_values = (entry_values - entry_values.min()) / (entry_values.max() - entry_values.min())
        scaled_sizes = normalized_values * (max_size - min_size) + min_size
        scaled_opacities = normalized_values * (max_opacity - min_opacity) + min_opacity
        return scaled_sizes, scaled_opacities

    def format_value(value):
        """Format the value to show 'k' for thousands and 'M' for millions."""
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"{value / 1_000:.0f}k"
        else:
            return f"{value:,}"

    # Ensure 'Entry Date' is in datetime format
    df['Entry Date'] = pd.to_datetime(df['Entry Date'])

    # Calculate IQR for 'Entry Value'
    Q1 = df[filter_type_str].quantile(0.25)
    Q3 = df[filter_type_str].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the upper bound for outliers
    upper_bound = Q3 + 1.5 * IQR

    # Use the manual threshold if provided, otherwise use the calculated upper bound
    threshold = entry_value_threshold if entry_value_threshold is not None else upper_bound

    # Filter out the outliers based on the threshold
    outliers = df[df[filter_type_str] > threshold]

    # Scale marker sizes and opacities
    marker_sizes, marker_opacities = scale_marker_size_and_opacity(outliers[filter_type_str])

    # Create a Plotly figure
    fig = go.Figure()

    # Group by 'Entry Date' and plot each group
    grouped = outliers.groupby('Entry Date')

    for entry_date, group in grouped:
        # Prepare hover text for all markers with the same entry date
        if len(group) > 1:
            hover_text = "<b>Connected Markers:</b><br>"
            for _, row in group.iterrows():
                hover_text += (
                    f"Price: {format_value(row['Strike Price'])} | Value: {format_value(row[filter_type_str])}<br>"
                )
        else:
            hover_text = "Connected Markers: N/A"

        # Add markers for each outlier
        for i, row in group.iterrows():
            # Determine marker color based on BlockTrade IDs
            marker_color = 'red' if pd.notnull(row['BlockTrade IDs']) else 'yellow'
            
            fig.add_trace(go.Scatter(
                x=[row['Entry Date']],
                y=[row['Strike Price']],
                mode='markers',
                marker=dict(
                    size=marker_sizes[i],  # Use scaled marker size
                    color=marker_color,
                    opacity=marker_opacities[i],  # Use scaled opacity
                    line=dict(color='black', width=1)  # Remove stroke around markers
                ),
                name=f"Strike: {row['Strike Price']}",
                hovertemplate=(
                    f"Entry Date: {row['Entry Date']}<br>"
                    f"Strike Price: {format_value(row['Strike Price'])}<br>"
                    f"Value: {format_value(row[filter_type_str])}<br>"
                    f"{hover_text}<extra></extra>"
                )
            ))

    # Update layout of the figure
    fig.update_layout(
        xaxis_title='Entry Date',
        yaxis_title='Strike Price',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),  # Set font color to white for contrast
        xaxis=dict(showgrid=False, zerolinecolor='gray'),
        yaxis=dict(showgrid=False, zerolinecolor='gray'),
        hovermode='closest'
    )

    # Return the filtered DataFrame and the figure
    return outliers, fig


def plot_expiration_profit(strategy_data, chart_type, trade_option_details):
    
    # Determine the minimum and maximum strike prices
    min_strike_price = strategy_data['Strike Price'].min()
    max_strike_price = strategy_data['Strike Price'].max()

    # Generate index price range based on strike prices
    index_price_range = np.arange(min_strike_price - 35000, max_strike_price + 35000, 1000)
    profit_matrix = []
    position_labels = []

    for i, position in strategy_data.iterrows():
        # Extract necessary data for profit calculation

        if chart_type == "Trade":
            trade_direction = trade_option_details[0]
            trade_qty = trade_option_details[1]
            if trade_direction == "Buy":
                position_value = position['Ask Price (USD)']
            else:
                position_value = position['Bid Price (USD)']
            position_size = trade_qty
            position_side = trade_direction

        if chart_type == "Public":
            position_size = position['Size']
            position_side = position['Side']
            position_value = position['Price (USD)']

        strike_price = position['Strike Price']
        position_type = position['Option Type'].lower()

        position_label = f"{int(strike_price)} - {position_type.upper()} - {position_side.upper()}"
        position_labels.append(position_label)  # Add to the list of labels

        premium_value = position_size * position_value
        if position_type == "put":
            breakeven = strike_price - premium_value
        else:
            breakeven = premium_value + strike_price

        profits = [
            calculate_profit(
                current_price=index_price,
                option_price=position_value,
                strike_price=strike_price,
                option_type=position_type,
                quantity=position_size,
                is_buy=(position_side.lower() == 'buy')
            )
            for index_price in index_price_range
        ]
        
        # Append profits to the matrix
        profit_matrix.append(profits)
    
    # Convert profit matrix to a numpy array for plotting
    profit_matrix = np.array(profit_matrix)
    
    # Calculate the sum of profits for each underlying price
    total_profit_row = np.sum(profit_matrix, axis=0)
    
    # Append the total profit row to the profit matrix
    profit_matrix = np.vstack([profit_matrix, total_profit_row])
    
    # Determine if all profit values are negative
    all_negative = np.all(total_profit_row < 0)
    
    # Choose colorscale based on profit values
    colorscale = 'Reds' if all_negative else 'RdYlGn'
    
    # Create a Plotly figure
    fig = go.Figure()

    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=profit_matrix,
        x=index_price_range,
        y=np.arange(len(strategy_data) + 1),  # Adjust for the new total profit row
        colorscale=colorscale,  # Use conditional colorscale
        colorbar=dict(title='Profit'),
        hovertemplate=(
            "Expiration Profit<br>"
            "Underlying Price: %{x}<br>"  # Use custom data for x-axis
            "Profit: %{customdata}<br>"  # Use custom data for z-axis
            "<extra></extra>"  # Suppress default hover info
        ),
        customdata=[
            [f"{int(val/1e6)}M" if abs(val) >= 1e6 else f"{int(val/1e3)}k" if abs(val) >= 1e3 else f"{int(val):,}" for val in row]
            for row in profit_matrix
        ],
        showscale=True,
        zsmooth=False,  # Disable smoothing to make grid lines visible
        xgap=0.5,  # Add gap between x values
        ygap=5   # Add gap between y values
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Underlying Price',
        yaxis=dict(tickvals=np.arange(len(strategy_data) + 1), ticktext=[*position_labels, 'Total Profit']),
        showlegend=False
    )

    return fig

# Function to calculate profits and create a plot
def plot_all_days_profits(group , chart_type , trade_option_details):
    # Calculate the minimum and maximum time to expiration in days for existing positions


    # Determine the minimum and maximum strike prices
    min_strike_price = group['Strike Price'].min()
    max_strike_price = group['Strike Price'].max()

    # Generate index price range based on strike prices
    index_price_range = np.arange(min_strike_price - 35000, max_strike_price + 35000, 1000)

    expiration_dates = [
        (pd.to_datetime(position['Expiration Date'], utc=True) - datetime.now(timezone.utc)).days 
        for i, position in group.iterrows()
    ]
    max_time_to_expiration_days = max(expiration_dates)
    if max_time_to_expiration_days <= 1:
        max_time_to_expiration_days = 1
    
    # Calculate profits for each day until expiration
    days_to_expiration = np.arange(1, max_time_to_expiration_days + 1)
    profit_over_days = {day: [] for day in days_to_expiration}
    
    for day in days_to_expiration:
        daily_profits = []
        for i, position in group.iterrows():
            # Calculate profits for the current day
            expiration_date = pd.to_datetime(position['Expiration Date'], utc=True)
            now_utc = datetime.now(timezone.utc)
            time_to_expiration_days = expiration_date - now_utc - timedelta(days=int(day))  # Convert to int
            
            time_to_expiration_years = max(time_to_expiration_days.total_seconds() / (365 * 24 * 3600), 0.0001)

            if chart_type == "Trade" : 
                trade_direction = trade_option_details[0]
                trade_qty = trade_option_details[1]
                if trade_direction == "Buy" : 
                    IV_str = 'Ask IV' 
                    position_price = position['Ask Price (USD)'] 
                else:
                    IV_str ='Bid IV'
                    position_price = position['Bid Price (USD)'] 
                position_size = trade_qty
            
            if chart_type == "Public" : 
                IV_str = 'IV (%)'
                position_size = position['Size']
                trade_direction = position['Side']
                position_price = position['Price (USD)']
                
            group = group.copy()        
            group[IV_str] = pd.to_numeric(group[IV_str], errors='coerce').fillna(0)   

            future_iv = position[IV_str] / 100
            risk_free_rate = 0.0  # Example risk-free rate

            profits = analytics.calculate_public_profits(
                (index_price_range, trade_direction, position['Strike Price'], position_price, 
                 position_size , time_to_expiration_years, risk_free_rate, future_iv, position['Option Type'].lower())
            )
            
            daily_profits.append(profits)
        
        # Sum profits for the current day
        total_daily_profit = np.sum(daily_profits, axis=0)
        profit_over_days[day] = total_daily_profit

    # Create a Plotly line chart for profits over all days
    fig_profit = go.Figure()

    for day in days_to_expiration:
        # Calculate breakeven prices for the current day
        breakeven_prices = []
        for i in range(1, len(index_price_range)):
            if (profit_over_days[day][i-1] < 0 and profit_over_days[day][i] > 0) or \
               (profit_over_days[day][i-1] > 0 and profit_over_days[day][i] < 0):
                breakeven_prices.append(index_price_range[i])

        # Format breakeven prices for hover
        breakeven_prices_str = ', '.join(
            f"{price/1000000:.1f}M" if abs(price) >= 1000000 else (f"{price/1000:.0f}k" if abs(price) >= 1000 else f"{price:,.0f}")
            for price in breakeven_prices
        ) if breakeven_prices else "N/A"

        fig_profit.add_trace(go.Scatter(
            x=index_price_range,
            y=profit_over_days[day],
            mode='lines',
            hovertemplate=(
                "%{text}<br>"  # Add number of days to expiration
                "Underlying Price: %{customdata[0]}<br>"  # Use custom data for formatted x-axis value
                "Breakeven Prices: %{customdata[2]}<br>"  # Add breakeven prices to hover
                "Profit: %{customdata[1]}<br>"  # Use custom data for formatted y-axis value
                "<extra></extra>"  # Suppress default hover info
            ),
            text=[f"Day {day}" for _ in range(len(profit_over_days[day]))],  # Add day information for hover
            customdata=[
                [
                    f"{x/1000000:.1f}M" if abs(x) >= 1000000 else (f"{x/1000:.0f}k" if abs(x) >= 1000 else f"{x:,.0f}"),
                    f"{y/1000000:.1f}M" if abs(y) >= 1000000 else (f"{y/1000:.0f}k" if abs(y) >= 1000 else f"{y:,.0f}"),
                    breakeven_prices_str
                ]
                for x, y in zip(index_price_range, profit_over_days[day])
            ]  # Format values greater than 1000000 with 'M' and greater than 1000 with 'k', considering negative values
        ))

    # Update layout for the profit chart
    fig_profit.update_layout(
        xaxis_title='Underlying Price',
        yaxis_title='Profit',
        showlegend=False
    )

    return fig_profit


def plot_public_profits(strategy_data , chart_type , trade_option_details):
                                # Calculate profits using multithreading
    with ThreadPoolExecutor() as executor:
        future_all_days_profits = executor.submit(plot_all_days_profits, strategy_data , chart_type , trade_option_details)
        future_strategy_profit = executor.submit(plot_expiration_profit, strategy_data, chart_type , trade_option_details)
                                    
                                    # Retrieve results
    fig_profit = future_all_days_profits.result()
    fig_strategy = future_strategy_profit.result()
                                
    return fig_profit, fig_strategy

def plot_top_strikes_pie_chart(top_strikes):
    """
    Plots a pie chart of the top strike prices with hover info showing
    total size and trade count.
    """
    # Ensure the DataFrame is not empty
    if top_strikes.empty:
        raise ValueError("The top_strikes DataFrame is empty. Cannot plot pie chart.")

    # Create hover text to display total size and trade count
    top_strikes['hover_text'] = (
        "Strike Price: " + top_strikes.index.astype(str) + "<br>" +
        "Total Size: " + top_strikes[('Total Size', 'sum')].astype(int).astype(str) + " BTC<br>" +
        "Count: " + top_strikes[('Total Size', 'count')].astype(int).astype(str) + " Strategies"
    )

    # Create a pie chart using Plotly
    fig = go.Figure(data=[go.Pie(
        labels=top_strikes.index.astype(str),
        values=top_strikes[('Total Size', 'sum')],
        hole=0.6,
        hoverinfo='text',
        textinfo='value+percent+label',
        text=top_strikes['hover_text'],
        textposition='outside'
    )])

    # Update the layout of the chart
    
    return fig


def plot_hourly_activity_radar(hourly_activity):
    """
    Plots a radar chart of hourly activities.
    
    Parameters:
        hourly_activity (pd.DataFrame): DataFrame containing 'Total Size' by hour.
    """
    # Ensure the DataFrame is not empty
    if hourly_activity.empty:
        raise ValueError("The hourly_activity DataFrame is empty. Cannot plot radar chart.")

    # Extract hours and total sizes
    hours = hourly_activity.index.astype(str)  # Convert hours to string for plotting
    total_sizes = hourly_activity['Total Size']['sum'].tolist()

    # Close the loop for the radar chart by repeating the first value
    hours = list(hours) + [hours[0]]
    total_sizes = total_sizes + [total_sizes[0]]

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=total_sizes,
        theta=hours,
        fill='toself',
        name='Hourly Activity',
        line=dict(color='blue', width=2)  # Change color of the line to blue
    ))

    # Update the layout of the radar chart
    fig.update_layout(
        title='Hourly Activity by Total Size',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(total_sizes) + 10],  # Adjust the range as necessary
            ),
            angularaxis=dict(
                tickvals=hours[:-1],  # Exclude the repeated first value for ticks
                ticktext=hours[:-1]   # Exclude the repeated first value for labels
            )
        ),
        showlegend=True
    )

    return fig



def plot_most_strategy_bar_chart(strategy_df_copy):
    """
    Plots a horizontal bar chart of the total size from the strategy DataFrame using Plotly's graph_objects.

    Parameters:
    - strategy_df_copy (pd.DataFrame): DataFrame containing the 'Total Size' data to plot.
    """
    if 'Total Size' in strategy_df_copy and 'sum' in strategy_df_copy['Total Size']:
        # Create a horizontal bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    y=strategy_df_copy.index,  # Use the index as the y-axis
                    x=strategy_df_copy['Total Size']['sum'],  # Use 'Total Size' as the x-axis
                    orientation='h'  # Horizontal orientation
                )
            ]
        )
        fig.update_layout(
            xaxis_title='Total Size',
            yaxis_title='Top Strategies'
        )
    
    return fig


def plot_hourly_activity(hourly_activity):
    """
    Plots a bar chart of hourly activity using Plotly.

    Parameters:
        hourly_activity (pd.DataFrame): DataFrame containing 'hour' and 'activity' columns.
    """
    # Ensure the DataFrame is not empty
    if hourly_activity.empty:
        raise ValueError("The hourly_activity DataFrame is empty. Cannot plot.")

    # Create a bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=hourly_activity.index,  # Assuming the index contains the hours
        y=hourly_activity['Total Size']['sum'],  # Replace 'activity' with the actual column name if different
        marker=dict(color='skyblue'),  # Set the color of the bars to sky blue
        hoverinfo='y',  # Show x and y values on hover
      ))

    # Update layout
    fig.update_layout(
        title='Hourly Activity',
        xaxis_title='Hour',
        yaxis_title='Activity (Volume)',
        template='plotly_white',  # Use a clean white template
        xaxis=dict(tickmode='linear'),  # Ensure all hours are shown
        height=300 # Decrease the height of the chart
    )

    return fig