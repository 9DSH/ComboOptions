from Fetch_data import Fetching_data

import plotly.graph_objects as go
import pandas as pd

fetch_data = Fetching_data()

fetch_data.load_from_csv(data_type="options_screener")
df = fetch_data.options_screener

# Drop rows with duplicate Trade IDs, keeping only the first occurrence
print(df.shape)



def plot_identified_whale_trades(df):
    
    def scale_marker_size_and_opacity(entry_values, min_size=5, max_size=50, min_opacity=0.3, max_opacity=1.0):
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
    Q1 = df['Entry Value'].quantile(0.25)
    Q3 = df['Entry Value'].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the upper bound for outliers
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    outliers = df[df['Entry Value'] > upper_bound]

    # Scale marker sizes and opacities
    marker_sizes, marker_opacities = scale_marker_size_and_opacity(outliers['Entry Value'])

    # Create a Plotly figure
    fig = go.Figure()

    # Group by 'Entry Date' and plot each group
    grouped = outliers.groupby('Entry Date')

    for entry_date, group in grouped:
        # Prepare hover text for all markers with the same entry date
        hover_text = "<b>Connected Markers:</b><br>"
        for _, row in group.iterrows():
            hover_text += (
                f"Price: {format_value(row['Strike Price'])} | Value: {format_value(row['Entry Value'])}<br>"
            )

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
                    line=dict(width=0)  # Remove stroke around markers
                ),
                name=f"Strike: {row['Strike Price']}",
                hovertemplate=(
                    f"Entry Date: {row['Entry Date']}<br>"
                    f"Strike Price: {format_value(row['Strike Price'])}<br>"
                    f"Entry Value: {format_value(row['Entry Value'])}<br>"
                    f"{hover_text}<extra></extra>"
                )
            ))

    # Update layout of the figure
    fig.update_layout(
        title='Custom Chart: Strike Price vs Entry Date',
        xaxis_title='Entry Date',
        yaxis_title='Strike Price',
        plot_bgcolor='black',  # Set plot background to black
        paper_bgcolor='black',  # Set paper background to black
        font=dict(color='white'),  # Set font color to white for contrast
        xaxis=dict(showgrid=False, zerolinecolor='gray'),
        yaxis=dict(showgrid=False, zerolinecolor='gray'),
        hovermode='closest'
    )

    # Show the plot
    return fig

# Example usage
# Assuming df is your DataFrame with 'Entry Date', 'Strike Price', and 'Entry Value' columns
plot_custom_chart(df)



def plot_identified_whale_trades(df, min_marker_size, max_marker_size, min_opacity, max_opacity, showlegend=True, entry_filter=None):
    # Ensure 'Entry Date' is in datetime format
    df['Entry Date'] = pd.to_datetime(df['Entry Date'])

    # Initialize a list to hold filtered trades
    filtered_trades = []

    # Step 1: Identify outliers based on the IQR method for each strike price
    strike_prices = df['Strike Price'].unique()

    for strike in strike_prices:
        trades_for_strike = df[df['Strike Price'] == strike]

        # Calculate the IQR
        Q1 = trades_for_strike['Entry Value'].quantile(0.25)
        Q3 = trades_for_strike['Entry Value'].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outliers
        lower_bound_value = Q1 - 1.5 * IQR
        upper_bound_value = Q3 + 1.5 * IQR

        # Select outliers that are larger than the upper bound
        outliers = trades_for_strike[trades_for_strike['Entry Value'] > upper_bound_value]

        # Track valid outliers only if they exceed the average size of remaining trades
        remaining_df = trades_for_strike[~trades_for_strike.index.isin(outliers.index)]
        avg_premium_remaining = remaining_df['Entry Value'].mean() if not remaining_df.empty else 0

        valid_premium_outliers = outliers[outliers['Entry Value'] > avg_premium_remaining]

        # Filter upper size 
        Q1 = valid_premium_outliers['Size'].quantile(0.25)
        Q3 = valid_premium_outliers['Size'].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Select outliers that are larger than the upper bound
        outliers_size = valid_premium_outliers[valid_premium_outliers['Size'] > upper_bound]

        # Track valid outliers only if they exceed the average size of remaining trades
        remaining_df_size = valid_premium_outliers[~valid_premium_outliers.index.isin(outliers_size.index)]
        avg_size_remaining = remaining_df_size['Size'].mean() if not remaining_df_size.empty else 0

        valid_outliers = outliers_size[outliers_size['Size'] > avg_size_remaining]

        # Append valid outliers to the filtered trades
        filtered_trades.append(valid_outliers)

    # Combine all filtered trades into a single DataFrame
    filtered_trades_df = pd.concat(filtered_trades) if filtered_trades else pd.DataFrame()

    # Step 4: Check if 'BlockTrade IDs' exists and further filter 
    if 'BlockTrade IDs' in filtered_trades_df.columns:
        block_trade_df = filtered_trades_df[filtered_trades_df['BlockTrade IDs'].notnull()]
        other_trades_df = filtered_trades_df[filtered_trades_df['BlockTrade IDs'].isnull()]
    else:
        block_trade_df = pd.DataFrame()  # Empty DataFrame if column doesn't exist
        other_trades_df = filtered_trades_df

    # Step 5: Group by Entry Date and Strike Price, counting instances and summing sizes
    def group_trades(trades):
        grouped = trades.groupby(['Entry Date', 'Strike Price', 'Side', 'Option Type', 'Expiration Date']).agg(
            total_size=('Entry Value', 'sum'),
            instances=('Entry Value', 'size')
        ).reset_index()
        return grouped

    grouped_block_trades = group_trades(block_trade_df)
    grouped_other_trades = group_trades(other_trades_df)

    # Combine grouped DataFrames
    combined_grouped = pd.concat([grouped_block_trades, grouped_other_trades], ignore_index=True)

    # Apply entry_filter if provided at the last stage before plotting
    if entry_filter is not None:
        combined_grouped = combined_grouped[combined_grouped['total_size'] > entry_filter]

    # Function for scaling marker size and opacity
    def compute_marker_size_and_opacity(group_instances, total_size, max_instances, max_total_size):
        # Calculate marker size
        size_scaling = np.interp(group_instances, 
                                  [1, max_instances], 
                                  [min_marker_size, max_marker_size])
        
        # Calculate opacity
        opacity_scaling = np.clip(total_size / max_total_size, min_opacity, max_opacity)
        
        return size_scaling, opacity_scaling

    # Step 6: Visualize the data
    fig = go.Figure()
    
    max_instances = combined_grouped['instances'].max() if not combined_grouped.empty else 1
    max_total_size = combined_grouped['total_size'].max() if not combined_grouped.empty else 1

    for _, group in combined_grouped.iterrows():
        entry_date = group['Entry Date']
        strike_price = group['Strike Price']
        total_size = group['total_size']
        instances_count = group['instances']
        option_type = group['Option Type']
        option_side = group['Side']
        option_expiration = group['Expiration Date']

        # Calculate marker size and opacity
        group_marker_size, opacity = compute_marker_size_and_opacity(instances_count, total_size, max_instances, max_total_size)

        # Check if the current group is in the block trades group
        is_block_trade = not block_trade_df[(block_trade_df['Entry Date'] == entry_date) & 
                                             (block_trade_df['Strike Price'] == strike_price)].empty

        # Set color based on block trade status
        color = 'red' if is_block_trade else 'yellow'
        text = 'white' if is_block_trade else 'black'
        
        # Construct the hover template
        hover_template = (
            'Entry Date: ' + entry_date.strftime("%Y-%m-%d %H:%M:%S") + '<br>' +
            'Strike Price: ' + str(strike_price) + '<br>' +
            'Side: ' + str(option_side) + '<br>' +
            'Type: ' + str(option_type) + '<br>' +
            'Total Values: ' + f'{total_size:,.0f}' + '<br>' +
            'Expiration Date: ' + str(option_expiration) + '<br>' +
            'Instances: ' + str(instances_count) +
            '<extra></extra>'  # Extra will suppress default hover info
        )

        # Add a trace for this specific Entry Date and Strike Price
        fig.add_trace(go.Scatter(
            x=[entry_date],
            y=[strike_price],
            mode='markers',
            marker=dict(size=group_marker_size, opacity=opacity, color=color),
            name=f'Strike: {strike_price} - Instances: {instances_count}',
            hovertemplate=hover_template,
            hoverinfo='text',
            hoverlabel=dict(bgcolor=color, font=dict(color=text))  # Set background color of hover label
        ))

    # Update layout of the figure
    fig.update_layout(
        xaxis_title='Entry Date',
        yaxis_title='Strike Price',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        showlegend=showlegend,
        xaxis=dict(showgrid=False, title_standoff=10, zerolinecolor='gray'),
        yaxis=dict(showgrid=False, title_standoff=10, zerolinecolor='gray'),
        hovermode='closest'
    )

    # Show the plot
    return fig