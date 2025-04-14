from Fetch_data import Fetching_data

import plotly.graph_objects as go
import pandas as pd

fetch_data = Fetching_data()

fetch_data.load_from_csv(data_type="options_screener")
df = fetch_data.options_screener

# Drop rows with duplicate Trade IDs, keeping only the first occurrence
print(df.shape)



def plot_custom_chart(df):
    
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
    fig.show()

# Example usage
# Assuming df is your DataFrame with 'Entry Date', 'Strike Price', and 'Entry Value' columns
plot_custom_chart(df)