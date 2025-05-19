from bokeh.plotting import figure, save, show, output_file
from bokeh.models import ColumnDataSource, CustomJS
import pandas as pd
import numpy as np
from datetime import datetime
from Fetch_data import Fetching_data

# Fetch the financial data
fetch = Fetching_data()
options_ = fetch.fetch_option_data()

def plot_identified_whale_trades(df, min_size=5, max_size=50, min_opacity=0.3, max_opacity=1.0, entry_value_threshold=None, filter_type=None):
    if filter_type == "Entry Value":
        filter_type_str = 'Entry Value'
    else:
        filter_type_str = 'Size'

    def scale_marker_size_and_opacity(entry_values):
        if entry_values.max() == entry_values.min():  # Avoid division by zero
            normalized_values = np.ones_like(entry_values)
        else:
            normalized_values = (entry_values - entry_values.min()) / (entry_values.max() - entry_values.min())
        scaled_sizes = normalized_values * (max_size - min_size) + min_size
        scaled_opacities = normalized_values * (max_opacity - min_opacity) + min_opacity
        return scaled_sizes, scaled_opacities

    df['Entry Date'] = pd.to_datetime(df['Entry Date'])

    Q1 = df[filter_type_str].quantile(0.25)
    Q3 = df[filter_type_str].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    threshold = entry_value_threshold if entry_value_threshold is not None else upper_bound
    outliers = df[df[filter_type_str] > threshold].copy()
    outliers.reset_index(drop=True, inplace=True)

    marker_sizes, marker_opacities = scale_marker_size_and_opacity(outliers[filter_type_str])
    
    outliers['marker_size'] = marker_sizes
    outliers['marker_opacity'] = marker_opacities

    source = ColumnDataSource(outliers)
    
    p = figure(
        title="Whale Trades", 
        x_axis_label='Entry Date', 
        y_axis_label='Strike Price', 
        x_axis_type='datetime', 
        width=800, 
        height=400, 
        tools="tap", 
        active_tap="tap")
    
    scatter_renderer = p.scatter(
        x='Entry Date', 
        y='Strike Price', 
        size='marker_size', 
        color='red', 
        alpha='marker_opacity', 
        source=source)

    # Add a JavaScript callback to get data from selected points in the scatter plot
    source.selected.js_on_change(
        "indices", 
        CustomJS(args=dict(source=source), code="""
        const indices = cb_obj.indices;
        const data = source.data;
        const selected_data = indices.map(i => {
            return {
                entry_date: data['Entry Date'][i],
                strike_price: data['Strike Price'][i],
                size: data['marker_size'][i],
            };
        });
        console.log("Selected data:", selected_data);
    """))

    output_file("bokeh_plot.html")
    save(p)
    show(p)

# Example usage
df = fetch.load_market_trades()
plot_identified_whale_trades(df)