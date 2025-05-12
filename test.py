import os
import pandas as pd
import plotly.graph_objects as go

# Load the CSV files
file_4h_path = os.path.join(os.getcwd(), 'technical_analysis_4h.csv')
file_daily_path = os.path.join(os.getcwd(), 'technical_analysis_daily.csv')

file_4h = pd.read_csv(file_4h_path)
file_daily = pd.read_csv(file_daily_path)

# Process the 4-hour data
file_4h['date'] = pd.to_datetime(file_4h['date'])
file_4h['trend_value'] = 0
file_4h.loc[file_4h['predicted_trend'] == 'Bullish', 'trend_value'] = 1
file_4h.loc[file_4h['predicted_trend'] == 'Bearish', 'trend_value'] = -1
file_4h.loc[file_4h['predicted_trend'] == 'Neutral', 'trend_value'] = 0
file_4h['cumulative_trend'] = file_4h['trend_value'].cumsum()

# Process the daily data
file_daily['date'] = pd.to_datetime(file_daily['date'])
file_daily['trend_value'] = 0
file_daily.loc[file_daily['predicted_trend'] == 'Bullish', 'trend_value'] = 1
file_daily.loc[file_daily['predicted_trend'] == 'Bearish', 'trend_value'] = -1
file_daily.loc[file_daily['predicted_trend'] == 'Neutral', 'trend_value'] = 0
file_daily['cumulative_trend'] = file_daily['trend_value'].cumsum()

# Create the Plotly figure
fig = go.Figure()

# Add 4H data to the plot
fig.add_trace(go.Scatter(x=file_4h['date'], 
                         y=file_4h['cumulative_trend'], 
                         mode='lines+markers',
                         name='Cumulative Trend - 4H'))

# Add Daily data to the plot
fig.add_trace(go.Scatter(x=file_daily['date'], 
                         y=file_daily['cumulative_trend'], 
                         mode='lines+markers',
                         name='Cumulative Trend - Daily'))

# Update layout
fig.update_layout(title='Cumulative Trend Over Time',
                  xaxis_title='Date',
                  yaxis_title='Cumulative Trend Value',
                  xaxis=dict(title='Date', tickformat='%Y-%m-%d'),
                  yaxis=dict(title='Cumulative Trend Value'),
                  legend=dict(x=0, y=1))

fig.show()