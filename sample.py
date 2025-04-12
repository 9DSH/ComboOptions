import plotly.graph_objects as go

def plot_hourly_activity_area(hourly_activity):
    # Extract values for the area chart
    hours = hourly_activity["Hour"]
    total_size_sum = hourly_activity["Total Size sum"]

    # Identify the most active hour
    max_size = max(total_size_sum)
    max_hour_index = total_size_sum.index(max_size)
    max_hour = hours[max_hour_index]

    # Create an area chart
    fig = go.Figure()

    # Add trace for the area chart
    fig.add_trace(go.Scatter(
        x=hours,
        y=total_size_sum,
        fill='tozeroy',  # Fill to the x-axis
        mode='lines+markers',
        line=dict(color='blue'),
        name='Total Size by Hour',
        hoverinfo='x+y',  # Show hour and total size on hover
        hovertext=[f"Hour: {hour}<br>Total Size: {size}" for hour, size in zip(hours, total_size_sum)]
    ))

    # Highlight the most active hour
    fig.add_trace(go.Scatter(
        x=[max_hour],
        y=[max_size],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Most Active Hour',
        hoverinfo='x+y',
        hovertext=f"Most Active Hour: {max_hour}<br>Total Size: {max_size}"
    ))

    # Update the layout of the area chart
    fig.update_layout(
        title='Total Size by Hour',
        xaxis=dict(title='Hour'),
        yaxis=dict(title='Total Size'),
        showlegend=True
    )

    return fig

# Sample data for testing
hourly_activity = {
    "Hour": [16, 17, 21, 14, 13, 5, 8, 18, 4, 20, 15, 10, 11, 3, 9, 1, 2, 12, 23, 7, 6],
    "Total Size sum": [1524.1, 1065.7, 1000.0, 952.0, 910.0, 870.0, 725.0, 654.4, 500.1, 400.0, 372.0, 350.0, 325.0, 300.0, 275.0, 100.0, 80.0, 50.0, 50.0, 25.0, 25.0]
}

# Plot the area chart
fig = plot_hourly_activity_area(hourly_activity)
fig.show()