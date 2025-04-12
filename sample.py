from Fetch_data import Fetching_data
from datetime import datetime, timedelta

fetch_data = Fetching_data()

# Set the default date to tomorrow's date as a datetime.date object
default_date = (datetime.now() + timedelta(days=1)).date()

# Fetch available dates and set the selected date to the default if available
available_dates = fetch_data.fetch_available_dates()
selected_date = default_date if default_date in available_dates else available_dates[0]

print(default_date)  # return format is datetime.date(2025, 4, 13)