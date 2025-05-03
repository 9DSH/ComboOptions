import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import subprocess
import threading
from datetime import date, datetime, timedelta , timezone
import logging
import warnings
from Fetch_data import Fetching_data
from Analytics import Analytic_processing
from Calculations import calculate_option_profit , calculate_totals_for_options, get_most_traded_instruments , calculate_sums_of_public_trades_profit
from Charts import plot_hourly_activity ,plot_most_strategy_bar_chart , plot_top_strikes_pie_chart , plot_strike_price_vs_entry_value , plot_stacked_calls_puts, plot_option_profit , plot_radar_chart, plot_price_vs_entry_date, plot_most_traded_instruments , plot_underlying_price_vs_entry_value , plot_identified_whale_trades, plot_public_profits
from Start_fetching_data import start_fetching_data_from_api,  get_btcusd_price
import plotly.graph_objects as go 
from AI import Chatbar

warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title='Trading Dashboard', layout='wide')



OpenAI_KEY = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

fetch_data = Fetching_data()
analytics = Analytic_processing()
chat = Chatbar(openai_api_key=OpenAI_KEY )


# Initialize the thread reference globally
data_refresh_thread = None
public_trades_thread = None


if 'data_refresh_thread' not in st.session_state:
    st.session_state.data_refresh_thread = None



def start_data_refresh_thread():
    if st.session_state.data_refresh_thread is None or not st.session_state.data_refresh_thread.is_alive():
        st.session_state.data_refresh_thread = threading.Thread(target=start_fetching_data_from_api)
        st.session_state.data_refresh_thread.start()  # Start the background thread


def app():
    start_data_refresh_thread()
    chat.display_chat()
    #disabled_refresh = False
        #disabled_refresh = True if st.session_state.data_refresh_thread is not None else False
        #data_c1 , data_c2 = st.columns(2)
        #with data_c1: 
            #st.markdown(f"<p style='font-size: 14px;; margin-top: 7px;'></p>", unsafe_allow_html=True) 
            #if st.button("Refresh Data", disabled=disabled_refresh):
            #    start_data_refresh_thread() 
       # with data_c2: 
          #  st.markdown(f"<p style='font-size: 14px;'>Only press button once, app will refresh data automatically.</p>", unsafe_allow_html=True) 

       # st.markdown("---")

    
    if 'most_profitable_df' not in st.session_state:
        st.session_state.most_profitable_df = pd.DataFrame()  # Initialize with an empty DataFrame

    combo_breakeven_sell = None
    combo_breakeven_buy = None
    title_row = st.container()
    # Fetch and display the current price
    btc_price , highest, lowest = get_btcusd_price()
    with title_row:
        col1, col2, col3 = st.columns([1, 1, 1])  # Adjust ratio for centering
        with col1:
            show_24h_public_trades = st.checkbox("Show 24h Public Trades", value=True)
            
        with col2:
            colmm1, colmm2 , colmm3= st.columns([1,1,1])
            with colmm1:
                st.markdown(f"<div style='font-size: 12px; margin-left: 80px;'>Lowest</div><div style='font-size: 16px;color: #f54b4b; margin-left: 80px;'>{lowest}</div>", unsafe_allow_html=True)
            with colmm2:
                btc_display_price = f"{btc_price:.0f}" if btc_price is not None else "Loading..."
                st.metric(label="BTC USD", value=btc_display_price, delta=None, delta_color="normal", help="Bitcoin price in USD ")
            with colmm3:
                st.markdown(f"<div style='font-size: 12px; margin-left: -50px;'>Highest</div><div style='font-size: 16px; color: #90EE90; margin-left: -50px;'>{highest}</div>", unsafe_allow_html=True)
            
                
        with col3:
            colmm1, colmm2 , colmm3= st.columns([2,2,1])
            with colmm3:
                current_utc_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                st.markdown(f"<div style='font-size: 12px; '>UTC Time</div><div style='font-size: 16px; color: gray;'>{current_utc_time}</div>", unsafe_allow_html=True)
            
            

                




    # Create columns for the layout
    colm1, colm2, colm3 = st.columns(3)

    # Populate each column with metrics using HTML for reduced size
    with colm1:
        st.write("")

    with colm2:
        st.write("")
    with colm3:
        st.write("")

    # Initialize data fetching at the start of the app
    # Button to refresh data
    

    premium_buy = 0
    premium_sell = 0

    currency = 'BTC'
        #current_date_initials = pd.to_datetime(datetime.now()).date()

        
        # Initialize session state for inputs if they don't exist
    if 'selected_date' not in st.session_state:
        default_date = (datetime.now() + timedelta(days=1)).date()
        st.session_state.selected_date = default_date
    if 'option_symbol' not in st.session_state:
            st.session_state.option_symbol = None  # Initialize this as None or an empty value
    if 'quantity' not in st.session_state:
            st.session_state.quantity = 0.1  # Default quantity
    if 'option_side' not in st.session_state:
            st.session_state.option_side = "BUY"  # Default side
        




    ##------------------------------------------------------
    ##---------------------- MAIN TABS ------------------------
    #-------------------------------------------------------
        

#---------------------------------------------------------------
#-----------------------Market Watch ---------------------------
#-------------------------------------------------------------
    main_tabs = st.tabs(["Market Watch",  "Trade Option", "Hedge Fund"])
    with main_tabs[0]: 
             # Initialize trades variable outside of any if conditions
            market_screener_df = fetch_data.load_market_trades( filter=None , drop = False ,show_24h_public_trades = show_24h_public_trades)
            total_number_public_trade = market_screener_df.shape[0]
            
            # here we can have the function that simulates the public trades profit

            if not market_screener_df.empty:
                market_screener_df.dropna(subset=['Entry Date', 'Underlying Price'], inplace=True)
                filter_row = st.container()
                with filter_row:
                    col_date,col_vertical_1, col_strike_size_range, col_vertical_2,  col_expiration, col_vertical_3, col_side_type = st.columns([0.3, 0.01, 0.3, 0.01, 0.5,0.01, 0.15])
                    #with col_refresh: 
                        #apply_market_filter = st.button(label="Apply", key="apply_market_filter")

                    with col_date:
                        date_row1 = st.container()
                        date_row2 = st.container()

                        with date_row1:
                            cc1, cc2, cc3 = st.columns([0.04, 0.02, 0.02])
                            with cc1:
                                start_date = st.date_input("Start Entry Date", value=date(2025, 3, 1))
                            with cc2:
                                start_hour = st.number_input("Hour", min_value=0, max_value=23, value=0)
                            with cc3:
                                start_minute = st.number_input("Minute", min_value=0, max_value=59, value=0)

                        with date_row2:
                            ca1, ca2, ca3 = st.columns([0.04, 0.02, 0.02])
                            with ca1:
                                current_utc_date = datetime.now(timezone.utc).date()
                                end_date = st.date_input("End Entry Date", value=current_utc_date)

                                with ca2:
                                    end_hour = st.number_input("Hour", min_value=0, max_value=23, value=23)
                                with ca3:
                                    end_minute = st.number_input("Minute", min_value=0, max_value=59, value=59)

                            # Combine date and time into a single datetime object
                        start_datetime = datetime.combine(start_date, datetime.min.time().replace(hour=start_hour, minute=start_minute))
                        end_datetime = datetime.combine(end_date, datetime.min.time().replace(hour=end_hour, minute=end_minute))
                    with col_vertical_1:
                            st.markdown("<div style='height: 150px; width: 1px; background-color: gray; margin: auto;'></div>", unsafe_allow_html=True)  # Vertical line

                    with col_strike_size_range:
                        row_one = st.container()
                        row_two = st.container()
                        with row_one:
                            strike_col1, strike_col2 = st.columns(2)
                            with strike_col1:
                                min_strike = st.number_input("Minimum strike", min_value=0, max_value=400000, value=60000)
                            with strike_col2:
                                max_strike = st.number_input("Maximum strike", min_value=0, max_value=400000, value=120000)
                        with row_two:
                            size_col1, size_col2 = st.columns(2)
                            with size_col1:
                                min_size= st.number_input("Minimum size", min_value=0.1, max_value=500.0, value=0.1)
                            with size_col2:
                                max_size = st.number_input("Maximum size", min_value=0.1, max_value=500.0, value=500.0)
                            size_range = (min_size, max_size)

                    with col_vertical_2:
                        st.markdown("<div style='height: 150px; width: 1px; background-color: gray; margin: auto;'></div>", unsafe_allow_html=True)  # Vertical line
                    with col_expiration:
                        row_expiration = st.container()
                        row_strike = st.container()
                        with row_expiration:
                            market_available_dates = market_screener_df['Expiration Date'].dropna().unique().tolist()

                            # Convert to datetime to sort
                            market_available_dates = pd.to_datetime(market_available_dates, format='%d-%b-%y', errors='coerce')
                            
                            # Filter out NaT values
                            market_available_dates = market_available_dates.dropna()
                            # Sort the dates
                            sorted_market_available_dates = sorted(market_available_dates)

                            # Optionally convert back to desired string format for display purposes
                            sorted_market_available_dates = [date.strftime("%#d-%b-%y") for date in sorted_market_available_dates]

                            selected_expiration_filter = st.multiselect("Filter by Expiration Date", sorted_market_available_dates, key="whatch_exp_filter")

                        with row_strike:
                            
                            strike_range = (min_strike, max_strike)
                            if 'Strike Price' in market_screener_df.columns and not market_screener_df.empty:
                                # Filter the DataFrame for strikes within the selected range
                                filtered_strikes_df = market_screener_df[
                                    (market_screener_df['Strike Price'] >= strike_range[0]) &
                                    (market_screener_df['Strike Price'] <= strike_range[1])
                                ]
                                unique_strikes = filtered_strikes_df['Strike Price'].unique()
                                sorted_strikes = sorted(unique_strikes, reverse=True)  # Sort in descending order

                                # Create the multiselect for the filtered strike prices
                                multi_strike_filter = st.multiselect("Filter by Strike Price", options=sorted_strikes)
                            else:
                                # Handle case where no strikes are available
                                multi_strike_filter = st.multiselect("Filter by Strike Price", options=[], default=[], help="No available strikes to select.")
                    
                    with col_vertical_3:
                            st.markdown("<div style='height: 150px; width: 1px; background-color: gray; margin: auto;'></div>", unsafe_allow_html=True)  # Vertical line

                    with col_side_type:
                            
                            side_row = st.container()
                            type_row = st.container()
                            with side_row:
                                show_sides_buy = st.checkbox("BUY", value=True, key='show_buys')
                                show_sides_sell = st.checkbox("SELL", value=True, key='show_sells')
                            with type_row:
                                show_type_call = st.checkbox("Call", value=True, key='show_calss')
                                show_type_put = st.checkbox("Put", value=True, key='show_puts')

                    start_strike, end_strike = strike_range  # Unpack the tuple to get start and end values
                    start_size , end_size = size_range
                    # Initial filtering by strike price and date range
                    filtered_df = market_screener_df[
                        (market_screener_df['Size'] >= start_size) &
                        (market_screener_df['Size'] <= end_size) &
                        (market_screener_df['Strike Price'] >= start_strike) &
                        (market_screener_df['Strike Price'] <= end_strike) &
                        (market_screener_df['Entry Date'] >= start_datetime) &
                        (market_screener_df['Entry Date'] <= end_datetime)
                    ]

                    if selected_expiration_filter:
                        filtered_df = filtered_df[(filtered_df['Expiration Date'].isin(selected_expiration_filter))]

                    # Filter by selected strikes
                    if multi_strike_filter:
                        filtered_df = filtered_df[filtered_df['Strike Price'].isin(multi_strike_filter)]

                    # Apply filtering for buy/sell sides
                    sides_to_filter = []
                    if show_sides_buy:
                        sides_to_filter.append('BUY')  # append the actual value as per your column data
                    if show_sides_sell:
                        sides_to_filter.append('SELL')  # append the actual value as per your column data

                    if sides_to_filter:
                        filtered_df = filtered_df[filtered_df['Side'].isin(sides_to_filter)]

                    # Apply filtering for call/put types
                    types_to_filter = []
                    if show_type_call:
                        types_to_filter.append('Call')  # append the actual value as per your column data
                    if show_type_put:
                        types_to_filter.append('Put')  # append the actual value as per your column data

                    if types_to_filter:
                        filtered_df = filtered_df[filtered_df['Option Type'].isin(types_to_filter)]
                        
            if not market_screener_df.empty:
                # Ensure 'Entry Date' is in datetime format
                market_screener_df['Entry Date'] = pd.to_datetime(market_screener_df['Entry Date'], errors='coerce')

                # Check for any NaT values
                if market_screener_df['Entry Date'].isna().any():
                    st.warning("Some entries in the 'Entry Date' column were invalid and have been set to NaT.")
                
                
                tabs = st.tabs(["Insights",  "Top Options", "Public Trade Strategies", "Whales" , "Data table"])

                with tabs[0]:
                    details_row =st.container()
                    with details_row : 
                        col1,col2,col3,col4,col5 = st.columns([0.2,0.1,0.1,0.1,0.2])
                        with col2:
                            total_options, total_amount, total_entry_values = calculate_totals_for_options(filtered_df)
                            total_trades_percentage = ( total_options/ total_number_public_trade) * 100

                            row_count_title = st.container()
                            row_count = st.container()
                            with row_count_title:
                                st.markdown(f"<p style='font-size: 12px; color: gray;'> Total Counts:</p>", unsafe_allow_html=True)
                            with row_count:
                                st.markdown(f"<p style='font-size: 17px; font-weight: bold;'> {total_options:,}</p>", unsafe_allow_html=True)
                            
                        with col3:
                            
                            row_count_title = st.container()
                            row_count = st.container()
                            with row_count_title:
                                st.markdown(f"<p style='font-size: 12px; color: gray;'> Percentage of Total Trades:</p>", unsafe_allow_html=True)
                            with row_count:
                                st.markdown(f"<p style='font-size: 17px; font-weight: bold;'> {total_trades_percentage:.1f}%</p>", unsafe_allow_html=True)

                        with col4:
                            
                            row_size_title = st.container()
                            row_size = st.container()
                            with row_size_title:
                                st.markdown(f"<p style='font-size: 12px; color: gray;'> Total Values:</p>", unsafe_allow_html=True)
                            with row_size:
                                st.markdown(f"<p style='font-size: 17px;font-weight: bold;'> {total_entry_values:,.0f}</p>", unsafe_allow_html=True)

                    detail_column_2, detail_column_3 = st.columns(2)                       

                    with detail_column_2:
                        fig_2 = plot_strike_price_vs_entry_value(filtered_df)
                        st.plotly_chart(fig_2)
                    with detail_column_3:
                        fig_3 = plot_stacked_calls_puts(filtered_df)
                        st.plotly_chart(fig_3)
                    st.markdown("---") 

                with tabs[1]:  
                    padding, cal1, cal2,cal3 = st.columns([0.02, 0.7, 0.01 ,0.6])

                    with padding: 
                        st.write("")

                    with cal1: 

                        most_traded_options , top_options_chains = get_most_traded_instruments(filtered_df)
                        fig_pie = plot_most_traded_instruments(most_traded_options)
                        st.plotly_chart(fig_pie)

                    with cal3:
                        
                        st.markdown(f"<p style='font-size: 14px; margin-top: 28px;'></p>", unsafe_allow_html=True) 
                        st.dataframe(top_options_chains, use_container_width=True, hide_index=True)
                      
                    

             #------------------------------------------
             #       public trades insights
             #-----------------------------------------------

                with tabs[2]:
                    target_columns = ['BlockTrade IDs', 'BlockTrade Count', 'Combo ID', 'ComboTrade IDs']
                    filtered_df = filtered_df.drop('hover_text', axis=1)
                    
                    filtered_startegy = filtered_df.copy()
                    # Separate strategy trades
                    strategy_trades_df = filtered_startegy[~filtered_startegy[target_columns].isna().all(axis=1)]
                    print(f'Strategy found : {strategy_trades_df.shape[0]}')

                    if not strategy_trades_df.empty:

                        # Group by BlockTrade IDs and Combo ID to identify unique strategies
                        strategy_groups = strategy_trades_df.groupby(['BlockTrade IDs', 'Combo ID'])
                        
                        # Create subtabs for different views
                        strategy_subtabs = st.tabs(["Strategy Overview", "Strategy Details"])
                        with strategy_subtabs[0]:
                            # Summary statistics for each strategy
                            strategy_df = analytics.Identify_combo_strategies(strategy_groups)
                            strategy_df_copy = strategy_df.copy()
                            if not strategy_df_copy.empty:
                                    
                                insights = analytics.analyze_block_trades(strategy_df_copy)
                                
                                st.caption(f"Analyzed {len(strategy_df_copy)} trades from {insights['summary_stats']['time_range_start']} to {insights['summary_stats']['time_range_end']}")

                                # 1. Key Metrics
                                padding , col1, col2, col3, col4, padding2 = st.columns([1,1,1,1,1,1])
                                col1.metric("Total Strategies", f"{len(strategy_df_copy)}")
                                col2.metric("Total Volume (BTC)", f"{insights['summary_stats']['total_size_btc']:,.1f}")
                                col3.metric("Average Trade Size", f"{insights['summary_stats']['avg_trade_size']:,.1f} BTC")
                                most_active_strategy = insights['strategy_analysis']['top_strategies'].index[0]
                                col4.metric("Most Active Strategy", str(most_active_strategy))
                                st.markdown("---")  # Horizontal line

                                # 2. Strategy Distribution
                                
                                colu1, colu2 = st.columns(2)
                                strategy_df_copy = insights['strategy_analysis']['strategy_distribution']
                                with colu1 : 
                                    most_strag_fig = plot_most_strategy_bar_chart(strategy_df_copy)
                                    st.plotly_chart( most_strag_fig)
                                with colu2 : 
                                    # 3. Top Strikes
                                    top_strikes = insights['strike_analysis']['top_strikes']
                                    fig_startegy_top_strikes = plot_top_strikes_pie_chart(top_strikes)
                                    st.plotly_chart(fig_startegy_top_strikes)

                                # 4. Time Analysis
                                hourly = insights['time_analysis']['hourly_activity']
                                fig_hourly = plot_hourly_activity(hourly)
                                st.plotly_chart(fig_hourly)
                            
                                # 5. Recommendations
                                st.subheader("💡 Trader Insights")
                                for rec in insights['recommendations']:
                                    st.info(rec)

                                # Raw data expander
                                with st.expander("📝 View Raw Analysis Data"):
                                    st.dataframe(strategy_df, use_container_width=True, hide_index=True)
                                                            
                        
                        with strategy_subtabs[1]:
                            # Detailed view of each strategy
                            # Convert strategy groups to list and sort by total premium
                            sorted_strategies = []
                            for (block_id, combo_id), group in strategy_groups:
                                total_premium = group['Entry Value'].sum()
                                strategy_type = strategy_df[strategy_df['Strategy ID'] == combo_id]['Strategy Type'].iloc[0]
                                sorted_strategies.append((block_id, combo_id, group, total_premium, strategy_type))
                            
                            # Sort by total premium in descending order
                            sorted_strategies.sort(key=lambda x: x[3], reverse=True)

                            # Create a list of strategy labels for the selectbox
                            def format_value(value):
                                if value > 1000000:
                                    return f"{value/1000000:.0f}M"
                                elif value > 1000:
                                    return f"{value/1000:.0f}k"
                                else:
                                    return f"{value:,.0f}"

                            strategy_labels = []
                            for _, _, group, total_premium, strategy_type in sorted_strategies:
                                option_details = " | ".join(
                                    f"{row['Side']}-{row['Option Type']}-{int(row['Strike Price'])}-{format_value(row['Entry Value'])}"
                                    for _, row in group.iterrows()
                                )
                                strategy_labels.append(f"{format_value(total_premium)} -- {option_details}")

                            # Use a selectbox to choose a strategy
                            selected_strategy_label = st.selectbox("Select a Strategy", strategy_labels)
                            
                            # Find the selected strategy details
                            selected_strategy = next((s for s in sorted_strategies if f"{format_value(s[3])} -- " + " | ".join(
                                f"{row['Side']}-{row['Option Type']}-{int(row['Strike Price'])}-{format_value(row['Entry Value'])}"
                                for _, row in s[2].iterrows()
                            ) == selected_strategy_label), None)
                            
                            if selected_strategy:
                                block_id, combo_id, group, total_premium, strategy_type = selected_strategy
                                
                                # Calculate strategy metrics
                                total_size = group['Size'].sum()
                                # Display strategy metrics
                                col1, col2, col3, col4 = st.columns([0.3, 0.2, 0.5, 0.2])
                                with col1:
                                    st.metric("Total Premium", f"${format_value(total_premium)}")
                                with col2:
                                    st.metric("Total Size", f"{total_size:,.0f}")
                                with col3:
                                    strategy_type_from_summary = strategy_df[strategy_df['Strategy ID'] == combo_id]['Strategy Type'].iloc[0]
                                    st.metric("Strategy Type", strategy_type_from_summary)
                                with col4:
                                    st.metric("Number of Legs", len(group))
                                
                                # Display strategy components
                                st.dataframe(group, use_container_width=True, hide_index=True)   
                                # Create visualization for this strategy
                                chart_col1, chart_col2 = st.columns(2)
                                strategy_data = group.copy()
                                
                                # Calculate profits using multithreading
                                fig_profit, fig_strategy = plot_public_profits(strategy_data, "Public", trade_option_details=None)
                                
                                with chart_col1:
                                    st.plotly_chart(fig_strategy, use_container_width=True, key=f"strategy_plot_{block_id}_{combo_id}")

                                with chart_col2:
                                    st.plotly_chart(fig_profit, use_container_width=True, key=f"alldays_plot_{block_id}_{combo_id}")
                              
                    else :               
                        st.warning("No strategy trades found in the current selection.")

                

                with tabs[3]:
                    whale_cal1,whale_cal2, whale_cal3 = st.columns([0.5,0.5,1])
                    with whale_cal1:
                        whale_filter_type = st.selectbox("Analyze values by:", options=['Size', 'Entry Value'], index=1)

                    with whale_cal2:
                        if whale_filter_type == "Entry Value" :
                            entry_filter = st.number_input("Set Entry Filter Value", min_value=0, value=10000, step=100)
                        else : entry_filter = st.number_input("Set Size Filter Value", min_value=0.1, value=2.0, step=0.1)
                        

                    outliers , whales_fig = plot_identified_whale_trades(filtered_df, min_size=8, max_size=35, min_opacity=0.2, max_opacity=0.8, entry_value_threshold = entry_filter , filter_type = whale_filter_type )
                    st.plotly_chart(whales_fig)

                    #st.markdown("---")

                    with st.expander("View the Data Table for these whales", expanded=True): 
                        st.dataframe(outliers , use_container_width=True, hide_index=True)  # Show index


                
                with tabs[4]:
                    datatable = st.tabs(["Processed Data" , "Raw Data"])
                    with datatable[0]:
                        
                        processed_df = filtered_df[filtered_df[target_columns].isna().all(axis=1)]
                        processed_df = processed_df.iloc[:, :-5]
                        processed_df = processed_df.sort_values(by='Entry Date', ascending=False)  # Sort by entry date
                        st.dataframe(processed_df, use_container_width=True, hide_index=False)  # Show index

                        
                        st.markdown("---")  # Horizontal line
                        analyze_row = st.container()
                        with analyze_row:
                            analyze_col1, analyze_col2 = st.columns([0.2,1])
                            with analyze_col1:
                                selected_index = st.selectbox("Select an Index to Analyze", options=processed_df.index, key="analyze_profit_select")
                            with analyze_col2:
                                if selected_index is not None:
                                        # Filter the raw data for the selected index
                                        selected_option_data = filtered_df.loc[[selected_index]]

                                        # Calculate profits using the existing function
                                        fig_public_profit, fig_public_strategy = plot_public_profits(selected_option_data, "Public", trade_option_details=None)

                                        instrument = selected_option_data['Instrument'].values[0]
                                        entry_value = selected_option_data['Entry Value'].values[0]
                                        expiration_date = selected_option_data['Expiration Date'].values[0]
                                        side = selected_option_data['Side'].values[0]
                                        size = selected_option_data['Size'].values[0]
                                        underlying_price = selected_option_data['Underlying Price'].values[0]
                                        
                                        entry_date = pd.to_datetime(selected_option_data['Entry Date'].values[0]).strftime('%d-%b-%y %H:%M')
                                        st.markdown(
                                            f"""
                                            <div style='display: flex; justify-content: flex-start; gap: 10px; padding-top: 10px;'>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: smaller; color: gray;'>Instrument</div>
                                                    {instrument}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: smaller; color: gray;'>Side</div>
                                                    {side}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: smaller; color: gray;'>Size</div>
                                                    {size}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: smaller; color: gray;'>Entry Value</div>
                                                    {entry_value}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: smaller; color: gray;'>Entry Date</div>
                                                    {entry_date}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: smaller; color: gray;'>Expiration Date</div>
                                                    {expiration_date}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: smaller; color: gray;'>Underlying Price</div>
                                                    {underlying_price}
                                                </div>
                                            </div>
                                            """, 
                                            unsafe_allow_html=True
                                        )

                        
                        st.markdown("---")  # Horizontal line
                        analyze_col1, analyze_col2 = st.columns(2)
                        # Create a selectbox for the user to choose an index from the processed_df
                       
                                # Display the profit chart
                        with analyze_col1:
                            st.plotly_chart( fig_public_strategy)
                        
                            
                        with analyze_col2:
                            st.plotly_chart(  fig_public_profit)


                    with datatable[1]:
                        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

                        
          
   
            else:
                st.warning("No trades available for the selected options, wait while app is fetching data...")

#------------------------------------------------------------------------------------
#----------------------------------------Trade an Option -------------------------------------
#--------------------------------------------------------------------------------------

    with main_tabs[1]:
            #--------------------------------------------------------------------------------
            #-------------------------------- Poltting Results -----------------------------
            #--------------------------------------------------------------------------------
            available_dates = fetch_data.fetch_available_dates()
            options_row = st.container()
            trade_option_detail = []
            with options_row:
                option_col1, option_col2, vertical_line , option_details= st.columns([1,0.5,0.1,4])
                with option_col1:
                    row_1 = st.container()
                    row_2 = st.container()
                    with row_1 : 
                    
                        selected_date = st.selectbox("Select Expiration Date", available_dates, 
                                                            index=available_dates.index(st.session_state.selected_date))
                        st.session_state.selected_date = selected_date  # Update session state

                    with row_2 : 

                        if selected_date:
                                # Fetch options available for the selected date
                            options_for_date = fetch_data.get_options_for_date(currency, selected_date)
                                
                                # Allow user to select an option
                            if options_for_date:
                                    # Check if the currently selected option_symbol is in the available options
                                if st.session_state.option_symbol not in options_for_date:
                                        # Reset option_symbol to None if it's not available
                                    st.session_state.option_symbol = None

                                option_symbol_index = 0 if st.session_state.option_symbol is None else options_for_date.index(st.session_state.option_symbol)
                                option_symbol = st.selectbox("Select Option", options_for_date, index=option_symbol_index)

                with option_col2:
                    row_1 = st.container()
                    row_2 = st.container()
                    # Use st.session_state.quantity directly in the number_input
                    with row_1:
                        quantity = st.number_input('Quantity',
                                                    min_value=0.1,
                                                    step=0.1,
                                                    value=st.session_state.quantity)  # Current value from session state
                        # Update session state only if the value changes
                        if quantity != st.session_state.quantity:
                            st.session_state.quantity = quantity  # Update the session state after the widget is used
                    with row_2:

                        option_side = st.selectbox("Select Side", options=['BUY', 'SELL'], 
                                                    index=['BUY', 'SELL'].index(st.session_state.option_side))
                        st.session_state.option_side = option_side  # Store input in session state

                    trade_option_detail.extend([option_side, quantity])

                #----------------------Vertical Line -----------------------
                with vertical_line:
                    st.markdown("<div style='height: 170px; width: 1px; background-color: gray; margin: auto;'></div>", unsafe_allow_html=True)  # Vertical line
                with option_details:
                    row_1 = st.container()
                    row_2 = st.container()
                    if option_symbol:
                            # Get and display the details of the selected option
                        option_details, option_index_price = fetch_data.fetch_option_data(option_symbol)
                        all_options_with_details = fetch_data.get_all_options(filter=None, type='data')
                        recent_public_trades_df = fetch_data.load_market_trades(filter= option_symbol , show_24h_public_trades = show_24h_public_trades)
                        
                        if not option_details.empty:
                                # Extracting details safely
                                
                            expiration_date_str = option_details['Expiration Date'].values[0]
                            expiration_date = pd.to_datetime(expiration_date_str).date()  # Ensure it's converted to date

                            option_type = option_details['Option Type'].values[0]
                            bid_iv = option_details['Bid IV'].values[0]
                            ask_iv = option_details['Ask IV'].values[0]
                            bid_price = option_details['Bid Price (USD)'].values[0]
                            ask_price = option_details['Ask Price (USD)'].values[0]
                            strike_price = option_details['Strike Price'].values[0]
                            premium_buy = ask_price * quantity
                            premium_sell = bid_price * quantity

                            breakeven_call_buy = premium_buy + strike_price
                            breakeven_call_sell = premium_sell+ strike_price
                            breakeven_put_buy = strike_price - premium_buy
                            breakeven_put_sell = strike_price - premium_sell
     
                                
                            breakeven_buy = breakeven_call_buy if option_type == 'call' else breakeven_put_buy
                            breakeven_sell = breakeven_put_sell if option_type == 'put' else breakeven_call_sell
                                
                            if option_side == "BUY":
                                breakeven_sell = None
                                premium = premium_buy
                                breakeven_price = breakeven_buy 
                                            
                            if option_side == "SELL":
                                breakeven_buy = None
                                premium = premium_sell
                                breakeven_price = breakeven_sell


                            now_utc = datetime.now(timezone.utc).date()

                                # Compute total days to expiration (at least 1 to avoid zero)
                            time_to_expiration_days = max((expiration_date - now_utc).days, 1)

                            fee_cap = 0.125 * premium
                            initial_fee = 0.0003 * btc_price * quantity
                            final_fee_selected_option = min(initial_fee, fee_cap)

                            with row_1: 
                                option_details_copy = option_details.copy().iloc[:, 3:]
                                st.dataframe(option_details_copy, use_container_width=True, hide_index=True)
                            
                            with row_2: 
                                st.markdown(
                                                f"""
                                                <div style='display: flex; justify-content: flex-start; gap: 10px; padding-top: 10px;'>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Instrument</div>
                                                        {option_symbol}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Entry Value</div>
                                                        {premium:.1f}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Fee (USD)</div>
                                                        {final_fee_selected_option:.1f}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Breakeven Price</div>
                                                        {breakeven_price:.0f}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Time To Expiration</div>
                                                        {time_to_expiration_days}
                                                    </div>
                                                </div>
                                                </div>
                                                """, 
                                                unsafe_allow_html=True
                                            )


                        else:
                            st.write("Error fetching option details.")     

            df_options_for_strike = fetch_data.get_all_strike_options(currency, strike_price, option_type)
            
            Other_tab = f'All Other Options for {strike_price:.0f} {option_type.upper()}'
            analytic_tabs = st.tabs(["Analytics", "Recent trades", Other_tab])
            with analytic_tabs[0]:
                if recent_public_trades_df is None or recent_public_trades_df.empty:
                    st.warning("There are no recent trades for the selected option.")
                
                if not recent_public_trades_df.empty:
                    st.subheader(f'Analytics for Option {strike_price:.0f} {option_type.upper()} ')
                    padding, chart_1, chart_2, chart_3 = st.columns([0.1, 2, 2, 2 ])
                    with padding: 
                        st.write("")
                    with chart_1:
                        fig_selected_symbol = plot_underlying_price_vs_entry_value(recent_public_trades_df, btc_price, premium)
                        st.plotly_chart(fig_selected_symbol)
                    with chart_2:
                        # whales_in_option_fig = plot_whales(recent_public_trades_df, min_count=2, min_avg_size=5, max_marker_size=30, showlegend=False)
                        # st.plotly_chart(whales_in_option_fig)
                        price_vs_date = plot_price_vs_entry_date(recent_public_trades_df)
                        st.plotly_chart(price_vs_date)
                    with chart_3:
                        open_by_expiration_radar = plot_radar_chart(df_options_for_strike)
                        st.plotly_chart(open_by_expiration_radar)

            with analytic_tabs[1]:
                 
                 if not recent_public_trades_df.empty:
                    st.subheader(f'Recent Trades for {strike_price:.0f} {option_type.upper()}')
                    st.dataframe(recent_public_trades_df, use_container_width=True, hide_index=False)
                 else :
                     st.warning(f'No Trade History for {strike_price:.0f} {option_type.upper()}')

            with analytic_tabs[2]:
                st.subheader(f'Available Dates for {strike_price:.0f} {option_type.upper()}')
                df_options_for_strike = df_options_for_strike.drop(columns=['Strike Price', 'Option Type'], errors='ignore')
                st.dataframe(df_options_for_strike, use_container_width=True, hide_index=True)


                      
            st.markdown("---")
            profit_fig , expiration_profit = plot_public_profits(option_details , "Trade", trade_option_detail)
            
            chart_col_1, chart_col_2 = st.columns(2)
            with chart_col_1:
                        
                st.plotly_chart(expiration_profit)
            with chart_col_2:
                st.plotly_chart(profit_fig)
                        


#--------------------------------------------------------------
#-----------------------Combinations ----------------------------
#-------------------------------------------------------------
    with main_tabs[2]:

            if st.session_state.most_profitable_df.shape[1] > 1:  # Assuming at least 'Underlying Price' and one profit column exists
                    most_profitable_df = st.session_state.most_profitable_df
                    num_combos = most_profitable_df.shape[1]
                    num_combo_to_show = 10  # Number of columns to show per tab
                    num_tabs = (num_combos // num_combo_to_show) + (num_combos % num_combo_to_show > 0)

                            # Create tab names based on the number of combinations
                    tab_names = [f"Combination {i + 1}" for i in range(num_tabs)]

                            # Create the tabs dynamically
                    combo_tabs = st.tabs(tab_names)

                            # Loop through each tab and display the corresponding data
                    for i, tab in enumerate(combo_tabs):
                        with tab:
                                    # Copy the main DataFrame to avoid modifying original
                            display_df = most_profitable_df.copy()
                                    
                                    # Isolate and remove the 'Underlying Price' column
                            underlying_price = display_df.pop('Underlying Price')

                                    # Determine the columns to display in this tab
                            start_col = i * num_combo_to_show
                            end_col = start_col + num_combo_to_show

                                    # Slice the DataFrame for the current tab.
                            columns_to_display = display_df.columns[start_col:end_col]
                                    
                                    # Create a new DataFrame for display without the 'Underlying Price'
                            sliced_df = display_df[list(columns_to_display)]
                                    
                                    # Insert the 'Underlying Price' column at the front
                            sliced_df.insert(0, 'Underlying Price', underlying_price)

                                    # Render the DataFrame with Streamlit
                            styled_results = style_combined_results(sliced_df)  # Pass the full DataFrame

                                    # Render the styled DataFrame in the tab using markdown
                            st.markdown(styled_results.to_html(escape=False), unsafe_allow_html=True)
                
            else:
                st.warning("No combinations meet the criteria, Press Analyze then Filter button.")
            

     


def style_combined_results(combined_results):
    """
    Apply conditional formatting to the combined results DataFrame with richer color distinctions.

    Parameters:
        combined_results (pd.DataFrame): The DataFrame to apply styles on.

    Returns:
        pd.io.formats.style.Styler: A styled DataFrame for better insights.
    """
    def color_profits(value):
        """
        Color profits based on their values using a gradient:
        - Green for positive profits
        - Yellow for values transitioning around zero
        - Red for negative profits
        """
        if value > 200:
            r = 0  # No red
            g = 255  # Full green
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Bright green

        elif 0 < value <= 200:
            # Gradient from green (at 200) to yellow (at 0)
            r = int((255 * (200 - value)) / 200)  # More red as value decreases
            g = 255  # Green stays full
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Green to yellow gradient

        elif value == 0:
            # Pure yellow for profit of zero
            r = 255  # Full red
            g = 255  # Full green
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Opaque yellow

        elif -100 < value < 0:
            # Gradient from yellow (at 0) to red (at -100)
            r = 255  # Full red
            g = int((255 * (100 + value)) / 100)  # Green decreases as it goes more negative
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Yellow to red gradient

        else:
            # Solid red for values lower than -100
            r = 255  # Full red
            g = 0  # No green
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: white'  # Solid red

    # Create a masking function to skip the first row and the "Premium" row
    def apply_color(row):
        # Check if the row name is "Premium", and skip coloring if it is
        if row.name == 'Premium':
            return [''] * len(row)  # No styling
        else:
            return [color_profits(value) for value in row]

    # Apply the styling using Styler.apply() on all columns except the first one
    styled_df = combined_results.style.apply(apply_color, axis=1, subset=combined_results.columns[1:])  # Skip the 'Underlying Price' column

    # Format numeric values with specific precision
    formatted_dict = {
        'Underlying Price': '{:.0f}',  # 0 decimal for underlying price
    }

    # Apply formatting for the 'Underlying Price' column and 1 decimal for other profit columns
    for col in combined_results.columns[1:]:  # Assuming first two are not profit columns
        formatted_dict[col] = '{:.1f}'  # 1 decimal for profit columns

    # Format the styled DataFrame
    styled_df = styled_df.format(formatted_dict)


    return styled_df

    
      

if __name__ == "__main__":
    
    # Check for our custom environment flag.
    if os.environ.get("STREAMLIT_RUN") != "1":
       os.environ["STREAMLIT_RUN"] = "1"
        # Launch the Streamlit app using the current Python interpreter.
       subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        # We are already running under "streamlit run" – proceed with your app.
        # Start the background thread once (if not already started).
        # Now call your app() function to render the Streamlit interface.
        
        app()
        
    