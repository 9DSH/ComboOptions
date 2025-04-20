import streamlit as st
import json
import pandas as pd
import re
from openai import OpenAI  # Instead of import openai
from Fetch_data import Fetching_data
from Calculations import get_most_traded_instruments

# Fetching data instance
fetch_data = Fetching_data()

# --- Configuration ---
OPENAI_API_KEY = "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"  # Replace with your OpenAI key
OPENAI_MODEL = "gpt-3.5-turbo"  # Free tier model

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Core Functions ---
def load_market_trades():
    """Load and return market trades as a DataFrame."""
    try:
        df = fetch_data.load_market_trades()
        if isinstance(df, tuple):
            raise ValueError("Expected a DataFrame but got a tuple.")
        
        # Now debug the output of get_most_traded_instruments
        df_most_traded, _ = get_most_traded_instruments(df)
        if not isinstance(df_most_traded, pd.DataFrame):  
            raise ValueError("Expected a DataFrame from get_most_traded_instruments.")
        
        return df_most_traded
    except Exception as e:
        st.error(f"Error loading market trades: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

def call_openai(messages):
    """Call OpenAI API with function calling."""
    params = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000,
    }
    
    response = client.chat.completions.create(**params)
    return response

# --- Streamlit UI ---
st.set_page_config(page_title="Deribit Options AI", layout="wide")
st.title("🤖 Deribit Options Analyst")

# Session state for market trades
if "market_trades_df" not in st.session_state:
    st.session_state.market_trades_df = load_market_trades()  # Load trades initially

# Initialize chat in session state
if "last_user_message" not in st.session_state:
    st.session_state.last_user_message = ""  # Store last user input
if "last_ai_message" not in st.session_state:
    st.session_state.last_ai_message = "Ask me about Deribit options analytics or market trades:"  # Initial assistant prompt

# Sidebar for chat interface
with st.sidebar:
    st.subheader("Chat")
    
    # Create an input field at the top of the sidebar
    prompt = st.text_input("Your question...", "", key="chat_input")

    # Create a container for chat messages
    chat_message = st.empty()  # This will be the container to display the latest AI response

    # Display the last assistant message
    with chat_message:
        st.chat_message("assistant").write(st.session_state.last_ai_message)

    # Handle the user input
    if prompt:
        st.session_state.last_user_message = prompt  # Update last user message
        
        # Prepare the messages for OpenAI API
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Process user input
        try:
            trades_df = st.session_state.market_trades_df

            # Check for specific queries about the most traded option
            if "most" in prompt.lower() or "latest" in prompt.lower() or "last" in prompt.lower():
                if not trades_df.empty:
                    most_traded = trades_df.loc[trades_df['Size'].idxmax()]
                    response_message = (
                        f"The most traded option is {most_traded['Instrument']} with a total size of {most_traded['Size']}.\n"
                        f"BUY Volume: {most_traded['BUY']}, SELL Volume: {most_traded['SELL']}."
                    )
                    st.session_state.last_ai_message = response_message  # Update last AI message
                    chat_message.empty()  # Clear the previous message
                    with chat_message:
                        st.chat_message("assistant").write(response_message)

                else:
                    response_message = "Market trades data is not available."
                    st.session_state.last_ai_message = response_message  # Update last AI message
                    chat_message.empty()  # Clear the previous message
                    with chat_message:
                        st.chat_message("assistant").write(response_message)

            # Additional queries based on other aspects of the DataFrame
            elif "total" in prompt.lower() or "buy" in prompt.lower() or "volume" in prompt.lower():
                if not trades_df.empty:
                    total_buy_volume = trades_df['BUY'].sum()
                    response_message = f"The total buy volume for all options is {total_buy_volume}."
                    st.session_state.last_ai_message = response_message  # Update last AI message
                    chat_message.empty()  # Clear the previous message
                    with chat_message:
                        st.chat_message("assistant").write(response_message)

                else:
                    response_message = "Market trades data is not available."
                    st.session_state.last_ai_message = response_message  # Update last AI message
                    chat_message.empty()  # Clear the previous message
                    with chat_message:
                        st.chat_message("assistant").write(response_message)

            elif "sell" in prompt.lower():
                if not trades_df.empty:
                    total_sell_volume = trades_df['SELL'].sum()
                    response_message = f"The total sell volume for all options is {total_sell_volume}."
                    st.session_state.last_ai_message = response_message  # Update last AI message
                    chat_message.empty()  # Clear the previous message
                    with chat_message:
                        st.chat_message("assistant").write(response_message)

                else:
                    response_message = "Market trades data is not available."
                    st.session_state.last_ai_message = response_message  # Update last AI message
                    chat_message.empty()  # Clear the previous message
                    with chat_message:
                        st.chat_message("assistant").write(response_message)
            
            else:
                # Call OpenAI API for other questions
                response = call_openai(messages)

                if not response.choices:
                    response_message = "No response from OpenAI."
                    st.session_state.last_ai_message = response_message  # Update last AI message
                    chat_message.empty()  # Clear the previous message
                    with chat_message:
                        st.chat_message("assistant").write(response_message)
                else:
                    ai_message = response.choices[0].message.content
                    st.session_state.last_ai_message = ai_message  # Update last AI message
                    chat_message.empty()  # Clear the previous message
                    with chat_message:
                        st.chat_message("assistant").write(ai_message)
            
        except Exception as e:
            response_message = f"Error: {str(e)}"
            st.session_state.last_ai_message = response_message  # Update last AI message
            chat_message.empty()  # Clear the previous message
            with chat_message:
                st.chat_message("assistant").write(response_message)

# Main area for additional outputs if required
st.subheader("Market Trades Data")
if not st.session_state.market_trades_df.empty:
    st.dataframe(st.session_state.market_trades_df)  # Display loaded trades