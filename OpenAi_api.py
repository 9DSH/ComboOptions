import streamlit as st
import pandas as pd
from openai import OpenAI  # Ensure you've imported the correct OpenAI package
from Fetch_data import Fetching_data
from Calculations import get_most_traded_instruments

class Chatbar:
    def __init__(self, openai_api_key, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.fetch_data = Fetching_data()
        self.init_session_state()
    
    def init_session_state(self):
        """Initializes the session state variables."""
        if "market_trades_df" not in st.session_state:
            st.session_state.market_trades_df = self.load_market_trades()
        
        if "last_user_message" not in st.session_state:
            st.session_state.last_user_message = ""
        
        if "last_ai_message" not in st.session_state:
            st.session_state.last_ai_message = "Ask me about Deribit options analytics or market trades:"
    
    def load_market_trades(self):
        """Loads market trades data."""
        try:
            df = self.fetch_data.load_market_trades()
            if isinstance(df, tuple):
                raise ValueError("Expected a DataFrame but got a tuple.")
            df_most_traded, _ = get_most_traded_instruments(df)
            if not isinstance(df_most_traded, pd.DataFrame):  
                raise ValueError("Expected a DataFrame from get_most_traded_instruments.")
            return df_most_traded
        except Exception as e:
            st.error(f"Error loading market trades: {str(e)}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error

    def call_openai(self, messages):
        """Calls the OpenAI API."""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        
        response = self.client.chat.completions.create(**params)
        return response

    def display_chat(self):
        """Handles the chat display and messaging logic."""
        with st.sidebar:
            st.subheader("AI Chat")
            prompt = st.text_input("Your question...", "", key="chat_input")
            chat_message = st.empty()

            with chat_message:
                st.chat_message("assistant").write(st.session_state.last_ai_message)

            if prompt:
                st.session_state.last_user_message = prompt
                
                # Prepare the messages for OpenAI API
                messages = [{"role": "user", "content": prompt}]
                
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
                            st.session_state.last_ai_message = response_message
                            chat_message.empty()
                            with chat_message:
                                st.chat_message("assistant").write(response_message)
                        else:
                            response_message = "Market trades data is not available."
                            st.session_state.last_ai_message = response_message
                            chat_message.empty()
                            with chat_message:
                                st.chat_message("assistant").write(response_message)

                    elif "total" in prompt.lower() or "buy" in prompt.lower() or "volume" in prompt.lower():
                        if not trades_df.empty:
                            total_buy_volume = trades_df['BUY'].sum()
                            response_message = f"The total buy volume for all options is {total_buy_volume}."
                            st.session_state.last_ai_message = response_message
                            chat_message.empty()
                            with chat_message:
                                st.chat_message("assistant").write(response_message)
                        else:
                            response_message = "Market trades data is not available."
                            st.session_state.last_ai_message = response_message
                            chat_message.empty()
                            with chat_message:
                                st.chat_message("assistant").write(response_message)

                    elif "sell" in prompt.lower():
                        if not trades_df.empty:
                            total_sell_volume = trades_df['SELL'].sum()
                            response_message = f"The total sell volume for all options is {total_sell_volume}."
                            st.session_state.last_ai_message = response_message
                            chat_message.empty()
                            with chat_message:
                                st.chat_message("assistant").write(response_message)
                        else:
                            response_message = "Market trades data is not available."
                            st.session_state.last_ai_message = response_message
                            chat_message.empty()
                            with chat_message:
                                st.chat_message("assistant").write(response_message)
                    else:
                        # Call OpenAI API for other questions
                        response = self.call_openai(messages)

                        if not response.choices:
                            response_message = "No response from OpenAI."
                            st.session_state.last_ai_message = response_message
                            chat_message.empty()
                            with chat_message:
                                st.chat_message("assistant").write(response_message)
                        else:
                            ai_message = response.choices[0].message.content
                            st.session_state.last_ai_message = ai_message
                            chat_message.empty()
                            with chat_message:
                                st.chat_message("assistant").write(ai_message)

                except Exception as e:
                    response_message = f"Error: {str(e)}"
                    st.session_state.last_ai_message = response_message
                    chat_message.empty()
                    with chat_message:
                        st.chat_message("assistant").write(response_message)