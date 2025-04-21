# deepseek_chat_app.py
import streamlit as st
import requests
import os
import json
from datetime import datetime
from typing import Generator

class DeepSeekAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"  # Update with actual API URL
    
    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """Stream chat completion from DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data:"):
                        data = decoded_line[5:].strip()
                        if data != "[DONE]":
                            try:
                                chunk = json.loads(data)["choices"][0]["delta"].get("content", "")
                                yield chunk
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield f"❌ Error: {str(e)}"

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    
    if "model" not in st.session_state:
        st.session_state.model = "deepseek-chat"
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1024

def render_sidebar():
    """Render the sidebar configuration options"""
    with st.sidebar:
        st.header("Configuration")
        
        # API Key Configuration
        api_key_source = st.radio(
            "API Key Source",
            ("Environment Variable", "Direct Input"),
            index=0,
            help="Choose how to provide your DeepSeek API key"
        )
        
        if api_key_source == "Environment Variable":
            env_var_name = st.text_input(
                "Environment Variable Name",
                value="DEEPSEEK_API_KEY",
                help="Name of the environment variable containing your API key"
            )
            st.session_state.api_key = os.getenv(env_var_name)
        else:
            st.session_state.api_key = st.text_input(
                "DeepSeek API Key",
                type="password",
                help="Enter your DeepSeek API key directly"
            )
        
        # Model Configuration
        st.session_state.model = st.selectbox(
            "Model",
            ["deepseek-chat", "deepseek-coder"],
            help="Select which DeepSeek model to use"
        )
        
        # Generation Parameters
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness. Lower = more deterministic."
        )
        
        st.session_state.max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=4096,
            value=1024,
            help="Maximum number of tokens to generate"
        )
        
        # Clear Chat Button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

def render_chat_messages():
    """Display all chat messages in the conversation"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"{message['timestamp']}")

def handle_user_input():
    """Process user input and generate assistant response"""
    if prompt := st.chat_input("What would you like to ask?"):
        # Add user message to chat history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt, 
            "timestamp": timestamp
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(timestamp)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            if not st.session_state.api_key:
                full_response = "⚠️ Please provide a valid DeepSeek API key in the sidebar."
            else:
                deepseek_client = DeepSeekAPI(st.session_state.api_key)
                
                # Prepare messages for API (without timestamps)
                api_messages = [
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.messages
                ]
                
                # Stream the response
                for chunk in deepseek_client.stream_chat(
                    model=st.session_state.model,
                    messages=api_messages,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.caption(timestamp)
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response, 
            "timestamp": timestamp
        })

def main():
    """Main application function"""
    st.set_page_config(
        page_title="DeepSeek Chat", 
        page_icon="🤖",
        layout="wide"
    )
    st.title("DeepSeek Chat Interface")
    
    initialize_session_state()
    render_sidebar()
    render_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    main()