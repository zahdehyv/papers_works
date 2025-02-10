import streamlit as st
import google.generativeai as genai
import re

query_pattern = r"<query>(.*?)</query>"
paper_pattern = r"<paper>(.*?)</paper>"

st.set_page_config(
    page_title="Paper-e  🔍",
    page_icon="📖",
    layout="wide"
    )


# Check if API key is already in session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# API Key Input Section
api_key_input_container = st.empty() # Create an empty container to conditionally render the input

if not st.session_state.api_key: # If API key is not in session state, show the input
    with api_key_input_container.container(): # Use the container to group elements if needed
        api_key = st.text_input("Enter your Google Gemini API Key for this session:", type="default", placeholder="sk-...", help="This API key will only be used for the current session and will not be saved.")
        if api_key:
            st.session_state.api_key = api_key # Store in session state
            api_key_input_container.empty() # Clear the input container after API key is entered
            st.rerun() # Rerun to initialize the model with the API key
        elif api_key == "": # Handle empty input (optional, maybe show a warning)
            st.warning("Please enter your API key to use the application.")


# Only proceed to configure the model and app if API key is available
if st.session_state.api_key:
    genai.configure(api_key=st.session_state.api_key)
    # Configure functions

    # Custom CSS to make the central column wider
    st.markdown(
        """
    <style>
        [data-testid="stAppViewContainer"] > .main {
            max-width: 90%;
            padding-top: 20px;
            padding-right: 20px;
            padding-left: 20px;
            padding-bottom: 20px;
        }
    </style>
        """,
        unsafe_allow_html=True,
    )


    if "messages" not in st.session_state:
        st.session_state.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
            }

        genai.configure(api_key=st.session_state.api_key) # Configure genai with API key
        st.session_state.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-lite-preview-02-05",
            generation_config=st.session_state.generation_config,
        )

        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.session_state.messages = []

    # Title and description
    st.title("Paper-e  🔍")
    st.markdown("""
    <style>
        .st-emotion-cache-1kyxreq {
            justify-content: center;
        }
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] {
            align-items: center;
        }
    </style>
    """, unsafe_allow_html=True)
    st.caption("search for papers")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])#, unsafe_allow_html=True)

    # Chat input and logic
    if prompt := st.chat_input("enter a query..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and stream response
        with st.chat_message("assistant"):
            response_container = st.empty()  # Create an empty container for streaming
            full_response = ""

            # Stream the response from Gemini
            for chunk in st.session_state.chat.send_message(prompt, stream=True):
                full_response += chunk.text
                response_container.markdown(full_response)#, unsafe_allow_html=True)  # Update the container with new text


        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

        # Rerun to show new messages
        st.rerun()