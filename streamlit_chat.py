import streamlit as st
import google.generativeai as genai

# Configure Gemini model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# Configure page
st.set_page_config(
    page_title="Paper-e  🔍",
    page_icon="📖",
    layout="centered"
)

# Initialize chat history
if "messages" not in st.session_state:
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
        st.markdown(message["content"])

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
            response_container.markdown(full_response)  # Update the container with new text

    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })
    
    # Rerun to show new messages
    st.rerun()