import streamlit as st
import re
import google.generativeai as genai
import os
import time

# --- Set up Gemini API Key ---
gemini_api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input(
    "Enter your Gemini API Key:", type="password"
)
if not gemini_api_key:
    st.warning("Please enter your Gemini API key to use the chatbot.")
else:
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# --- Gemini Setup Functions ---
def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")
    print()

def setup_gemini_model():
    """Sets up the Gemini Generative Model."""
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 65536,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21", # or a suitable Gemini model
        generation_config=generation_config,
    )
    return model


def process_query_gemini(query, chat_session, pdf_file_gemini):
    """Processes the query with Gemini and extracts reference."""
    if not gemini_api_key:
        return "Please enter your Gemini API key to use the chatbot.", None

    try:
        response = chat_session.send_message([query, pdf_file_gemini]) # Send query and Gemini file
        answer = response.text

        ref_text = None
        ref_match = re.search(r"<ref>(.*?)</ref>", answer, re.DOTALL) # re.DOTALL to match across newlines
        if ref_match:
            ref_text = ref_match.group(1)
            # Clean up answer by removing ref tags for display
            answer = re.sub(r"<ref>.*?</ref>", "", answer).strip()

        return answer, ref_text

    except Exception as e:
        return f"Error processing query with Gemini: {e}", None


# --- Streamlit App Layout ---
st.title("PDF Chatbot (Gemini Only)")

col1, col2 = st.columns([1, 1]) # Equal columns now

with col1:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Upload your PDF", type=['pdf'])
    gemini_file_placeholder = st.empty() # Placeholder for Gemini file object

    if pdf_file is not None:
        if gemini_api_key:
            gemini_model = setup_gemini_model()
            chat_session = gemini_model.start_chat(history=[])

            # Upload PDF to Gemini and wait for it to be active
            temp_pdf_path = "temp_pdf.pdf" # Temporary file to save uploaded PDF
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.read())
            gemini_uploaded_file = upload_to_gemini(temp_pdf_path, mime_type="application/pdf")
            wait_for_files_active([gemini_uploaded_file])
            st.session_state.gemini_pdf_file = gemini_uploaded_file # Store Gemini file object in session state
            gemini_file_placeholder.success("PDF uploaded to Gemini for processing.") # Indicate PDF upload to Gemini
            os.remove(temp_pdf_path) # Clean up temporary file
        else:
            gemini_model = None
            chat_session = None


with col2:
    st.header("Chatbot")
    if pdf_file is None:
        st.info("Please upload a PDF file to activate the chatbot.")
    elif not gemini_api_key:
        st.info("Enter your Gemini API key to use the chatbot.")
    elif chat_session and 'gemini_pdf_file' in st.session_state: # Check for chat_session and Gemini file
        chat_placeholder = st.empty()
        query = st.chat_input("Ask questions about the PDF:")

        if query:
            with chat_placeholder.container():
                st.chat_message("user").write(query)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking with Gemini..."):
                        answer, ref_text = process_query_gemini(query, chat_session, st.session_state.gemini_pdf_file) # Pass Gemini file object
                        st.write(answer)

                        if ref_text:
                            st.info(f"Reference text found: `{ref_text}`")
                            st.warning("Highlighting is removed in this simplified version.") # Indicate highlighting removed
                        else:
                            st.warning("No reference text found in the LLM's response.")
    else:
        st.info("Gemini Chatbot not initialized. Please upload PDF and enter API key.")