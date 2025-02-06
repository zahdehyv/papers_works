import streamlit as st
import google.generativeai as genai
import arxiv

# Configure Gemini model
# Assuming you have your API key set up in environment variables or st.secrets
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"]) # Uncomment if needed and configured

# Configure page
st.set_page_config(
    page_title="Paper-e  🔍",
    page_icon="📖",
    layout="centered"
)

if "messages" not in st.session_state:
    st.session_state.generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    st.session_state.model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp", # or a suitable Gemini model
        generation_config=st.session_state.generation_config,
    )
    st.session_state.chat = st.session_state.model.start_chat(history=[])
    st.session_state.client = arxiv.Client()
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
st.caption("search for papers on arXiv")

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

        # --- arXiv Search Logic ---
        search_query = prompt  # Use the user's prompt as the arXiv search query
        search = arxiv.Search(
            query=search_query,
            max_results=5,  # Limit to a reasonable number of results for chat
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )

        results = st.session_state.client.results(search)

        paper_responses = []
        for paper in results:
            paper_response = f"""**{paper.title}**
*Authors*: {', '.join([author.name for author in paper.authors])}
*Published*: {paper.published.date()}
[PDF Link]({paper.pdf_url})

> {paper.summary[:200]}... [Read More]({paper.entry_id})

---
"""  # Basic paper info, truncated summary
            paper_responses.append(paper_response)

        if paper_responses:
            full_response = "Here are some papers from arXiv:\n\n" + "\n".join(paper_responses)
        else:
            full_response = "No papers found on arXiv for your query. Please try a different search term."

        response_container.markdown(full_response)  # Display arXiv results

    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

    # Rerun to show new messages
    st.rerun()