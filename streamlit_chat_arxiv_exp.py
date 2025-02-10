import streamlit as st
import google.generativeai as genai
import arxiv
import Levenshtein
import re
import time

query_pattern = r"<query>(.*?)</query>"
paper_pattern = r"<paper>(.*?)</paper>"

st.set_page_config(
    page_title="Paper-e  🔍",
    page_icon="📖",
    layout="wide"
)

if "results" not in st.session_state:
    st.session_state.results = {}

# Check if API key is already in session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# API Key Input Section
api_key_input_container = st.empty()  # container for input

if not st.session_state.api_key:
    with api_key_input_container.container():
        api_key = st.text_input("Enter your Google Gemini API Key for this session:",
                                type="default", placeholder="sk-...", 
                                help="This API key will only be used for the current session and will not be saved.")
        if api_key:
            st.session_state.api_key = api_key
            api_key_input_container.empty()
            st.rerun()
        elif api_key == "":
            st.warning("Please enter your API key to use the application.")

if st.session_state.api_key:
    genai.configure(api_key=st.session_state.api_key)
    def find_nearest_key_levenshtein_lib(dictionary, target_key, tolerance):
        nearest_key = None
        min_distance = float('inf')
        for key in dictionary:
            distance = Levenshtein.distance(key, target_key)
            if distance < min_distance:
                min_distance = distance
                nearest_key = key
        if min_distance <= tolerance:
            return dictionary[nearest_key]
        else:
            return None

    def replace_paper_content(match):
        paper_content = match.group(1)
        result: arxiv.Result = find_nearest_key_levenshtein_lib(st.session_state.results, paper_content, 5)
        if not result:
            return "\n[NOT FOUND]\n"
        authors = ", ".join([author.name for author in result.authors])
        categories = ", ".join(result.categories)
        transformed_content = f"""
<div style="border: 1px solid #e6e9ef; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
    <h4><a href="{result.links[0].href}" target="_blank">{result.title}</a> <a href="{result.pdf_url}" target="_blank">[PDF]</a></h4>
    <p><b>Authors:</b> {authors}</p>
    <p><b>Categories:</b> {categories}</p>{f"\n   <p><b>Journal Reference:</b> {result.journal_ref}</p>" if result.journal_ref else ""}
    <p><b>Date:</b> {result.published}</p>
    <p>{result.summary}</p>
    </div>"""
        return transformed_content

    # Custom CSS for the main app view remains as in the original code.
    st.markdown(
        """
    <style>
        [data-testid="stAppViewContainer"] > .main {
            max-width: 90%;
            padding: 20px;
        }
    </style>
        """, unsafe_allow_html=True
    )

    if "messages" not in st.session_state:
        st.session_state.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        genai.configure(api_key=st.session_state.api_key)
        st.session_state.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-lite-preview-02-05",
            system_instruction="""Your instructions for generating queries and answers...
[see original instructions]"""
            , generation_config=st.session_state.generation_config,
        )
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.session_state.client = arxiv.Client()
        st.session_state.messages = []
        st.session_state.results = {}

    st.title("Paper-e  🔍")
    st.caption("search for papers")

    # --- New Layout: Split interface into two columns ---
    left_col, right_col = st.columns([0.6, 0.4])

    # Left column: Chat messages and input
    with left_col:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        # Chat input and processing
        if prompt := st.chat_input("enter a query..."):
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Generating queries..."):
                    feedback_container = st.empty()
                    feedback_container.markdown("Sending request...")
                    queries_response = ""
                    for chunk in st.session_state.chat.send_message(
                        f"generate a QUERY or QUERIES for the user prompt (remember the use of <query></query>):\n'{prompt}'\n\n (If the user only asks for clarification you can just use the responses from the previous queries)", 
                        stream=True):
                        queries_response += chunk.text
                        feedback_container.markdown(queries_response)
                    feedback_container.empty()
                    found = 0
                    queries = re.findall(query_pattern, queries_response)
                    result_to_prompt = "### RESULTS:\n"
                    for query in queries:
                        with st.spinner("Processing query: $" + query):
                            search = arxiv.Search(
                                query=query,
                                max_results=100,
                                sort_by=arxiv.SortCriterion.Relevance
                            )
                            for result in st.session_state.client.results(search):
                                found += 1
                                st.info(f"Added document: '{result.title}'")
                                st.session_state.results[result.title] = result
                                result_to_prompt += f"""- ####'{result.title}':
    ##### Abstract: {result.summary}{f"\n##### Journal Reference: {result.journal_ref}" if result.journal_ref else ""}
    """
                with st.spinner("Generating response..."):
                    response_container = st.empty()
                    full_response = ""
                    # Updated prompt instructing the answer format:
                    if found > 0:
                        prompt_answer = (f"These are the results to the queries:\n{result_to_prompt}\n"
                                         "Generate an ANSWER that first states the criteria for selecting the papers, "
                                         "then provides a summary and conclusions, and finally lists the paper titles (without any <paper> tags).")
                    else:
                        prompt_answer = ("The user probably only asked for clarification, check for it. "
                                         "(Remember to list any relevant paper titles if applicable).")
                    chunkn = 0
                    for chunk in st.session_state.chat.send_message(prompt_answer, stream=True):
                        chunkn += 1
                        full_response += chunk.text
                        if chunkn % 100 == 0:
                            full_response = re.sub(paper_pattern, replace_paper_content, full_response)
                        response_container.markdown(full_response, unsafe_allow_html=True)
                    full_response = re.sub(paper_pattern, replace_paper_content, full_response)
                    # Remove any <paper> tags from the final answer text
                    final_answer_text = re.sub(r'</?paper>', '', full_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer_text
                    })
                    # Optionally, update the chat message display
                    response_container.markdown(final_answer_text, unsafe_allow_html=True)
            st.rerun()

    # Right column: Results panel (only shown if there are results)
    with right_col:
        if st.session_state.results:
            st.subheader("Results")
            # For each result, display a card-like item using an expander
            for title, result in st.session_state.results.items():
                authors = ", ".join([author.name for author in result.authors])
                expander_label = f"{result.title} - Authors: {authors}"
                with st.expander(label=expander_label, expanded=False):
                    st.markdown(f"""
<div style="background-color: #ffcccc; padding: 10px; border-radius: 5px;">
    <p><b>Abstract:</b> {result.summary}</p>
    {"<p><b>Journal Reference:</b> " + result.journal_ref + "</p>" if result.journal_ref else ""}
    <p><b>Date:</b> {result.published}</p>
</div>
""", unsafe_allow_html=True)
