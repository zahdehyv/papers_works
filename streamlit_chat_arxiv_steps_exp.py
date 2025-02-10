import streamlit as st
import google.generativeai as genai
import arxiv
import Levenshtein
import re
import time
import bleach  # Added for sanitization

# Updated pattern: using <paper-card> instead of <paper>
query_pattern = r"<query>(.*?)</query>"
paper_pattern = r"<paper-card>(.*?)</paper-card>"

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
api_key_input_container = st.empty()  # Create an empty container for conditional input

if not st.session_state.api_key:
    with api_key_input_container.container():
        api_key = st.text_input(
            "Enter your Google Gemini API Key for this session:",
            type="default",
            placeholder="sk-...",
            help="This API key will only be used for the current session and will not be saved."
        )
        if api_key:
            st.session_state.api_key = api_key
            api_key_input_container.empty()
            st.rerun()
        elif api_key == "":
            st.warning("Please enter your API key to use the application.")

# Proceed only if API key is available
if st.session_state.api_key:
    genai.configure(api_key=st.session_state.api_key)

    # Helper functions
    def find_nearest_key_levenshtein_lib(dictionary, target_key, tolerance):
        """Finds the nearest string key in a dictionary using python-Levenshtein."""
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
        """
        Transform the content inside <paper-card> tags by looking up the paper in the results
        and returning a formatted card.
        """
        paper_content = match.group(1)
        result = find_nearest_key_levenshtein_lib(st.session_state.results, paper_content, 5)
        if not result:
            return "\n[NOT FOUND]\n"
        authors = ", ".join([author.name for author in result.authors])
        categories = ", ".join([category for category in result.categories])
        transformed_content = f"""
<div style="border: 1px solid #e6e9ef; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
    <h4><a href="{result.links[0].href}" target="_blank">{result.title}</a> <a href="{result.pdf_url}" target="_blank">[PDF]</a></h4>
    <p><b>Authors:</b> {authors}</p>
    <p><b>Categories:</b> {categories}</p>{f"\n   <p><b>Journal Reference:</b> {result.journal_ref}</p>" if result.journal_ref else ""}
    <p><b>Date:</b> {result.published}</p>
    <p>{result.summary}</p>
</div>"""
        return transformed_content

    # Custom CSS for layout improvements
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] > .main {
                max-width: 90%;
                padding: 20px;
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
        genai.configure(api_key=st.session_state.api_key)
        st.session_state.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-lite-preview-02-05",
            system_instruction="""You are a query creator, reviewer, and answer generator model.
You can create ArXiv queries using the following syntax:
- ti -> Title
- au -> Author
- abs -> Abstract
- co -> Comment
- jr -> Journal Reference
- cat -> Subject Category
- rn -> Report Number
- id -> Id (use id_list instead)
- all -> All of the above

Boolean operations: AND, OR, ANDNOT

Always use <query></query> tags for queries.
When generating an ANSWER, explain your selection criteria and list paper titles within <paper-card>TITLE</paper-card> tags on isolated plain text lines (do not use triple backticks or markdown code blocks). Order the papers by relevance.""",
            generation_config=st.session_state.generation_config,
        )
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.session_state.client = arxiv.Client()
        st.session_state.messages = []
        st.session_state.results = {}

    # Title and description
    st.title("Paper-e  🔍")
    st.caption("Search for papers")

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Let the user configure the number of iterative refinements
    num_iterations = st.number_input("Number of Iterations", min_value=1, max_value=5, value=1, step=1)

    # Chat input and main logic
    if prompt := st.chat_input("Enter a query..."):
        # Append user prompt to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Container for streaming responses
        with st.chat_message("assistant"):
            feedback_container = st.empty()
            feedback_container.info("Sending request for initial query...")
            queries_response = ""
            init_prompt = (
                f"Based on the following user prompt, generate one or more ArXiv search queries. "
                f"Each query must be enclosed in <query> and </query> tags.\n\nUser Prompt: {prompt}"
            )
            for chunk in st.session_state.chat.send_message(init_prompt, stream=True):
                queries_response += chunk.text
                feedback_container.markdown(queries_response)
            feedback_container.empty()

            # Debug: display raw query output for inspection
            st.markdown("**Debug: Raw Query Response**")
            st.markdown(queries_response)

            # Extract initial queries with re.DOTALL flag and strip whitespace
            queries = re.findall(query_pattern, queries_response, flags=re.DOTALL)
            queries = [q.strip() for q in queries]
            if not queries:
                st.markdown("**No queries detected. Using a default query based on the user prompt.**")
                queries = [f"all:{prompt}"]

            # Initialize accumulation variables
            found_total = 0
            result_to_prompt = "### RESULTS:\n"

            # Iteratively refine queries and accumulate results
            for iter_index in range(int(num_iterations)):
                st.markdown(f"**Iteration {iter_index + 1}**")
                # For iterations beyond the first, generate refined queries based on accumulated results
                if iter_index > 0:
                    refinement_prompt = (
                        "QUERY: Based on the following accumulated search results, analyze them for new patterns (such as frequent authors, categories, or keywords) and generate additional ArXiv search queries to further refine the search results.\n\n"
                        f"Previous Results:\n{result_to_prompt}\n\n"
                        "Please generate new query(ies) in the following format: <query>Your query here</query>."
                    )
                    queries_response = ""
                    feedback_container = st.empty()
                    feedback_container.info("Sending request for refined query...")
                    for chunk in st.session_state.chat.send_message(refinement_prompt, stream=True):
                        queries_response += chunk.text
                        feedback_container.markdown(queries_response)
                    feedback_container.empty()

                    # Debug refined query output
                    st.markdown("**Debug: Refined Query Response**")
                    st.markdown(queries_response)

                    new_queries = re.findall(query_pattern, queries_response, flags=re.DOTALL)
                    new_queries = [q.strip() for q in new_queries]
                    if new_queries:
                        queries = new_queries
                    else:
                        st.markdown("**No refined queries detected. Retaining previous queries.**")

                # Process each query: perform ArXiv search and accumulate results
                for query in queries:
                    st.markdown(f"Processing query: **{query}**")
                    try:
                        search = arxiv.Search(
                            query=query,
                            max_results=100,
                            sort_by=arxiv.SortCriterion.Relevance
                        )
                        results = st.session_state.client.results(search)
                        for result in results:
                            found_total += 1
                            st.markdown(f"- Added document: **{result.title}**")
                            st.session_state.results[result.title] = result
                            result_to_prompt += (
                                f"- ####'{result.title}':\n"
                                f"##### Abstract: {result.summary}"
                                f"{f'\n##### Journal Reference: {result.journal_ref}' if result.journal_ref else ''}\n"
                            )
                    except Exception as e:
                        st.error(f"Error processing query '{query}': {e}")

            # Prepare final prompt for answer generation with ordering instructions
            if found_total > 0:
                final_prompt = (
                    f"These are the accumulated search results from all iterations:\n{result_to_prompt}\n\n"
                    "Generate a final ANSWER that explains the criteria used to select the most relevant papers, orders them by relevance, and lists the paper titles. "
                    "For each paper, output its title on a separate, isolated plain text line, wrapped in <paper-card>TITLE</paper-card> tags. "
                    "Do not use triple backticks or markdown code blocks for the <paper-card></paper-card> tags."
                )
            else:
                final_prompt = (
                    "It seems no search results were found. The user might have only asked for clarification. "
                    "Generate an ANSWER that includes <paper-card>TITLE</paper-card> tags for each suggested paper on separate lines."
                )

            feedback_container = st.empty()
            feedback_container.info("Waiting for final answer...")
            response_container = st.empty()
            full_response = ""
            chunk_count = 0
            for chunk in st.session_state.chat.send_message(final_prompt, stream=True):
                chunk_count += 1
                full_response += chunk.text
                if chunk_count % 100 == 0:
                    full_response = re.sub(paper_pattern, replace_paper_content, full_response)
                response_container.markdown(full_response, unsafe_allow_html=True)
            full_response = re.sub(paper_pattern, replace_paper_content, full_response)

            # Sanitize the final output so that only allowed tags remain
            allowed_tags = ['paper-card', 'div', 'p', 'h4', 'a', 'b']
            full_response = bleach.clean(full_response, tags=allowed_tags, strip=True)

        # Append the final answer to chat history and refresh display
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
        st.rerun()
