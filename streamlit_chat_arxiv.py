import streamlit as st
import google.generativeai as genai
import arxiv
import Levenshtein
import re

query_pattern = r"<query>(.*?)</query>"
paper_pattern = r"<paper>(.*?)</paper>"

st.set_page_config(
    page_title="Paper-e  🔍",
    page_icon="📖",
    layout="centered"
    )

if "results" not in st.session_state:
    st.session_state.results = {}

# Check if API key is already in session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# API Key Input Section
api_key_input_container = st.empty() # Create an empty container to conditionally render the input

if not st.session_state.api_key: # If API key is not in session state, show the input
    with api_key_input_container.container(): # Use the container to group elements if needed
        api_key = st.text_input("Enter your Google Gemini API Key for this session:", type="password", placeholder="sk-...", help="This API key will only be used for the current session and will not be saved.")
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
    def find_nearest_key_levenshtein_lib(dictionary, target_key, tolerance):
        """Finds the nearest string key in a dictionary using python-Levenshtein.

        Args:
            dictionary: The dictionary with string keys.
            target_key: The string to search for a near match for.
            tolerance: The maximum acceptable Levenshtein distance.

        Returns:
            The nearest key if found within the tolerance, otherwise None.
        """
        nearest_key = None
        min_distance = float('inf')

        for key in dictionary:
            distance = Levenshtein.distance(key, target_key) # Using library function
            if distance < min_distance:
                min_distance = distance
                nearest_key = key

        if min_distance <= tolerance:
            return dictionary[nearest_key]
        else:
            return None

    def replace_paper_content(match):
        """
        Function to transform the content inside <paper> tags and use it as replacement.
        For this example, we'll just uppercase the content.
        """
        paper_content = match.group(1) # Access the captured group (text inside <paper>)
        result: arxiv.Result = find_nearest_key_levenshtein_lib(st.session_state.results, paper_content, 5)
        if not result:
            return "\n[NOT FOUND]\n"
        authors = ", ".join([author.name for author in result.authors])
        transformed_content = f"""
<div style="border: 1px solid #e6e9ef; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
    <h4><a href="{result.pdf_url}" target="_blank">{result.title}</a></h4>
    <p><b>Authors:</b> {authors}</p>
    <p><b>Date:</b> {result.published}</p>
    <p>{result.summary}</p>
    </div>"""
        return transformed_content


    # Configure page
    

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
            system_instruction="""You are a query creator, reviewer, and answer generator model, you can create arxiv queries using the basic syntax
    Here are some prefixes to indicate the queried field:
    - ti -> Title
    - au -> Author
    - abs -> Abstract
    - co -> Comment
    - jr -> Journal Reference
    - cat -> Subject Category
    - rn -> Report Number
    - id -> Id (use id_list instead)
    - all -> All of the above

    some logical operations:
    - AND
    - OR
    - ANDNOT
    The ANDNOT Boolean operator is particularly useful, as it allows us to filter search results based on certain fields. For example, if we wanted all of the articles by the author Adrian DelMaestro with titles that did not contain the word checkerboard, we could construct the following query:

    <query>au:del_maestro ANDNOT ti:checkerboard</query>

    and agrupation terms like:
    - ( )Used to group Boolean expressions for Boolean operator precedence.
    - double quotes	Used to group multiple words into phrases to search a particular field.
    - space	Used to extend a search_query to include multiple fields.

    example: <query>ti:(reasoning AND llm)</query>
    always use the <query></query> and provide multiple queries if needed.

    When asked for QUERY you must propose the queries inside <query></query> tags.
    When asked for an ANSWER, you must select from the papers the most relevant ones and provide an answer to the user encapsuling the titles of the papers in <paper>TITLE</paper> tags, and then you must provide a little explanation on why it is relevant. You must also provide a table at the end, offering a brief summary and key insights
    Consider that the <paper></paper> tags will be replaced by a CARD, so do not put it as a continuation of any other fragment or in a list (eg. starting with '-')""",
            generation_config=st.session_state.generation_config,
        )

        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.session_state.client = arxiv.Client()
        st.session_state.messages = []
        st.session_state.results = {}

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
            st.markdown(message["content"], unsafe_allow_html=True)

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
            feedback_container = st.empty()  # Create an empty container for streaming
            # feedback_container.markdown()  # Initial feedback message
            queries_response = "### Generating queries:\n"


            # Stream the response from Gemini

            for chunk in st.session_state.chat.send_message(f"generate a QUERY or QUERIES for the user prompt:\n{prompt}", stream = True):
                queries_response += chunk.text
                feedback_container.markdown(queries_response, unsafe_allow_html=True)

            queries = re.findall(query_pattern, queries_response)
            result_to_prompt = "RESULTS:\n"
            qur_cnt = []
            for query in queries:
                qur_cnt.append(st.empty())
                qur_cnt[-1].markdown("#### processing query: '"+query+"'", unsafe_allow_html=True)
                search = arxiv.Search(
                    query=query,
                    max_results=10,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                results = st.session_state.client.results(search)
                for result in results:
                    qur_cnt.append(st.empty())
                    qur_cnt[-1].markdown("adding document: '"+result.title+"'", unsafe_allow_html=True)
                    st.session_state.results[result.title] = result
                    result_to_prompt+=f"""- {result.title}: {result.summary}
    """

            response_container = st.empty()  # Create an empty container for streaming
            full_response = ""


            # Stream the response from Gemini
            for chunk in st.session_state.chat.send_message(f"These are the results to the queries:\n{result_to_prompt}\nUse them to generate an ANSWER.", stream=True):
                full_response += chunk.text
                response_container.markdown(full_response, unsafe_allow_html=True)  # Update the container with new text

            full_response = re.sub(paper_pattern, replace_paper_content, full_response)


        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

        # Rerun to show new messages
        st.rerun()