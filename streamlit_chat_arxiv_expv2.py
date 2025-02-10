import streamlit as st
import google.generativeai as genai
import arxiv
import Levenshtein
import re
import time
from math import sqrt
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
        
        # print(result.journal_ref)
        
        authors = ", ".join([author.name for author in result.authors])
        categories = ", ".join([category for category in result.categories])
#         transformed_content = f"""
# <div style="border: 1px solid #e6e9ef; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
#     <h4><a href="{result.links[0].href}" target="_blank">{result.title}</a> <a href="{result.pdf_url}" target="_blank">[PDF]</a></h4>
#     <p><b>Authors:</b> {authors}</p>
#     <p><b>Categories:</b> {categories}</p>{f"\n   <p><b>Journal Reference:</b> {result.journal_ref}</p>" if result.journal_ref else ""}
#     <p><b>Date:</b> {result.published}</p>
#     <p>{result.summary}</p>
#     </div>"""
        transformed_content = f"- ###### {result.title}"
        st.session_state.last_query_results.append(result)
        return transformed_content


    # Configure page
    
    chat_container, res_container = st.columns([(1+sqrt(5))/2,1], border=True)
    # chat_container.heigh = "100"
    # res_container.height = "100"
    
    with chat_container:
        with st.container(height=600, border=True):
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
        When asked for an ANSWER, you must state a criteria for selecting the papers, then select from the papers the most relevant ones and provide an answer to the user encapsulating the titles of the papers in <paper>TITLE</paper> tags.
        Consider that the <paper></paper> tags will be replaced by a CARD, so put it as an independent line, only containing the tags and title.
        For example:
        <paper>TITLE</paper>
        I would like the final answer to have the following structure:
        1. Think step by step to state the selection criteria for the most relevant papers (consider if it has a journal reference and it is relevant).
        2. Use that criteria and introduce, evaluating the fit to the criteria, the most relevant papers, one by one:
        For example:
        I selected the following papers because [explanation]

        <paper>TITLE</paper>

        <paper>TITLE</paper>

        (continue...)

        I also considered those relevant because [explanation]

        <paper>TITLE</paper>

        <paper>TITLE</paper>

        """,
                    generation_config=st.session_state.generation_config,
                )

                st.session_state.chat = st.session_state.model.start_chat(history=[])
                st.session_state.client = arxiv.Client()
                st.session_state.messages = []
                st.session_state.results = {}
                st.session_state.last_query_results = []

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
                    with st.spinner("Generating queries..."):
                        feedback_container = st.empty()  # Create an empty container for streaming
                        feedback_container.markdown("Sending request...")  # Initial feedback message
                        queries_response = ""


                        # Stream the response from Gemini

                        for chunk in st.session_state.chat.send_message(f"generate a QUERY or QUERIES for the user prompt (remember the use of <query></query>):\n'{prompt}'\n\n (If the user only asks for clarification you can just use the responses from the previous queries)", stream = True):
                            queries_response += chunk.text
                            feedback_container.markdown(queries_response)
                        
                        feedback_container.empty()
                        
                        found = 0
                        queries = re.findall(query_pattern, queries_response)
                        result_to_prompt = "### RESULTS:\n"
                        qur_cnt = []
                        for query in queries:
                            qur_cnt = []
                            qur_cnt.append(st.empty())
                            with st.spinner("Processing query: $"+query+""):
                                search = arxiv.Search(
                                    query=query,
                                    max_results=100,
                                    sort_by=arxiv.SortCriterion.Relevance
                                )
                                results = st.session_state.client.results(search)
                                qur_cnt.append(st.empty())
                                for result in results:
                                    found += 1
                                    # time.sleep(0.07)
                                    qur_cnt[-1].markdown("- Added document: '"+result.title+"'")
                                    st.session_state.results[result.title] = result
                                    result_to_prompt+=f"""- ####'{result.title}':
            ##### Abstract: {result.summary}{f"\n##### Journal Reference: {result.journal_ref}" if result.journal_ref else ""}
            """
                    with st.spinner("Generating response..."):
                        response_container = st.empty()  # Create an empty container for streaming
                        full_response = ""

                        if found>0:
                            prompt = f"These are the results to the queries:\n{result_to_prompt}\nUse them to generate an ANSWER (Remember to include the <paper>TITLE</paper> tags for each answer, and state them in isolated lines (as they will be converted to cards with the info))."
                        else:
                            prompt = "The user probably only asked for clarification, check for it. (Remember to include the <paper>TITLE</paper> tags for each answer)."
                        # Stream the response from Gemini
                        
                        st.session_state.last_query_results = []
                        
                        chunkn = 0
                        
                        for chunk in st.session_state.chat.send_message(prompt, stream=True):
                            chunkn+=1
                            full_response += chunk.text
                            if chunkn%23==0:
                                full_response = re.sub(paper_pattern, replace_paper_content, full_response)
                            response_container.markdown(full_response, unsafe_allow_html=True)  # Update the container with new text

                        full_response = re.sub(paper_pattern, replace_paper_content, full_response)


                        # Add assistant response to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response
                        })

                # Rerun to show new messages
                st.rerun()
    if st.session_state.last_query_results is not None and len(st.session_state.last_query_results)>0:
        with res_container:
            with st.container(height=600, border=True):
                resss = []
                for result in st.session_state.last_query_results:
                    result: arxiv.Result
                    resss.append(st.container(border=True, height=256))
                    resss[-1].markdown("#### "+f"[{result.title}]({result.links[0]}) [**[PDF]**]({result.pdf_url})")
                    if result.journal_ref:
                        resss[-1].markdown("##### **Journal Reference:** "+f"{result.journal_ref}")
                    resss[-1].markdown("###### **Authors:** "+f"{", ".join([author.name for author in result.authors])}")
                    resss[-1].markdown("**Abstract:** "+f"{result.summary}")
            