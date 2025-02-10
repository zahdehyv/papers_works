import streamlit as st
import google.generativeai as genai
from googlesearch import search


st.set_page_config(
    page_title="Google  🔍",
    page_icon="🔍",
    layout="wide"
    )

if not "results" in st.session_state:
    st.session_state.results = []

if prompt := st.chat_input("enter a query..."):
    st.session_state.results = search(prompt, advanced=True, num_results=100)
    
    st.rerun()

result_c = []
for res in st.session_state.results:
    result_c.append(st.container(border=True))
    result_c[-1].markdown(f"##### [{res.title}]({res.url})")
    result_c[-1].markdown(f"{res.description}")
