import os
import streamlit as st
from langchain.agents import load_tools
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from llama_agi.execution_agent import ToolExecutionAgent
from llama_agi.runners import AutoStreamlitAGIRunner
from llama_agi.task_manager import LlamaTaskManager
from llama_agi.tools import search_notes, record_note, search_webpage

from llama_index import ServiceContext, LLMPredictor


st.set_page_config(layout="wide")
st.header("🤖 Llama AGI 🦙")
st.text("Use the setup tab to configure your LLM settings and initial objective+tasks.")
st.text("Then use the Launch tab to run the AGI. Kill the process by clicking 'Stop' in the top right.")

setup_tab, launch_tab = st.tabs(["Setup", "Launch"])

with setup_tab:
    st.subheader("LLM Setup")
    col1, col2, col3 = st.columns(3)

    with col1:
        openai_api_key = st.text_input("Enter your OpenAI API key here", type="password")
        llm_name = st.selectbox(
            "Which LLM?", ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]
        )

    with col2:
        google_api_key = st.text_input("Enter your Google API key here", type="password")
        model_temperature = st.slider(
            "LLM Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.0
        )
    
    with col3:
        google_cse_id = st.text_input("Enter your Google CSE ID key here", type="password")
        max_tokens = st.slider(
            "LLM Max Tokens", min_value=256, max_value=1024, step=8, value=512
        )

    st.subheader("AGI Setup")
    objective = st.text_input("Objective:", value="Solve world hunger")
    initial_task = st.text_input("Initial Task:", value="Create a list of tasks")


    if st.button('Initialize?'):
        os.environ['OPENAI_API_KEY'] = openai_api_key
        os.environ['GOOGLE_API_KEY'] = google_api_key
        os.environ['GOOGLE_CSE_ID'] = google_cse_id
        if llm_name == "text-davinci-003":
            llm = OpenAI(
                temperature=model_temperature, model_name=llm_name, max_tokens=max_tokens
            )
        else:
            llm= ChatOpenAI(
                temperature=model_temperature, model_name=llm_name, max_tokens=max_tokens
            )
        
        service_context = ServiceContext.from_defaults(
            llm_predictor=LLMPredictor(llm=llm), chunk_size_limit=512
        )

        st.session_state['task_manager'] = LlamaTaskManager(
            [initial_task], task_service_context=service_context
        )

        tools = load_tools(["google-search-results-json"])
        tools = tools + [search_notes, record_note, search_webpage]
        st.session_state['execution_agent'] = ToolExecutionAgent(llm=llm, tools=tools)

        st.session_state['initial_task'] = initial_task
        st.session_state['objective'] = objective

        st.session_state['init'] = True
        st.success("Initialized!")

with launch_tab:
    st.subheader("AGI Status")
    if st.button("Launch"):
        if st.session_state.get('init', False):
            # launch the auto runner
            with st.spinner("Running!"):
                runner = AutoStreamlitAGIRunner(st.session_state['task_manager'], st.session_state['execution_agent'])
                runner.run(st.session_state['objective'], st.session_state['initial_task'], 2)
