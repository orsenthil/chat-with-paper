from enum import Enum

import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from langchain_callbacks import StreamHandler, PrintRetrievalHandler
from models import GPT3_LLM_MODEL, GPT4_LLM_MODEL, GPT4o_LLM_MODEL, LLMProviderEnum
from retriever import document_retriever

chain = None
uploaded_files = []


class ChatProfileRoleEnum(str, Enum):
    HUMAN = "human"
    AI = "ai"


# Setup the sidebar for streamlit

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"

# Streamlit app configuration

st.set_page_config(
    page_title="Chat with paper",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": """Chat with a Paper."""},
)


with st.sidebar:
    st.write("Chat with paper.")
    # noinspection PyPackageRequirements
    with st.container():
        # Model
        selected_model = st.selectbox(
            "Select a model",
            options=[
                LLMProviderEnum.GPT3.value,
                LLMProviderEnum.GPT4.value,
                LLMProviderEnum.GPT4o.value,
            ],
            index=0,
            placeholder="Select a model to analyze the paper with.",
        )

        if selected_model:
            api_key = st.text_input(f" {selected_model} API Key", type="password")
            if selected_model == LLMProviderEnum.GPT3:
                model_name = GPT3_LLM_MODEL
            elif selected_model == LLMProviderEnum.GPT4:
                model_name = GPT4_LLM_MODEL
            elif selected_model == LLMProviderEnum.GPT4o:
                model_name = GPT4o_LLM_MODEL

        messages = StreamlitChatMessageHistory()

        if len(messages.messages) == 0 or st.button("Clear Chat"):
            messages.clear()
            messages.add_ai_message(
                """
            Hello, I have studied your paper, and I am ready to answer your questions.

            What would you like to know?
            """
            )

        uploaded_files = st.file_uploader(
            "Upload a paper to chat with",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            disabled=(not selected_model or not api_key),
        )

if not selected_model:
    st.info("Please select a model to chat with.")

if not api_key:
    st.info(f"Please enter the API key of {selected_model} to chat with the paper.")

if uploaded_files:
    retriever = document_retriever(uploaded_files, use_compression=False)
    if retriever is not None:
        pass
        # st.write("Got your Paper.")

    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=messages, return_messages=True)
    if selected_model in [LLMProviderEnum.GPT3, LLMProviderEnum.GPT4, LLMProviderEnum.GPT4o]:
        st.write(f"Chatting with {LLMProviderEnum(selected_model).value} model.")
        llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0, streaming=True)

        if llm is None:
            st.error("Failed to create the LLM model. Please check your API Key.")

        # Create the Conversational RetrievalChain using the LLM instance.
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True,
            max_tokens_limit=4000,
        )

        avatars = {
            ChatProfileRoleEnum.AI.value: "ai",
            ChatProfileRoleEnum.HUMAN.value: "human",
        }

        for msg in messages.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)

# Get User Input and Generate Response

if user_query := st.chat_input(placeholder="Ask me Anything!", disabled=(not uploaded_files)):
    st.chat_message("human").write(user_query)
    with st.chat_message("ai"):
        retrieval_handler = PrintRetrievalHandler(st.empty())
        stream_handler = StreamHandler(st.empty())
        chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

if selected_model and model_name:
    st.sidebar.caption(f"Using {selected_model} model.")
