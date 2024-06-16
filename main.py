import os
import tempfile
from enum import Enum
from typing import Any
from uuid import UUID

import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

GPT3_LLM_MODEL = "gpt-3.5-turbo"
GPT4_LLM_MODEL = "gpt-4-turbo"
GPT4o_LLM_MODEL = "gpt-4o"

uploaded_files = []
chain = None


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("thinking...")
        self.container = container

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.status.text("Retrieving...")

    def on_retriever_end(self, documents, **kwargs):
        self.status.text("Retrieved.")
        self.container.empty()


def document_retriever(files, use_compression=False):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in files:
        temp_file_path = os.path.join(temp_dir.name, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getvalue())
        _, ext = os.path.splitext(file.name)
        if ext == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif ext == ".txt":
            loader = TextLoader(temp_file_path)
        else:
            st.write("Unsupported file format. Use .txt or .pdf files.")
            return
        docs.extend(loader.load())

    # Split Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    vectordb_retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )

    if not use_compression:
        return vectordb_retriever

    compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)


class ChatProfileRoleEnum(str, Enum):
    HUMAN = "human"
    AI = "ai"


class LLMProviderEnum(str, Enum):
    GPT3 = "GPT-3"
    GPT4 = "GPT-4"
    GPT4o = "GPT-4o"


# Setup the sidebar for streamlit

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"


# Streamlit app configuration

st.set_page_config(
    page_title="Chat with paper",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": """Chat with a Paper."""
    }
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
            placeholder="Select a model to analyze the paper with."
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
            messages.add_ai_message("""
            Hello, I have studied your paper, and I am ready to answer your questions.

            What would you like to know?
            """)

        uploaded_files = st.file_uploader("Upload a paper to chat with",
                                          type=["pdf", "txt"],
                                          accept_multiple_files=True,
                                          disabled=(not selected_model or not api_key))
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
        response = chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

if selected_model and model_name:
    st.sidebar.caption(f"Using {selected_model} model.")
