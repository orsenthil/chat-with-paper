import os
import tempfile
from enum import Enum
from typing import Dict, Any, Optional, List
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
GPT4_LLM_MODEL = "gpt-4"
GPT4o_LLM_MODEL = "gpt-4-o"

uploaded_files = []
chain = None


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text: str=""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized:dict, prompts: list, **kwargs):
        self.container.write(f"LLM Start: {serialized}")
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
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )

    if not use_compression:
        return retriever

    compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)



class ChatProfileRoleEnum(str, Enum):
    HUMAN = "Human"
    AI = "AI"


class LLMProviderEnum(str, Enum):
    GPT3 = "GPT-3"
    GPT4 = "GPT-4"
    GPT4o = "GPT-4o"
    BERT = "BERT"
    RoBERTa = "RoBERTa"
    DistilBERT = "DistilBERT"
    T5 = "T5"
    XLNet = "XLNet"
    ALBERT = "ALBERT"
    Electra = "Electra"
    BART = "BART"
    CamemBERT = "CamemBERT"
    Pegasus = "Pegasus"
    ProphetNet = "ProphetNet"
    MBart = "MBart"
    MarianMT = "MarianMT"
    XLM = "XLM"
    XLM_R = "XLM-R"
    TuringNLG = "Turing-NLG"
    DialoGPT = "DialoGPT"
    BlenderBot = "BlenderBot"
    Meena = "Meena"
    LaMDA = "LaMDA"
    DALLE = "DALL-E"
    CLIP = "CLIP"
    VQ_VAE_2 = "VQ-VAE-2"
    StyleGAN = "StyleGAN"
    BigGAN = "BigGAN"
    StyleGAN2 = "StyleGAN2"
    StyleGAN3 = "StyleGAN3"
    BigGAN_512px = "BigGAN-512px"
    StyleGAN2_ADA = "StyleGAN2-ADA"
    StyleGAN2_ADA_512px = "StyleGAN2-ADA-512px"
    StyleGAN2_ADA_1024px = "StyleGAN2-ADA-1024px"


# Setup the sidebar for streamlit

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"


# Streamlit app configuration

st.set_page_config(
    page_title="Streamlit Sidebar",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state=st.session_state.sidebar_state,
    menu_items={
        "About": """Chat with a Paper."""
    }
)

with st.sidebar:
    st.write("Chat with a paper.")
    st.write("You can expand or collapse it using the button below.")
    if st.button("Expand/Collapse Sidebar"):
        st.session_state.sidebar_state = "expanded" if st.session_state.sidebar_state == "collapsed" else "collapsed"
        st.experimental_rerun()
    with st.container():
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            st.write("This is a column")
        with col2:
            st.write("This is another column")

        # Model
        selected_model = st.selectbox(
            "Select a model",
            options = [
                LLMProviderEnum.GPT3.value,
                LLMProviderEnum.GPT4.value,
                LLMProviderEnum.GPT4o.value,
                LLMProviderEnum.BERT.value,
                LLMProviderEnum.RoBERTa.value,
                LLMProviderEnum.DistilBERT.value,
                LLMProviderEnum.T5.value,
                LLMProviderEnum.XLNet.value,
                LLMProviderEnum.ALBERT.value,
                LLMProviderEnum.Electra.value,
                LLMProviderEnum.BART.value,
                LLMProviderEnum.CamemBERT.value,
                LLMProviderEnum.Pegasus.value,
                LLMProviderEnum.ProphetNet.value,
                LLMProviderEnum.MBart.value,
                LLMProviderEnum.MarianMT.value,
                LLMProviderEnum.XLM.value,
                LLMProviderEnum.XLM_R.value,
                LLMProviderEnum.TuringNLG.value,
                LLMProviderEnum.DialoGPT.value,
                LLMProviderEnum.BlenderBot.value,
                LLMProviderEnum.Meena.value,
                LLMProviderEnum.LaMDA.value,
                LLMProviderEnum.DALLE.value,
                LLMProviderEnum.CLIP.value,
                LLMProviderEnum.VQ_VAE_2,
                LLMProviderEnum.StyleGAN,
                LLMProviderEnum.BigGAN,
                LLMProviderEnum.StyleGAN2,
                LLMProviderEnum.StyleGAN3,
                LLMProviderEnum.BigGAN_512px,
                LLMProviderEnum.StyleGAN2_ADA,
                LLMProviderEnum.StyleGAN2_ADA_512px,
                LLMProviderEnum.StyleGAN2_ADA_1024px
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

        if api_key:
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
        st.write("Retriever created successfully.")

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
            ChatProfileRoleEnum.AI.value: "AI",
            ChatProfileRoleEnum.HUMAN.value: "Human",
        }

        for msg in messages.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)

# Get User Input and Generate Response

if user_query := st.chat_input(placeholder="Ask me Anything!", disabled=(not uploaded_files)):
    st.chat_message("Human").write(user_query)
    with st.chat_message("AI"):
        retrieval_handler = PrintRetrievalHandler(st.empty())
        stream_handler = StreamHandler(st.empty())
        response = chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

if selected_model and model_name:
    st.sidebar.caption(f"Using {selected_model} model.")