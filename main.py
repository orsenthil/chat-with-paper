import os
import tempfile
from enum import Enum

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

from langchain_text_splitters import RecursiveCharacterTextSplitter

GPT3_LLM_MODEL = "gpt-3.5-turbo"
GPT4_LLM_MODEL = "gpt-4"
GPT4o_LLM_MODEL = "gpt-4-o"


def get_document_text(files, use_compression=False):
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



