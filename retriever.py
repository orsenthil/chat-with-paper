import os
import tempfile

import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter


def document_retriever(files, use_compression=False):
    """
    Create a retriever from a list of files uploaded by the user.

    :param files:  list of files provided by the user.
    :param use_compression: flag to use compression while retrieving documents.
    :return: retriever object.
    """
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
    vectordb_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    if not use_compression:
        return vectordb_retriever

    compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vectordb_retriever)
