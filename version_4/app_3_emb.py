import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Updated imports
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document  # Requested change

load_dotenv(override=True)

def get_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing env var: {name}")
    return value

def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=get_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=get_env("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=get_env("AZURE_OPENAI_ENDPOINT"),
        api_key=get_env("AZURE_OPENAI_KEY"),
    )

def process_ppt_to_chroma(uploaded_files):
    embeddings = get_embeddings()
    all_slide_docs = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = Path(temp_dir) / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())
            
            # Load elements
            loader = UnstructuredPowerPointLoader(str(file_path), mode="elements")
            elements = loader.load()

            # Group by slide number (Strictly 1 Slide = 1 Chunk)
            slides_content = {}
            for el in elements:
                page_num = el.metadata.get("page_number", 1)
                if page_num not in slides_content:
                    slides_content[page_num] = []
                slides_content[page_num].append(el.page_content)

            for page_num, contents in slides_content.items():
                all_slide_docs.append(Document(
                    page_content="\n".join(contents),
                    metadata={"source": uploaded_file.name, "page": page_num}
                ))

    # Initialize Chroma
    vectorstore = Chroma.from_documents(
        documents=all_slide_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    # Returns the VectorStoreRetriever
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(
        f"--- Slide {d.metadata['page']} (File: {d.metadata['source']}) ---\n{d.page_content}"
        for d in docs
    )

# --- UI ---
st.set_page_config(page_title="PPT Vector Chat", layout="wide")
st.title("🎯 Precise PPT Retriever")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    ppts = st.file_uploader("Upload PPTX", type=["pptx", "ppt"], accept_multiple_files=True)
    if st.button("Index Slides", type="primary"):
        if ppts:
            with st.spinner("Indexing..."):
                st.session_state.retriever = process_ppt_to_chroma(ppts)
                st.success("Indexing complete!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your slides..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.retriever:
        st.error("Please Index Slides first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching slides..."):
                # FIXED: Use .invoke() instead of .get_relevant_documents()
                context_docs = st.session_state.retriever.invoke(prompt)
                context_text = format_docs(context_docs)

                llm = AzureChatOpenAI(
                    deployment_name=get_env("AZURE_OPENAI_DEPLOYMENT"),
                    openai_api_version=get_env("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=get_env("AZURE_OPENAI_ENDPOINT"),
                    api_key=get_env("AZURE_OPENAI_KEY"),
                    temperature=0
                )

                template = """Answer using only the PPT context. If unknown, say "I don't know".
                Context: {context}
                Question: {question}
                Answer:"""
                
                chain = ChatPromptTemplate.from_template(template) | llm | StrOutputParser()
                response = chain.invoke({"context": context_text, "question": prompt})
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
