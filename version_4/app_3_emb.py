import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document

load_dotenv(override=True)

# --- Configuration Helpers ---
def get_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing env var: {name}")
    return value

# --- Core RAG Logic ---
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
            
            # Load elements from PPT
            loader = UnstructuredPowerPointLoader(str(file_path), mode="elements")
            elements = loader.load()

            # Group elements by slide number to ensure 1 slide = 1 chunk
            slides_content = {}
            for el in elements:
                page_num = el.metadata.get("page_number", 1)
                if page_num not in slides_content:
                    slides_content[page_num] = []
                slides_content[page_num].append(el.page_content)

            # Create individual Documents per slide
            for page_num, contents in slides_content.items():
                merged_text = "\n".join(contents)
                all_slide_docs.append(Document(
                    page_content=merged_text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_num
                    }
                ))

    # Initialize Chroma (persist_directory creates a local folder 'chroma_db')
    vectorstore = Chroma.from_documents(
        documents=all_slide_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    formatted = []
    for doc in docs:
        formatted.append(f"--- Slide {doc.metadata['page']} (File: {doc.metadata['source']}) ---\n{doc.page_content}")
    return "\n\n".join(formatted)

# --- UI Setup ---
st.set_page_config(page_title="PPT Vector Chat", layout="wide")
st.title("🎯 Precise PPT Retriever")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    st.header("Setup")
    ppts = st.file_uploader("Upload PPTX", type=["pptx", "ppt"], accept_multiple_files=True)
    if st.button("Index Slides", type="primary"):
        if ppts:
            with st.spinner("Embedding slides into ChromaDB..."):
                st.session_state.retriever = process_ppt_to_chroma(ppts)
                st.success("Indexing complete!")
        else:
            st.warning("Upload files first.")

# --- Chat Interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What is mentioned on the slides?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.retriever:
        st.error("Please upload and 'Index Slides' in the sidebar first.")
    else:
        with st.chat_message("assistant"):
            # Retrieval and Generation
            retriever = st.session_state.retriever
            llm = AzureChatOpenAI(
                deployment_name=get_env("AZURE_OPENAI_DEPLOYMENT"),
                openai_api_version=get_env("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=get_env("AZURE_OPENAI_ENDPOINT"),
                api_key=get_env("AZURE_OPENAI_KEY"),
                temperature=0
            )

            # Build RAG Chain
            template = """You are a precise assistant. Use the following PPT slide context to answer the question. 
If the answer isn't there, say you don't know. 
Always cite the File Name and Slide Number.

Context:
{context}

Question: {question}
Answer:"""
            
            rag_prompt = ChatPromptTemplate.from_template(template)
            
            # Chain execution
            context_docs = retriever.get_relevant_documents(prompt)
            context_text = format_docs(context_docs)
            
            chain = rag_prompt | llm | StrOutputParser()
            response = chain.invoke({"context": context_text, "question": prompt})
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
