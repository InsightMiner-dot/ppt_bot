import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import AzureChatOpenAI

load_dotenv(override=True)

# Updated to only focus on PPT/PPTX as requested
SUPPORTED_EXTENSIONS = {".pptx", ".ppt"}

def get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value

def build_chain():
    llm = AzureChatOpenAI(
        openai_api_version=get_required_env("AZURE_OPENAI_API_VERSION"),
        model=get_required_env("MODEL_NAME"),
        deployment_name=get_required_env("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=get_required_env("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=get_required_env("AZURE_OPENAI_KEY"),
        openai_api_type="azure",
        temperature=0.1,
    )

    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful AI assistant who answers user questions using only the provided context."
    )

    human_prompt = HumanMessagePromptTemplate.from_template(
        """Answer the user question based only on the provided context.
If the answer is not in the context, say "I don't know".

### Context:
{context}

### Question:
{question}

IMPORTANT:
At the end of the answer, include a "References" section.

### Answer:"""
    )

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    return prompt | llm | StrOutputParser()

def save_uploaded_file(uploaded_file, directory):
    file_path = Path(directory) / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path

def build_context(uploaded_files):
    context_parts = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file, temp_dir)
            try:
                # Optimized specifically for PowerPoint
                loader = UnstructuredPowerPointLoader(str(file_path), mode="elements")
                docs = loader.load()
                
                for doc in docs:
                    content = (doc.page_content or "").strip()
                    if content:
                        metadata = doc.metadata
                        page = metadata.get("page_number", "Unknown")
                        context_parts.append(f"File: {uploaded_file.name} | Page: {page}\n{content}")
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
    
    return "\n\n".join(context_parts)

# --- UI Layout ---
st.set_page_config(page_title="PPT Chat Assistant", layout="wide")
st.title("📊 PowerPoint RAG Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for File Uploads
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PowerPoint files", 
        type=["ppt", "pptx"], 
        accept_multiple_files=True
    )
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your slides..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process and Generate Response
    if not uploaded_files:
        with st.chat_message("assistant"):
            msg = "Please upload at least one PowerPoint file to begin."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing slides..."):
                try:
                    context = build_context(uploaded_files)
                    if not context:
                        response = "I couldn't find any readable text in those slides."
                    else:
                        chain = build_chain()
                        response = chain.invoke({"context": context, "question": prompt})
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
