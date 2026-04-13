import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# LangChain Core & OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Vector Store & Loaders
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.vectorstores import Chroma

load_dotenv(override=True)

# --- Configuration Helpers ---
def get_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing env var: {name}")
    return value

def get_llm(temperature=0):
    return AzureChatOpenAI(
        deployment_name=get_env("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=get_env("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=get_env("AZURE_OPENAI_ENDPOINT"),
        api_key=get_env("AZURE_OPENAI_KEY"),
        temperature=temperature
    )

def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=get_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=get_env("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=get_env("AZURE_OPENAI_ENDPOINT"),
        api_key=get_env("AZURE_OPENAI_KEY"),
    )

# --- Processing Logic ---
def process_ppt_to_chroma(uploaded_files):
    embeddings = get_embeddings()
    all_slide_docs = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = Path(temp_dir) / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())
            
            loader = UnstructuredPowerPointLoader(str(file_path), mode="elements")
            elements = loader.load()

            # STRICT: Group by slide number
            slides_dict = {}
            for el in elements:
                page_num = el.metadata.get("page_number", 1)
                if page_num not in slides_dict:
                    slides_dict[page_num] = []
                slides_dict[page_num].append(el.page_content)

            for page_num, contents in slides_dict.items():
                all_slide_docs.append(Document(
                    page_content="\n".join(contents),
                    metadata={"source": uploaded_file.name, "page": page_num}
                ))

    vectorstore = Chroma.from_documents(
        documents=all_slide_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

# --- Query Enhancer Logic ---
def get_enhanced_queries(original_query: str) -> List[str]:
    """Rewrites the user query into 3 variations to improve retrieval."""
    llm = get_llm(temperature=0.7)
    
    template = """You are an AI assistant. Your task is to generate 3 different versions 
    of the given user question to retrieve relevant documents from a vector database. 
    By providing multiple perspectives on the user question, your goal is to help 
    the user overcome some of the limitations of distance-based similarity search. 
    
    Original question: {question}
    
    Provide these alternative questions separated by newlines. Do not add numbers.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"question": original_query})
    queries = [q.strip() for q in response.split("\n") if q.strip()]
    # Include original query as well
    queries.append(original_query)
    return list(set(queries))

def format_docs(docs):
    return "\n\n".join(
        f"--- Slide {d.metadata['page']} (File: {d.metadata['source']}) ---\n{d.page_content}"
        for d in docs
    )

# --- UI Layout ---
st.set_page_config(page_title="Enhanced PPT RAG", layout="wide")
st.title("🚀 Powerpoint RAG with Query Enhancement")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:
    st.header("1. Data Ingestion")
    ppts = st.file_uploader("Upload Files", type=["pptx", "ppt"], accept_multiple_files=True)
    if st.button("Index Slides", type="primary"):
        if ppts:
            with st.spinner("Creating Vector Store..."):
                st.session_state.vectorstore = process_ppt_to_chroma(ppts)
                st.success("Indexing Complete!")
        else:
            st.warning("Please upload files first.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Logic
if prompt := st.chat_input("What would you like to know from the slides?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.vectorstore:
        st.error("Vector database not found. Please index slides in the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Enhancing query and searching..."):
                # 1. Enhance Query
                enhanced_queries = get_enhanced_queries(prompt)
                
                # 2. Retrieve for all queries (Multi-Query approach)
                all_relevant_docs = []
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
                
                for q in enhanced_queries:
                    docs = retriever.invoke(q)
                    all_relevant_docs.extend(docs)
                
                # Deduplicate docs based on source and page
                unique_docs = {f"{d.metadata['source']}_{d.metadata['page']}": d for d in all_relevant_docs}.values()
                context_text = format_docs(unique_docs)

                # 3. Final Generation
                llm = get_llm(temperature=0)
                qa_template = """Answer the question strictly using the provided PPT context.
                If the answer is not in the context, state that the information is missing.
                
                Context:
                {context}
                
                Question: {question}
                Answer:"""
                
                qa_prompt = ChatPromptTemplate.from_template(qa_template)
                chain = qa_prompt | llm | StrOutputParser()
                
                response = chain.invoke({"context": context_text, "question": prompt})
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Debugging Info (Optional)
                with st.expander("System Trace"):
                    st.write("**Enhanced Queries Used:**")
                    st.write(enhanced_queries)
                    st.write(f"**Unique Slides Retrieved:** {len(unique_docs)}")
