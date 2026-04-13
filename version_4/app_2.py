import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import AzureChatOpenAI


load_dotenv(override=True)

SUPPORTED_EXTENSIONS = {
    ".pptx",
    ".ppt",
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".md",
    ".csv",
}


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
If multiple files are uploaded, compare them when needed.

### Context:
{context}

### Chat History:
{chat_history}

### Question:
{question}

IMPORTANT:
At the end of the answer, include a "References" section with Source, Filename, Last Modified, and Page Number.

### Answer:"""
    )

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    return prompt | llm | StrOutputParser()


def ask_llm(context, question, chat_history):
    chain = build_chain()
    return chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": chat_history,
        }
    )


def save_uploaded_file(uploaded_file, directory):
    file_path = Path(directory) / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def get_loader(file_path):
    extension = file_path.suffix.lower()
    if extension in {".pptx", ".ppt"}:
        return UnstructuredPowerPointLoader(str(file_path), mode="elements")
    return UnstructuredFileLoader(str(file_path), mode="elements")


def build_context(uploaded_files):
    context_parts = []
    added_files = []
    skipped_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            extension = Path(uploaded_file.name).suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                skipped_files.append(uploaded_file.name)
                continue

            file_path = save_uploaded_file(uploaded_file, temp_dir)

            try:
                loader = get_loader(file_path)
                docs = loader.load()
            except Exception as exc:
                skipped_files.append(f"{uploaded_file.name} ({exc})")
                continue

            has_content = False
            for index, doc in enumerate(docs, start=1):
                metadata = dict(doc.metadata or {})
                page_number = (
                    metadata.get("page_number")
                    or metadata.get("page")
                    or metadata.get("page_index")
                    or index
                )
                source = metadata.get("source", str(file_path))
                last_modified = metadata.get("last_modified", "Unknown Date")
                filename = metadata.get("filename", uploaded_file.name)
                content = (doc.page_content or "").strip()

                if not content:
                    continue

                has_content = True
                reference = (
                    f"Source: {source}, Filename: {filename}, "
                    f"Last Modified: {last_modified}, Page Number: {page_number}"
                )
                context_parts.append(
                    f"## File: {filename} | Page: {page_number}\n"
                    f"[Reference Metadata -> {reference}]\n"
                    f"{content}"
                )

            if has_content:
                added_files.append(uploaded_file.name)

    return "\n\n".join(context_parts), added_files, skipped_files


def format_chat_history(messages):
    history_lines = []
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {message['content']}")
    return "\n".join(history_lines)


def stream_text(text):
    for chunk in text.split(" "):
        yield chunk + " "
        time.sleep(0.02)


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "context" not in st.session_state:
    st.session_state.context = ""

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

if "skipped_files" not in st.session_state:
    st.session_state.skipped_files = []


# --- Page Config & UI Layout ---
st.set_page_config(page_title="Multi-file Chatbot", layout="wide")
st.title("Multi-file Chatbot")
st.caption("Upload files from the sidebar, then chat with your documents.")

with st.sidebar:
    st.subheader("File uploader")
    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )

    if st.button("Process Files", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.status("Processing uploaded files...", expanded=True) as status:
                context, added_files, skipped_files = build_context(uploaded_files)

                if not context.strip():
                    status.update(
                        label="No readable content found",
                        state="error",
                        expanded=True,
                    )
                    st.error("No readable content was found in the uploaded files.")
                else:
                    st.session_state.context = context
                    st.session_state.uploaded_names = added_files
                    st.session_state.skipped_files = skipped_files
                    st.session_state.messages = []
                    status.write(f"Processed {len(added_files)} file(s).")
                    if skipped_files:
                        status.write(f"Skipped {len(skipped_files)} file(s).")
                    status.update(label="Files ready", state="complete", expanded=False)

    st.subheader("Loaded files")
    if st.session_state.uploaded_names:
        for name in st.session_state.uploaded_names:
            st.write(f"- {name}")
    else:
        st.write("No files processed yet.")

    if st.session_state.skipped_files:
        st.subheader("Skipped files")
        for skipped_file in st.session_state.skipped_files:
            st.write(f"- {skipped_file}")

    if st.button("Clear Chat"):
        st.session_state.messages = []

    if st.button("Reset All"):
        st.session_state.messages = []
        st.session_state.context = ""
        st.session_state.uploaded_names = []
        st.session_state.skipped_files = []


# --- Main Chat Interface ---

# 1. Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle new user input
if user_question := st.chat_input("Ask a question about the uploaded files"):
    
    # Display and save user message
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        if not st.session_state.context:
            error_msg = "Please upload and process files from the sidebar first."
            st.warning(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            try:
                # Simple spinner blocks the UI cleanly while LLM generates the full answer
                with st.spinner("Searching documents and generating answer..."):
                    chat_history = format_chat_history(st.session_state.messages[:-1])
                    
                    answer = ask_llm(
                        st.session_state.context,
                        user_question,
                        chat_history,
                    )
                
                # Stream the final text directly into the chat bubble
                streamed_answer = st.write_stream(stream_text(answer))
                
                # Save assistant response to state
                st.session_state.messages.append(
                    {"role": "assistant", "content": streamed_answer}
                )
                
            except Exception as exc:
                error_message = f"An error occurred: {str(exc)}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
