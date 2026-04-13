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
    for chunk in text.split():
        yield chunk + " "
        time.sleep(0.02)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "context" not in st.session_state:
    st.session_state.context = ""

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

if "skipped_files" not in st.session_state:
    st.session_state.skipped_files = []

if "pending_files" not in st.session_state:
    st.session_state.pending_files = []


st.set_page_config(page_title="Multi-file Chatbot", layout="wide")
st.title("Multi-file Chatbot")
st.caption("A chatbot-style document assistant with custom chat controls.")

with st.sidebar:
    st.subheader("Documents")
    pending_files = st.multiselect(
        "Selected files",
        options=st.session_state.uploaded_names,
        default=st.session_state.uploaded_names,
        disabled=True,
    )
    st.write(f"Loaded: {len(pending_files)}")

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
        st.session_state.pending_files = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


with st.container():
    with st.form("chatbot_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Attach files",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        user_question = st.text_area(
            "Type your message",
            placeholder="Ask a question about your files...",
            height=100,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send")


if submitted:
    user_parts = []
    if user_question.strip():
        user_parts.append(user_question.strip())
    if uploaded_files:
        user_parts.append(
            "Attached files: " + ", ".join(uploaded_file.name for uploaded_file in uploaded_files)
        )

    user_message = "\n".join(user_parts) if user_parts else "Uploaded file(s)"
    st.session_state.messages.append({"role": "user", "content": user_message})

    with st.chat_message("user"):
        st.write(user_message)

    with st.chat_message("assistant"):
        with st.status("Working on your request...", expanded=True) as status:
            try:
                if uploaded_files:
                    status.write(f"Processing {len(uploaded_files)} file(s)...")
                    new_context, added_files, skipped_files = build_context(uploaded_files)
                    if new_context:
                        if st.session_state.context:
                            st.session_state.context += "\n\n" + new_context
                        else:
                            st.session_state.context = new_context
                        for added_file in added_files:
                            if added_file not in st.session_state.uploaded_names:
                                st.session_state.uploaded_names.append(added_file)
                    st.session_state.skipped_files.extend(skipped_files)
                    status.write("Files processed.")

                if not st.session_state.context:
                    answer = "Please attach at least one supported file so I can answer from its content."
                elif not user_question.strip():
                    answer = "Your files are ready. Ask me a question about the uploaded content."
                else:
                    status.write("Preparing chat history...")
                    chat_history = format_chat_history(st.session_state.messages[:-1])
                    status.write("Calling Azure OpenAI...")
                    answer = ask_llm(
                        st.session_state.context,
                        user_question.strip(),
                        chat_history,
                    )
                    status.write("Streaming response...")

                streamed_answer = st.write_stream(stream_text(answer))
                st.session_state.messages.append(
                    {"role": "assistant", "content": streamed_answer}
                )
                status.update(label="Done", state="complete", expanded=False)
            except Exception as exc:
                error_message = str(exc)
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
                status.update(label="Failed", state="error", expanded=True)
