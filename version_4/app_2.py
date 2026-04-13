import os
import tempfile
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

SUPPORTED_EXTENSIONS = ["pptx", "ppt", "pdf", "docx", "doc", "txt", "md", "csv"]


def get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_chain():
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
        "You are a helpful document chatbot. Answer only from the provided context."
    )
    human_prompt = HumanMessagePromptTemplate.from_template(
        """Answer the user's question using only the context below.
If the answer is not available in the context, say "I don't know".

Context:
{context}

Question:
{question}

At the end, add a short "References" section with the file name and page number you used.
"""
    )

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    return prompt | llm | StrOutputParser()


def get_loader(file_path):
    suffix = file_path.suffix.lower()
    if suffix in {".pptx", ".ppt"}:
        return UnstructuredPowerPointLoader(str(file_path), mode="elements")
    return UnstructuredFileLoader(str(file_path), mode="elements")


def build_context(uploaded_files):
    context_parts = []
    loaded_files = []
    skipped_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = Path(temp_dir) / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())

            try:
                docs = get_loader(file_path).load()
            except Exception as exc:
                skipped_files.append(f"{uploaded_file.name} ({exc})")
                continue

            has_text = False
            for index, doc in enumerate(docs, start=1):
                content = (doc.page_content or "").strip()
                if not content:
                    continue

                has_text = True
                metadata = doc.metadata or {}
                page_number = (
                    metadata.get("page_number")
                    or metadata.get("page")
                    or metadata.get("page_index")
                    or index
                )
                filename = metadata.get("filename", uploaded_file.name)
                context_parts.append(
                    f"File: {filename}\nPage: {page_number}\nContent:\n{content}"
                )

            if has_text:
                loaded_files.append(uploaded_file.name)
            else:
                skipped_files.append(f"{uploaded_file.name} (no readable text found)")

    return "\n\n".join(context_parts), loaded_files, skipped_files


def ask_question(question):
    chain = get_chain()
    return chain.invoke(
        {
            "context": st.session_state.context,
            "question": question,
        }
    )


def handle_user_question(question):
    st.session_state.messages.append({"role": "user", "content": question})

    if not st.session_state.context:
        answer = "Please upload and process files first."
    else:
        answer = ask_question(question)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload documents from the sidebar and ask me anything about them.",
        }
    ]

if "context" not in st.session_state:
    st.session_state.context = ""

if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []

if "skipped_files" not in st.session_state:
    st.session_state.skipped_files = []


st.set_page_config(page_title="Document Chatbot", layout="wide")
st.title("Document Chatbot")

with st.sidebar:
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
    )

    if st.button("Process Files", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing files..."):
                context, loaded_files, skipped_files = build_context(uploaded_files)

            if not context:
                st.error("No readable content found in the uploaded files.")
            else:
                st.session_state.context = context
                st.session_state.loaded_files = loaded_files
                st.session_state.skipped_files = skipped_files
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Files are ready. Ask me questions about your documents.",
                    }
                ]
                st.success(f"Processed {len(loaded_files)} file(s).")

    if st.session_state.loaded_files:
        st.subheader("Loaded")
        for name in st.session_state.loaded_files:
            st.write(f"- {name}")

    if st.session_state.skipped_files:
        st.subheader("Skipped")
        for name in st.session_state.skipped_files:
            st.write(f"- {name}")

    if st.button("Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Ask a new question whenever you're ready.",
            }
        ]


question = st.chat_input("Ask a question about your files")

if question:
    with st.spinner("Thinking..."):
        handle_user_question(question)
    st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
