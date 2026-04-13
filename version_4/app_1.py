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

### Question:
{question}

IMPORTANT:
At the end of the answer, include a "References" section with Source, Filename, Last Modified, and Page Number.

### Answer:"""
    )

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    return prompt | llm | StrOutputParser()


def ask_llm(context, question):
    chain = build_chain()
    return chain.invoke({"context": context, "question": question})


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

                reference = (
                    f"Source: {source}, Filename: {filename}, "
                    f"Last Modified: {last_modified}, Page Number: {page_number}"
                )
                context_parts.append(
                    f"## File: {filename} | Page: {page_number}\n"
                    f"[Reference Metadata -> {reference}]\n"
                    f"{content}"
                )

    return "\n\n".join(context_parts), skipped_files


st.set_page_config(page_title="Multi-file RAG App", layout="wide")
st.title("Multi-file Question Answering")
st.caption("Upload multiple files and ask questions based on their content.")

with st.sidebar:
    st.subheader("Supported file types")
    st.write(", ".join(sorted(SUPPORTED_EXTENSIONS)))

uploaded_files = st.file_uploader(
    "Upload one or more files",
    type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    accept_multiple_files=True,
)
question = st.text_area("Ask your question", height=140)

if st.button("Get Answer", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing files and generating answer..."):
            try:
                context, skipped_files = build_context(uploaded_files)

                if not context.strip():
                    st.error("No readable content was found in the uploaded files.")
                else:
                    answer = ask_llm(context, question.strip())
                    st.subheader("Answer")
                    st.write(answer)

                if skipped_files:
                    st.subheader("Skipped files")
                    for skipped_file in skipped_files:
                        st.write(f"- {skipped_file}")

            except Exception as exc:
                st.error(str(exc))

st.markdown(
    """
Run with:

```bash
streamlit run app.py
```
"""
)
