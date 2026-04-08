import os
import hashlib
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma


# =========================
# ✅ CONFIG
# =========================
PERSIST_DIR = "chroma_db"

AZURE_OPENAI_API_KEY = "YOUR_KEY"
AZURE_OPENAI_ENDPOINT = "YOUR_ENDPOINT"

EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
CHAT_DEPLOYMENT = "gpt-4o-mini"
API_VERSION = "2024-02-01"


# =========================
# ✅ VERSION (FILE HASH)
# =========================
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# =========================
# ✅ EXTRACTION (TEXT + CHARTS ONLY)
# =========================
def extract_ppt(file_path):
    prs = Presentation(file_path)
    file_name = os.path.basename(file_path)
    version = get_file_hash(file_path)

    documents = []

    for i, slide in enumerate(prs.slides):
        slide_text = []
        slide_title = None

        for shape in slide.shapes:

            # TEXT
            if shape.has_text_frame:
                text = shape.text.strip()
                if text:
                    slide_text.append(text)
                    if not slide_title:
                        slide_title = text

            # CHART
            if shape.shape_type == MSO_SHAPE_TYPE.CHART:
                chart = shape.chart

                if chart.has_title:
                    slide_text.append(f"Chart: {chart.chart_title.text_frame.text}")

                for series in chart.series:
                    slide_text.append(
                        f"{series.name} values: {list(series.values)}"
                    )

        documents.append({
            "content": " ".join(slide_text),
            "metadata": {
                "file_name": file_name,
                "version": version,
                "slide_number": i + 1,
                "slide_title": slide_title,
                "source": f"{file_name}_slide_{i+1}"
            }
        })

    return documents


# =========================
# ✅ CONVERT TO DOCUMENT
# =========================
def to_langchain_docs(docs):
    return [
        Document(page_content=d["content"], metadata=d["metadata"])
        for d in docs
    ]


# =========================
# ✅ CHUNK ID
# =========================
def generate_chunk_id(file_name, slide_number, chunk_id, version):
    base = f"{file_name}_{slide_number}_{chunk_id}_{version}"
    return hashlib.md5(base.encode()).hexdigest()


# =========================
# ✅ CHUNKING
# =========================
def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = []

    for doc in documents:
        text = doc.page_content.strip()
        meta = doc.metadata.copy()

        if not text:
            continue

        context_prefix = f"Slide Title: {meta.get('slide_title', '')}\n"

        if len(text) <= chunk_size:
            meta["chunk_id"] = 0
            meta["chunk_uid"] = generate_chunk_id(
                meta["file_name"], meta["slide_number"], 0, meta["version"]
            )

            chunked_docs.append(
                Document(page_content=context_prefix + text, metadata=meta)
            )
            continue

        chunks = splitter.split_text(text)

        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue

            meta_copy = meta.copy()
            meta_copy["chunk_id"] = idx
            meta_copy["chunk_uid"] = generate_chunk_id(
                meta["file_name"], meta["slide_number"], idx, meta["version"]
            )

            chunked_docs.append(
                Document(page_content=context_prefix + chunk, metadata=meta_copy)
            )

    return chunked_docs


# =========================
# ✅ EMBEDDING MODEL
# =========================
def get_embedding():
    return AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_DEPLOYMENT,
        openai_api_version=API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )


# =========================
# ✅ VECTOR STORE
# =========================
def get_vector_store(embedding):
    return Chroma(
        collection_name="ppt_rag",
        embedding_function=embedding,
        persist_directory=PERSIST_DIR
    )


# =========================
# ✅ UPSERT
# =========================
def upsert_documents(vector_store, docs, batch_size=50):
    texts, metas, ids = [], [], []

    for d in docs:
        texts.append(d.page_content)
        metas.append(d.metadata)
        ids.append(d.metadata["chunk_uid"])

    for i in range(0, len(texts), batch_size):
        vector_store.add_texts(
            texts=texts[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )


# =========================
# ✅ LLM
# =========================
def get_llm():
    return AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT,
        openai_api_version=API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        temperature=0
    )


# =========================
# ✅ RETRIEVE
# =========================
def retrieve(vector_store, query, k=5, filters=None):
    return vector_store.similarity_search(query, k=k, filter=filters)


# =========================
# ✅ CONTEXT BUILDER
# =========================
def build_context(docs):
    return "\n\n".join([
        f"Source: {d.metadata['source']}\n{d.page_content}"
        for d in docs
    ])


# =========================
# ✅ QA
# =========================
def answer(query, vector_store, llm, filters=None):
    docs = retrieve(vector_store, query, filters=filters)

    if not docs:
        return "No relevant data found."

    context = build_context(docs)

    prompt = f"""
Answer using ONLY the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt).content


# =========================
# ✅ MAIN
# =========================
if __name__ == "__main__":
    ppt_file = "input.pptx"

    # 1. Extraction
    raw_docs = extract_ppt(ppt_file)

    # 2. Convert
    lc_docs = to_langchain_docs(raw_docs)

    # 3. Chunk
    chunked_docs = chunk_documents(lc_docs)

    # 4. Embed + Store
    embedding = get_embedding()
    vector_store = get_vector_store(embedding)
    upsert_documents(vector_store, chunked_docs)

    print("✅ Ingestion + Embedding Done")

    # 5. Query
    llm = get_llm()

    query = "What is the revenue trend?"
    result = answer(query, vector_store, llm)

    print("\n=== ANSWER ===\n")
    print(result)
