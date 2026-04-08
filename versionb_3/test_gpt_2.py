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
# ✅ EXTRACTION
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

            if shape.has_text_frame:
                text = shape.text.strip()
                if text:
                    slide_text.append(text)
                    if not slide_title:
                        slide_title = text

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
# ✅ TO DOCUMENT
# =========================
def to_docs(docs):
    return [
        Document(page_content=d["content"], metadata=d["metadata"])
        for d in docs
    ]


# =========================
# ✅ CHUNK ID
# =========================
def chunk_id(file, slide, idx, version):
    return hashlib.md5(f"{file}_{slide}_{idx}_{version}".encode()).hexdigest()


# =========================
# ✅ CHUNKING
# =========================
def chunk_documents(docs, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    final_docs = []

    for doc in docs:
        text = doc.page_content.strip()
        meta = doc.metadata.copy()

        if not text:
            continue

        # 🔥 TITLE BOOST
        prefix = f"""
Slide Title: {meta.get('slide_title', '')}
Title: {meta.get('slide_title', '')}
"""

        if len(text) <= chunk_size:
            meta["chunk_id"] = 0
            meta["chunk_uid"] = chunk_id(
                meta["file_name"], meta["slide_number"], 0, meta["version"]
            )

            final_docs.append(
                Document(page_content=prefix + text, metadata=meta)
            )
            continue

        chunks = splitter.split_text(text)

        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            meta_copy = meta.copy()
            meta_copy["chunk_id"] = idx
            meta_copy["chunk_uid"] = chunk_id(
                meta["file_name"], meta["slide_number"], idx, meta["version"]
            )

            final_docs.append(
                Document(page_content=prefix + chunk, metadata=meta_copy)
            )

    return final_docs


# =========================
# ✅ EMBEDDING
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
def upsert(vs, docs):
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    ids = [d.metadata["chunk_uid"] for d in docs]

    vs.add_texts(texts=texts, metadatas=metas, ids=ids)


# =========================
# ✅ TITLE-AWARE RETRIEVAL
# =========================
def retrieve(vs, query, k=5):

    # Semantic
    docs = vs.similarity_search(query, k=k)

    # Title match (🔥 critical)
    all_data = vs.get()
    title_docs = []

    for content, meta in zip(all_data["documents"], all_data["metadatas"]):
        title = (meta.get("slide_title") or "").lower()

        if any(word in title for word in query.lower().split()):
            title_docs.append(
                Document(page_content=content, metadata=meta)
            )

    # Merge + dedupe
    seen = set()
    final = []

    for d in title_docs + docs:
        uid = d.metadata["chunk_uid"]
        if uid not in seen:
            seen.add(uid)
            final.append(d)

    return final[:k]


# =========================
# ✅ CONTEXT
# =========================
def build_context(docs):
    return "\n\n".join([
        f"Slide {d.metadata['slide_number']}:\n{d.page_content}"
        for d in docs
    ])


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
# ✅ QA
# =========================
def answer(query, vs, llm):

    docs = retrieve(vs, query, k=8)

    if not docs:
        return "No data found."

    context = build_context(docs)

    prompt = f"""
Extract ALL comments or bullet points from the context.

Return ONLY as bullet list.

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

    # 1. Ingest
    raw = extract_ppt(ppt_file)

    # 2. Convert
    docs = to_docs(raw)

    # 3. Chunk
    chunks = chunk_documents(docs)

    # 4. Embed
    emb = get_embedding()
    vs = get_vector_store(emb)
    upsert(vs, chunks)

    print("✅ Pipeline Ready")

    # 5. Query
    llm = get_llm()

    q = "What are the comments in G&A Evaluation - MTD vs FC7+5"

    res = answer(q, vs, llm)

    print("\n=== ANSWER ===\n")
    print(res)
