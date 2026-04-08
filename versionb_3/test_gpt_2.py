import hashlib
from collections import defaultdict

from langchain_community.document_loaders import UnstructuredPowerPointLoader
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
# ✅ VERSION
# =========================
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# =========================
# ✅ LOAD PPT (ELEMENT MODE)
# =========================
def load_ppt_elements(file_path):
    loader = UnstructuredPowerPointLoader(
        file_path,
        mode="elements"   # 🔥 KEY
    )
    return loader.load()


# =========================
# ✅ GROUP BY SLIDE
# =========================
def group_by_slide(elements, file_name, version):

    slides = defaultdict(lambda: {
        "title": None,
        "comments": []
    })

    for el in elements:
        meta = el.metadata

        slide_num = meta.get("page_number")
        category = meta.get("category")  # Title, ListItem, NarrativeText

        text = el.page_content.strip()

        if not slide_num or not text:
            continue

        # Title
        if category == "Title":
            slides[slide_num]["title"] = text

        # Comments / bullets
        elif category in ["ListItem", "NarrativeText"]:
            slides[slide_num]["comments"].append(text)

    docs = []

    for slide, data in slides.items():
        content = "\n".join(data["comments"])

        docs.append({
            "content": content,
            "metadata": {
                "file_name": file_name,
                "version": version,
                "slide_number": slide,
                "slide_title": data["title"],
                "source": f"{file_name}_slide_{slide}"
            }
        })

    return docs


# =========================
# ✅ TO DOCUMENT
# =========================
def to_docs(docs):
    return [Document(page_content=d["content"], metadata=d["metadata"]) for d in docs]


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

        prefix = f"Slide Title: {meta.get('slide_title','')}\n"

        chunks = splitter.split_text(text)

        for idx, chunk in enumerate(chunks):
            meta_copy = meta.copy()
            meta_copy["chunk_id"] = idx
            meta_copy["chunk_uid"] = chunk_id(
                meta["file_name"],
                meta["slide_number"],
                idx,
                meta["version"]
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
    vs.add_texts(
        texts=[d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
        ids=[d.metadata["chunk_uid"] for d in docs]
    )


# =========================
# ✅ RETRIEVE (TITLE FIRST)
# =========================
def retrieve(vs, query, k=5):

    all_data = vs.get()

    title_docs = []

    for content, meta in zip(all_data["documents"], all_data["metadatas"]):
        title = (meta.get("slide_title") or "").lower()

        if title and title in query.lower():
            title_docs.append(Document(page_content=content, metadata=meta))

    if title_docs:
        return title_docs[:k]

    return vs.similarity_search(query, k=k)


# =========================
# ✅ CONTEXT
# =========================
def build_context(docs):
    return "\n\n".join([
        f"File: {d.metadata['file_name']} | Slide: {d.metadata['slide_number']}\n{d.page_content}"
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

    docs = retrieve(vs, query, k=5)

    if not docs:
        return "No data found."

    context = build_context(docs)

    prompt = f"""
Answer the question based on the slide content.

If asking for comments → return bullet points.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt).content

    refs = "\n".join({
        f"(File: {d.metadata['file_name']}, Slide: {d.metadata['slide_number']})"
        for d in docs
    })

    return response + "\n\nSources:\n" + refs


# =========================
# ✅ MAIN
# =========================
if __name__ == "__main__":

    ppt_file = "input.pptx"

    elements = load_ppt_elements(ppt_file)

    version = get_file_hash(ppt_file)
    docs = group_by_slide(elements, ppt_file, version)

    docs = to_docs(docs)
    chunks = chunk_documents(docs)

    emb = get_embedding()
    vs = get_vector_store(emb)

    upsert(vs, chunks)

    print("✅ New Pipeline Ready")

    llm = get_llm()

    query = "What are the comments in G&A Evaluation - MTD vs FC7+5"

    result = answer(query, vs, llm)

    print("\n=== ANSWER ===\n")
    print(result)
