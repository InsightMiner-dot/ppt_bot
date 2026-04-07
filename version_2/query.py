import logging
import re
import sys
from pathlib import Path

from langchain_chroma import Chroma

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from scripts import llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _extract_target_phrase(question: str) -> str:
    patterns = [
        r"what\s+comments?\s+(?:are\s+)?(?:present|on|in|from)\s+(.+)",
        r"comments?\s+(?:present|on|in|from)\s+(.+)",
        r"slide\s+title\s+(.+)",
    ]
    cleaned_question = question.strip().rstrip("?.!")
    for pattern in patterns:
        match = re.search(pattern, cleaned_question, re.IGNORECASE)
        if match:
            return match.group(1).strip(" \"'")
    return ""


def _score_chunk(question: str, chunk) -> int:
    target_phrase = _extract_target_phrase(question)
    if not target_phrase:
        return 0

    normalized_target = _normalize(target_phrase)
    if not normalized_target:
        return 0

    score = 0
    title_hint = _normalize(str(chunk.metadata.get("slide_title", "")))
    if title_hint:
        if title_hint == normalized_target:
            score += 10
        elif title_hint.startswith(normalized_target + " "):
            score -= 2
        elif normalized_target in title_hint:
            score += 3

    if chunk.metadata.get("content_type") == "comments":
        score += 2

    for line in chunk.page_content.splitlines():
        normalized_line = _normalize(line)
        if not normalized_line:
            continue
        if normalized_line == normalized_target:
            score += 6
        elif normalized_line.startswith(normalized_target + " "):
            score -= 1
        elif normalized_target in normalized_line:
            score += 2

    return score


def query_presentations(question: str):
    db_directory = "./chroma_db_simple"

    logging.info("Loading Chroma DB...")
    try:
        vectorstore = Chroma(
            persist_directory=db_directory,
            embedding_function=llm.embeddings,
        )
    except Exception as exc:
        logging.error(f"Failed to load Chroma DB. Did you run ingest.py first? Error: {exc}")
        return

    logging.info(f"Searching for: '{question}'...")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 24},
    )
    retrieved_chunks = retriever.invoke(question)
    relevant_chunks = sorted(
        retrieved_chunks,
        key=lambda chunk: _score_chunk(question, chunk),
        reverse=True,
    )[:6]

    if not relevant_chunks:
        print("No relevant information found in the database.")
        return

    context = "Here are the most relevant retrieved slide chunks:\n\n"
    for idx, chunk in enumerate(relevant_chunks, start=1):
        meta = chunk.metadata
        context += f"### Retrieved Chunk {idx} ###\n"
        context += (
            f"[Source: {meta.get('filename', 'Unknown')}, "
            f"Slide: {meta.get('slide', 'Unknown')}, "
            f"Type: {meta.get('content_type', 'Unknown')}]\n"
        )
        context += f"Content:\n{chunk.page_content}\n\n"
        context += "-" * 40 + "\n\n"

    instruction = """

    IMPORTANT INSTRUCTION FOR REFERENCES:
    At the very end of your response, you MUST include a comprehensive "References" section citing the exact documents you used to formulate your answer.
    Format it as a bulleted list exactly like this:
    - Filename: [Insert Filename], Slide Number: [Insert Slide]

    MATCHING GUIDANCE:
    Prefer chunks whose slide title most closely matches the user question.
    If similar slide titles appear, avoid mixing them unless the retrieved evidence clearly supports it.
    If relevant evidence is present in the retrieved chunks, answer from it directly instead of saying "I don't know".
    If the evidence is partial, give the partial answer and briefly say what is unclear.
    """

    logging.info("Sending retrieved context to Azure OpenAI...")
    try:
        response = llm.ask_llm(question=question + instruction, context=context)
        print("\n" + "=" * 50)
        print("LLM RESPONSE:")
        print("=" * 50)
        print(response)

        with open(r"response_simple.md", "w", encoding="utf-8") as file_handle:
            file_handle.write(response)
    except Exception as exc:
        logging.error(f"Error querying the LLM: {exc}")


if __name__ == "__main__":
    user_query = "What comments are present in November MTD vs FC7+5?"
    query_presentations(user_query)
