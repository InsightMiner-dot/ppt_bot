import logging
import sys
from pathlib import Path

from langchain_chroma import Chroma

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from scripts import llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def query_presentations(question: str):
    db_directory = "./chroma_db"

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
        search_kwargs={"k": 6, "fetch_k": 12},
    )
    relevant_chunks = retriever.invoke(question)

    if not relevant_chunks:
        print("No relevant information found in the database.")
        return

    context = "Here are the most relevant retrieved slide chunks:\n\n"
    for idx, chunk in enumerate(relevant_chunks, start=1):
        meta = chunk.metadata
        context += f"### Retrieved Chunk {idx} ###\n"
        context += (
            f"[Source: {meta.get('filename', 'Unknown')}, "
            f"Page: {meta.get('page', 'Unknown')}, "
            f"Type: {meta.get('content_type', 'Unknown')}]\n"
        )
        context += f"Content:\n{chunk.page_content}\n\n"
        context += "-" * 40 + "\n\n"

    instruction = """

    IMPORTANT INSTRUCTION FOR REFERENCES:
    At the very end of your response, you MUST include a comprehensive "References" section citing the exact documents you used to formulate your answer.
    Format it as a bulleted list exactly like this:
    - Filename: [Insert Filename], Page Number: [Insert Page]
    """

    logging.info("Sending retrieved context to Azure OpenAI...")
    try:
        response = llm.ask_llm(question=question + instruction, context=context)

        print("\n" + "=" * 50)
        print("LLM RESPONSE:")
        print("=" * 50)
        print(response)

        with open(r"response.md", "w", encoding="utf-8") as file_handle:
            file_handle.write(response)

    except Exception as exc:
        logging.error(f"Error querying the LLM: {exc}")


if __name__ == "__main__":
    user_query = "Summarize the key financial trends across all the Q3 reports."
    query_presentations(user_query)
