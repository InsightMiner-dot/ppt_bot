import logging
from langchain_chroma import Chroma
from scripts import llm  # Ensure this imports your configured llm and embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def query_presentations(question: str):
    db_directory = "./chroma_db"
    
    # 1. Load the existing Chroma Database
    logging.info("Loading Chroma DB...")
    try:
        vectorstore = Chroma(
            persist_directory=db_directory, 
            embedding_function=llm.embeddings
        )
    except Exception as e:
        logging.error(f"Failed to load Chroma DB. Did you run ingest.py first? Error: {e}")
        return

    # 2. Retrieve the top 5 most relevant chunks
    logging.info(f"Searching for: '{question}'...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_chunks = retriever.invoke(question)

    if not relevant_chunks:
        print("No relevant information found in the database.")
        return

    # 3. Format the retrieved chunks into our Context string
    context = "Here are the most relevant retrieved slide chunks:\n\n"
    for idx, chunk in enumerate(relevant_chunks, start=1):
        meta = chunk.metadata
        context += f"### Retrieved Chunk {idx} ###\n"
        context += f"[Source: {meta.get('filename', 'Unknown')}, Page: {meta.get('page', 'Unknown')}]\n"
        context += f"Content:\n{chunk.page_content}\n\n"
        context += "-" * 40 + "\n\n"

    # 4. Strict instruction to cite the retrieved metadata
    instruction = """
    
    IMPORTANT INSTRUCTION FOR REFERENCES:
    At the very end of your response, you MUST include a comprehensive "References" section citing the exact documents you used to formulate your answer.
    Format it as a bulleted list exactly like this:
    - Filename: [Insert Filename], Page Number: [Insert Page]
    """

    # 5. Query the LLM
    logging.info("Sending retrieved context to Azure OpenAI...")
    try:
        response = llm.ask_llm(question=question + instruction, context=context)
        
        print("\n" + "="*50)
        print("LLM RESPONSE:")
        print("="*50)
        print(response)
        
        with open(r"response.md", "w", encoding="utf-8") as f:
            f.write(response)
            
    except Exception as e:
        logging.error(f"Error querying the LLM: {e}")

if __name__ == "__main__":
    # Test your query here
    user_query = "Summarize the key financial trends across all the Q3 reports."
    query_presentations(user_query)