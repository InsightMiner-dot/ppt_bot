import os
from dotenv import load_dotenv

# ✅ Modern LangChain Imports
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# =========================
# ✅ CONFIGURATION
# =========================
CHROMA_PERSIST_DIR = "./chroma_db_storage"
COLLECTION_NAME = "ppt_production_kb"
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

# =========================
# ✅ 1. INITIALIZE RESOURCES
# =========================
def get_rag_chain():
    """Sets up the Vector DB, Retriever, and LLM Chain."""
    
    # 1. Re-initialize the exact same embedding model
    embedding_model = AzureOpenAIEmbeddings(azure_deployment=EMBEDDING_DEPLOYMENT)

    # 2. Connect to the existing Chroma Database
    print(f"🔌 Connecting to ChromaDB at {CHROMA_PERSIST_DIR}...")
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model
    )

    # 3. Create the Retriever (Fetch top 3 most relevant slides)
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 4. Initialize the LLM (Azure OpenAI Chat Model)
    llm = AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT,
        temperature=0.0, # 0.0 prevents hallucinations for factual RAG
    )

    # 5. Build the Production Prompt Template
    # We instruct the LLM to use the context and cite the slide metadata.
    system_prompt = (
        "You are an expert corporate assistant analyzing presentation slides.\n"
        "Use the following retrieved context to answer the user's question.\n"
        "If you don't know the answer based on the context, say 'I cannot find this in the presentations.'\n"
        "Always cite your sources at the end of your answer using the slide metadata provided in the context.\n"
        "\n"
        "Context:\n{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 6. Construct the LCEL Chain
    # This chain stuffs the retrieved documents into the {context} variable of the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    
    # This chain handles taking the user input, retrieving docs, and passing them to the QA chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


# =========================
# ✅ 2. EXECUTE QUERY
# =========================
def ask_question(query: str):
    chain = get_rag_chain()
    
    print(f"\n🤔 Question: {query}")
    print("⏳ Searching slides and generating answer...\n")
    
    # Invoke the chain
    response = chain.invoke({"input": query})
    
    # Print the LLM's Answer
    print("="*60)
    print("🎯 ANSWER:")
    print(response["answer"])
    print("="*60)
    
    # Print the exact slides it used to generate the answer (Traceability)
    print("\n📚 SOURCES USED:")
    for i, doc in enumerate(response["context"]):
        file_name = doc.metadata.get('file_name', 'Unknown')
        slide_num = doc.metadata.get('slide_number', '?')
        slide_title = doc.metadata.get('slide_title', 'Untitled')
        print(f"  [{i+1}] {file_name} - Slide {slide_num}: {slide_title}")


if __name__ == "__main__":
    # Test your pipeline! Change this to something relevant to your PPTX.
    user_query = "What were the key financial takeaways from Q3?"
    
    try:
        ask_question(user_query)
    except Exception as e:
        print(f"❌ Error during RAG execution: {e}")
