import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_KEY")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") # e.g., gpt-4o-mini
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    openai_api_version=azure_api_version,
    azure_deployment=azure_deployment,
    openai_api_key=azure_api_key,
    temperature=0
)

# ... (Keep all your existing llm.py code above this) ...
from langchain_openai import AzureOpenAIEmbeddings

# NEW: Initialize the Embedding Model for Chroma DB
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint,
    openai_api_version=azure_api_version,
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_key=azure_api_key,
)

# --- Text Summary Chain ---
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who answers user's questions based on the retrieved documents or context. Always cite the metadata provided."),
    ("user", """Answer the question based on the following retrieved documents or context. If you don't know the answer, say you don't know.

### Context:
{context}

### Question:
{question}

### Answer:""")
])

qna_chain = template | llm | StrOutputParser()

def ask_llm(question: str, context: str) -> str:
    return qna_chain.invoke({"question": question, "context": context})

# --- Vision Chain ---
def describe_image(base64_image: str, metadata: dict) -> str:
    vision_prompt = (
        f"You are looking at an extracted image/chart from a PowerPoint presentation.\n"
        f"Context - Filename: {metadata.get('filename', 'Unknown')}, Slide Number: {metadata.get('page_number', 'Unknown')}.\n"
        f"Describe this image, chart, or table in detail. Extract any numbers, trends, or text present "
        f"so this description can be used as context to answer user questions later."
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )
    
    response = llm.invoke([message])
    return response.content
