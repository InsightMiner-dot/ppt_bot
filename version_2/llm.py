import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv(override=True)

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_KEY")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    openai_api_version=azure_api_version,
    azure_deployment=azure_deployment,
    openai_api_key=azure_api_key,
    temperature=0,
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint,
    openai_api_version=azure_api_version,
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_key=azure_api_key,
)

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who answers from the retrieved PowerPoint content and cites the metadata provided. Use the strongest matching retrieved evidence. If the answer is partially available, provide the partial answer instead of defaulting to 'I don't know'.",
        ),
        (
            "user",
            """Answer the question based on the following retrieved documents or context.
If the answer is present or partially present in the retrieved content, answer using that evidence.
Only say you don't know when the retrieved content truly does not contain the answer.

### Context:
{context}

### Question:
{question}

### Answer:""",
        ),
    ]
)

qna_chain = template | llm | StrOutputParser()


def ask_llm(question: str, context: str) -> str:
    return qna_chain.invoke({"question": question, "context": context})
