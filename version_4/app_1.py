import streamlit as st
import os
import glob
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (SystemMessagePromptTemplate, 
                                    HumanMessagePromptTemplate,
                                    ChatPromptTemplate)

# --- 1. Page Configuration ---
st.set_page_config(page_title="PPT Analysis Assistant", page_icon="📊", layout="wide")
st.title("📊 Multi-PPT Insight Extractor")

# --- 2. Configuration & LLM Setup ---
load_dotenv(override=True)

def get_llm():
    """Initializes the Azure OpenAI client."""
    return AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model=os.getenv("MODEL_NAME"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
        openai_api_type="azure",
        temperature=0.1
    )

# --- 3. Prompt Logic ---
system_template = SystemMessagePromptTemplate.from_template(
    "You are a helpful AI assistant who answers user questions based on the provided context."
)

human_template_string = """Answer user question based on the provided context ONLY! 
If you do not know the answer, just say "I don't know".
If there are multiple ppts or files, compare them and provide the answer from both.

### Context:
{context}

### Question:
{question}

IMPORTANT INSTRUCTION: At the very end of your answer, you must include a "Reference" section.
Format: - Filename: [Name], Page Number: [Number]

### Answer:"""

human_template = HumanMessagePromptTemplate.from_template(human_template_string)
chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])

# --- 4. Sidebar: File Upload ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PPTX files", type="pptx", accept_multiple_files=True)
    process_button = st.button("Process Presentations")

# --- 5. Main Logic ---
if "context_data" not in st.session_state:
    st.session_state.context_data = ""

if process_button and uploaded_files:
    with st.spinner("Extracting content from PowerPoints..."):
        all_ppt_content = ""
        
        # We use a temporary directory to save uploaded files so the Loader can read them
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    loader = UnstructuredPowerPointLoader(file_path, mode="elements")
                    docs = loader.load()
                    
                    for doc in docs:
                        page = doc.metadata.get("page_number", "Unknown")
                        # Build a chunk for the LLM
                        chunk = f"## Source: {uploaded_file.name}, Page: {page}\n{doc.page_content}\n\n"
                        all_ppt_content += chunk
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
            
        st.session_state.context_data = all_ppt_content
        st.success(f"Processed {len(uploaded_files)} files successfully!")

# --- 6. Chat Interface ---
if st.session_state.context_data:
    st.divider()
    user_question = st.text_input("Ask a question about your presentations:", 
                                  placeholder="e.g., What are the comments in G&A Evolution?")

    if user_question:
        with st.spinner("Thinking..."):
            try:
                llm = get_llm()
                qna_chain = chat_prompt | llm | StrOutputParser()
                
                response = qna_chain.invoke({
                    'context': st.session_state.context_data, 
                    'question': user_question
                })
                
                st.markdown("### Response")
                st.write(response)
                
                # Option to download the response
                st.download_button("Download Response as MD", response, file_name="ai_response.md")
                
            except Exception as e:
                st.error(f"LLM Error: {e}")
else:
    st.info("Please upload and process PPTX files from the sidebar to begin.")
