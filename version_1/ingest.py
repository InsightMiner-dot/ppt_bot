import os
import glob
import base64
import logging
from pptx import Presentation
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from scripts import llm  # Your llm.py file containing the 'embeddings' variable

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_vector_database():
    target_folder = "presentations"
    os.makedirs(target_folder, exist_ok=True)
    ppt_files = glob.glob(os.path.join(target_folder, "*.pptx"))

    if not ppt_files:
        logging.warning("No .pptx files found. Please add some and run again.")
        return

    raw_documents = []

    # --- 1. Multimodal Extraction ---
    for file_path in ppt_files:
        actual_filename = os.path.basename(file_path)
        logging.info(f"Extracting: {actual_filename}")
        
        try:
            prs = Presentation(file_path)
            for slide_index, slide in enumerate(prs.slides, start=1):
                slide_content_elements = []
                
                for shape in slide.shapes:
                    # Text
                    if shape.has_text_frame and shape.text.strip():
                        slide_content_elements.append(shape.text.strip())
                    
                    # Tables
                    elif shape.has_table:
                        table_rows = []
                        for row in shape.table.rows:
                            row_data = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                            table_rows.append(" | ".join(row_data))
                        if table_rows:
                            slide_content_elements.append("### Table Data ###\n" + "\n".join(table_rows))
                    
                    # Images (Vision)
                    elif hasattr(shape, "image"):
                        logging.info(f"   -> Processing image on Slide {slide_index}...")
                        base64_img = base64.b64encode(shape.image.blob).decode('utf-8')
                        image_metadata = {"filename": actual_filename, "page_number": slide_index}
                        try:
                            desc = llm.describe_image(base64_image=base64_img, metadata=image_metadata)
                            slide_content_elements.append(f"### Image/Chart Description ###\n{desc}")
                        except Exception as e:
                            logging.error(f"Vision failed on slide {slide_index}: {e}")

                # If slide has content, create a LangChain Document
                if slide_content_elements:
                    slide_text = "\n\n".join(slide_content_elements)
                    
                    # Store exact metadata in the LangChain dictionary format
                    metadata = {
                        "source": file_path,
                        "filename": actual_filename,
                        "page": str(slide_index)
                    }
                    
                    doc = Document(page_content=slide_text, metadata=metadata)
                    raw_documents.append(doc)
                    
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")

    logging.info(f"Extracted {len(raw_documents)} total slides. Starting chunking...")

    # --- 2. Chunking ---
    # We use a text splitter just in case a single slide has massive tables or text blocks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=150
    )
    chunked_docs = text_splitter.split_documents(raw_documents)
    logging.info(f"Created {len(chunked_docs)} chunks.")

    # --- 3. Save to Chroma DB ---
    db_directory = "./chroma_db"
    logging.info("Saving to Chroma DB...")
    
    Chroma.from_documents(
        documents=chunked_docs,
        embedding=llm.embeddings,
        persist_directory=db_directory
    )
    
    logging.info("Database built successfully!")

if __name__ == "__main__":
    build_vector_database()
