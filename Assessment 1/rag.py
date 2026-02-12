import os
from unstructured.partition.pdf import partition_pdf
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_document(file_path):
    # 1. Partition the PDF
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=1000,
    )
    
    # 2. Extract text for the vector store
    text_content = [el.text for el in elements if el.category in ["NarrativeText", "Title"]]
    
    # 3. Create Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(
        texts=text_content,
        embedding=embeddings,
        persist_directory="./apple_db"
    )
    return vector_store
