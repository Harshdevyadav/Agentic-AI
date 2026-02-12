import streamlit as st
from rag import process_document

st.set_page_config(page_title="RAG Question Answering System")
st.title("📄 Multi-Modal RAG System")

# Sidebar for file upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your PDF (e.g., Attention is All You Need)", type="pdf")

if uploaded_file:
    # Save the file locally to process it
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File Uploaded! Processing...")
    vector_store = process_document("temp.pdf")
    
    # Chat interface
    user_query = st.chat_input("Ask a question about the document...")
    if user_query:
        st.chat_message("user").write(user_query)
        
        # Retrieval
        results = vector_store.similarity_search(user_query, k=3)
        response = results[0].page_content if results else "No context found."
        
        st.chat_message("assistant").write(response)
