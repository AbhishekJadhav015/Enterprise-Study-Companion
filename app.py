import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from src.utils.logger import logging
from src.utils.exception import CustomException
import mlflow
import sys

# Load environment variables
load_dotenv()

# Backend components
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.vector_store import StudyVectorDB
from src.generation.prompt import get_study_prompt
from src.generation.chain import RAGChain
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import DocumentChunker

# Enable MLflow tracing
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Enterprise_Study_Companion")
mlflow.langchain.autolog()

st.set_page_config(page_title="Enterprise Study Companion", page_icon="📚", layout="centered")
st.title("📚 Enterprise Study Companion")
st.markdown("Upload documents and query your knowledge base. Powered by **Qdrant**, **Gemini**, and **MLflow**.")

@st.cache_resource
def get_database():
    try:
        """Creates a single, persistent Qdrant connection."""
        embedder = EmbeddingModel().get_embeddings()
        return StudyVectorDB(embeddings=embedder)
    except Exception as e:
        raise CustomException(e ,sys)

@st.cache_resource
def get_chain(_db):
    try:
        """Builds the LangChain pipeline using the existing DB connection."""
        retriever = _db.get_retriever(top_k=3)
        prompt = get_study_prompt()
        return RAGChain(retriever=retriever).build(prompt)
    except Exception as e:
        raise CustomException(e, sys)

# Initialize once per session
db = get_database()
rag_chain = get_chain(db)

# ==========================================
# SIDEBAR: DYNAMIC DOCUMENT UPLOAD 
# ==========================================
with st.sidebar:
    st.header("📄 Add Study Materials")
    uploaded_file = st.file_uploader("Upload a PDF textbook or guide", type=["pdf"])
    
    if st.button("Process Document", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Chunking and vectorizing document..."):
                try:
                    logging.info(f"Started processing uploaded file: {uploaded_file.name}")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_pdf_path = tmp_file.name
                        logging.info(f"Temporary file created at: {temp_pdf_path}")

                    try:
                        # Parse and Chunk
                        parser = DocumentParser(temp_pdf_path)
                        docs = parser.load_documents()
                        chunks = DocumentChunker().split(docs)
                        logging.info(f"Successfully chunked {uploaded_file.name} into {len(chunks)} pieces.")
                        db.add_chunks(chunks)
                        logging.info("Chunks successfully committed to Qdrant vector store.")
                        
                    finally:
                        # Clean up the temp file
                        if os.path.exists(temp_pdf_path):
                            os.remove(temp_pdf_path)
                        logging.info(f"Cleaned up temporary file: {temp_pdf_path}")
                        
                    st.success(f"✅ Successfully indexed {len(chunks)} chunks from {uploaded_file.name}!")
                    
                except Exception as e:
                    custom_error = CustomException(e, sys)
                    logging.error(f"Ingestion Pipeline Failed: {custom_error.error_message}")
        else:
            st.warning("Please select a PDF file first.")

# ==========================================
# MAIN UI: KNOWLEDGE RETRIEVAL
# ==========================================
st.divider()
query = st.text_input("What would you like to know about your study materials?", placeholder="e.g., What are the exact steps for using the Pomodoro Technique?")

if st.button("Generate Answer", type="primary"):
    if query:
        with st.spinner("Searching local knowledge base and synthesizing answer..."):
            try:
                logging.info(f"User initiated search query: '{query}'")
                response = rag_chain.invoke(query)
                logging.info("Successfully generated RAG response.")
                
                st.markdown("### Answer")
                st.info(response)
                
            except Exception as e:
                custom_error = CustomException(e, sys)
                logging.error(f"Retrieval/Generation Pipeline Failed: {custom_error.error_message}")
                st.error("An error occurred while generating the answer. Our system has logged the issue.")
            
    else:
        st.warning("Please enter a question to search.")