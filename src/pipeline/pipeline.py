from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import DocumentChunker
from src.retrieval.vector_store import StudyVectorDB
from src.retrieval.embedder import EmbeddingModel
from utils.exception import CustomException
from utils.logger import logging
import sys

def run_ingestion_test(file_path: str):
    try:
        #step 1: Load documents
        logging.info("Loading Documents")
        parser = DocumentParser(file_path)
        raw_docs = parser.load_documents()
        
        #step 2: Chunk documents
        logging.info("chunking Documents")
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.split(raw_docs)
        
        #step 3 : Inspect the first chunk
        if chunks:
            print("\n--- SAMPLE CHUNK ---")
            print(f"Metadata: {chunks[0].metadata}")
            print(f"Content: {chunks[0].page_content[:300]}...\n")
    except Exception as e:
        return CustomException(e,sys)
        
def Build_knowledge_base(file_path: str):
    try:   
        print("Ingestion process started...")
        logging.info("Data Ingestion process started")
        parser = DocumentParser(file_path)
        chunks = DocumentChunker().split(parser.load_documents())
        
        print(" Vectorization")
        logging.info(" Vectorization started")
        embedder = EmbeddingModel().get_embedding()
        db = StudyVectorDB(embeddings=embedder)
        
        db.add_chunks(chunks)
        
        # Test a retrieval
        logging.info("Retrival started")
        retriever = db.get_retriever(top_k=2)
        query = "what is the main topic of this document ?"
        results = retriever.invoke(query)
        
        print(f"\n Top Match for '{query}':")
        print(results[0].page_content[:200] + "...")
    except Exception as e :
        return CustomException(e ,sys)
    
    
if __name__ == "__main__":
    Build_knowledge_base("data/raw/sample_textbook.pdf")  