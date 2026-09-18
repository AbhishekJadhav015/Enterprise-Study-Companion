from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import DocumentChunker
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.vector_store import StudyVectorDB

def force_ingest():
    print("1. Parsing and Chunking...")
    parser = DocumentParser("data/raw/sample_textbook.pdf")
    chunks = DocumentChunker().split(parser.load_documents())
    
    print("2. Loading Embedding Model...")
    embedder = EmbeddingModel().get_embeddings()

    print("3. Connecting to Qdrant and saving chunks...")
    db = StudyVectorDB(embeddings=embedder)
    db.add_chunks(chunks)
    
    # CRITICAL FIX FOR WINDOWS: 
    # Explicitly close the database connection so it flushes to disk.
    db.client.close()
    
    print(f"✅ Successfully saved {len(chunks)} chunks to local disk!")

if __name__ == "__main__":
    force_ingest()