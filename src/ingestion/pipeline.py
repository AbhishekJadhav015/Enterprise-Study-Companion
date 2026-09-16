from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import DocumentChunker

def run_ingestion_test(file_path: str):
    #step 1: Load documents
    parser = DocumentParser(file_path)
    raw_docs = parser.load_documents()
    
    #step 2: Chunk documents
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split(raw_docs)
    
    #step 3 : Inspect the first chunk
    if chunks:
        print("\n--- SAMPLE CHUNK ---")
        print(f"Metadata: {chunks[0].metadata}")
        print(f"Content: {chunks[0].page_content[:300]}...\n")
        
if __name__ == "__main__":
    run_ingestion_test("data/raw/sample_textbook.pdf")  # Replace with your actual PDF file path