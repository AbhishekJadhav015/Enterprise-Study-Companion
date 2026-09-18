# Use the exact same imports you have at the top of your main.py
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.vector_store import StudyVectorDB
def test_database():
    print("1. Loading Embeddings...")
    embedder = EmbeddingModel().get_embeddings()
    
    print("2. Connecting to Qdrant Database...")
    db = StudyVectorDB(embeddings=embedder)
    retriever = db.get_retriever(top_k=3)
    
    query = "What are the exact steps for using the Pomodoro Technique?"
    print(f"\nSearching for: '{query}'\n")
    
    # We query the database directly, bypassing the LLM entirely
    docs = retriever.invoke(query)
    
    print(f"✅ Found {len(docs)} chunks!\n")
    
    for i, doc in enumerate(docs):
        print(f"--- MATCH {i+1} (Page {doc.metadata.get('page', 'Unknown')}) ---")
        print(doc.page_content)
        print("-" * 40 + "\n")

if __name__ == "__main__":
    test_database()