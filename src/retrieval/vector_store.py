from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance , VectorParams
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from typing import List
from src.utils.logger import logging
from src.utils.exception import CustomException
import os 
import sys

class StudyVectorDB :
    def __init__(self, embeddings , collection_name = "study_materials", persist_dir =".data/qdrant_db"):
        try:   
            self.embeddings = embeddings
            self.collection_name = collection_name
            #ensure the persist_dir exists
            os.makedirs(persist_dir, exist_ok=True)
            
            self.client = QdrantClient(path=persist_dir)
            
            if not self.client.collection_exists(self.collection_name):
                sample_embedding = self.embeddings.embed_query("test")
                dimensions = len(sample_embedding)
                
                self.client.create_collection(
                    collection_name= self.collection_name,
                    vectors_config = VectorParams(size = dimensions ,distance=Distance.COSINE),
                )
                print(f"created new Qdrant collection :{self.collection_name}")
            
            
            self.vector_store = QdrantVectorStore(client=self.client,
                                                collection_name =self.collection_name,
                                                embedding=self.embeddings)
        except Exception as e:
            raise CustomException(e, sys)
        
    def add_chunks(self , chunks: List[Document]):
            try:   
                print(f"Indexing {len(chunks)} chunks into Qdrant ...")
                self.vector_store.add_documents(chunks)
                print("Indexing Completed")
                
            except Exception as e:
                raise CustomException(e ,sys)
            
    def get_retriever(self ,top_k : int = 5):
            #converts the vector store into langchain retriever interface
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})
        