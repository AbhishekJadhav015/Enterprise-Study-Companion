import os
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.vector_store import StudyVectorDB
from src.generation.prompt import get_study_prompt
from src.generation.chain import RAGChain
from dotenv import load_dotenv
from src.utils.exception import CustomException
from src.utils.logger import logging
import mlflow
import sys

load_dotenv()   

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Enterprise_Study_Companion")
mlflow.langchain.autolog()

def main():
    try:
        print("Initializaing Knowledge Base and Models...")
        logging.info("Initializing knowledge Base and Embedding Models")
        embedder = EmbeddingModel().get_embeddings()
        db = StudyVectorDB(embeddings= embedder)
        retriever = db.get_retriever(top_k = 3)
        
        logging.info("Connecting Retriever and prompt with LCEL chain")
        prompt = get_study_prompt()
        rag_chain = RAGChain(retriever = retriever).build(prompt)
        
        logging.info("Aksing query and generating response")
        query = "What are the exact steps for using the Pomodoro Technique?"
        print(f"\nStudnet Query: '{query}'\n")
        
        response = rag_chain.invoke(query)
        print("__ Study Companion Response __")
        print(response)
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
        main()