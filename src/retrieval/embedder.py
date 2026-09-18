from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from src.utils.exception import CustomException
from src.utils.logger import logging
import sys 

class EmbeddingModel:
    "Loads and manages Qdrant's model"
    
    def __init__(self, model_name:str = "BAAI/bge-small-en-v1.5"):
        try:
            """FastEmbed uses ONNX runtim , requires no PyTorch dependencies"""
            self.model_name = model_name
            self.embeddings = FastEmbedEmbeddings(model_name=self.model_name)
        except Exception as e:
            raise CustomException(e ,sys)
            
        
    def get_embeddings(self)-> FastEmbedEmbeddings:
        "Returns the instantiated LangChain embeddings object"
        return self.embeddings 
        
        