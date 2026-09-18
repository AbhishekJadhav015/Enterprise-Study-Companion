from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
from src.utils.exception import CustomException
import sys

class DocumentParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load_documents(self) -> List[Document]:
        try:
            "handles the extraction of raw text data and metadata from files"
            print(f"Loading documents :{self.file_path}...")
            loader = PyPDFLoader(self.file_path)
            return loader.load()
        except Exception as e :
            raise CustomException(e,sys)
    