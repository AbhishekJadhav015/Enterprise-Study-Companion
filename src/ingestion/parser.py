from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List

class DocumentParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load_documents(self) -> List[Document]:
        "handles the extraction of raw text data and metadata from files"
        print(f"Loading documents :{self.file_path}...")
        loader = PyPDFLoader(self.file_path)
        return loader.load()
    