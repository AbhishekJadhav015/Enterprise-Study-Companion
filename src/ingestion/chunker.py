from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from src.utils.exception import CustomException
import sys

class DocumentChunker:
    "splits documents logically while maintaining context overlap"
    
    def __init__(self , chunk_size: int = 1000, chunk_overlap: int = 200) :
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # the heierarchy of the text splitter is as follows: paragraphs -> lines -> sentences -> words
        
    def split(self, documents: List[Document]) -> List[Document]:
        try:
            print(f"chunking {len(documents)} pages....")
            chunks = self.splitter.split_documents(documents)
            print(f"chunked {len(chunks)} contextual chunks")
            return chunks
        except Exception as e:
            raise CustomException(e ,sys)