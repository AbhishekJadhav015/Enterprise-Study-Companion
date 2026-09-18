from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
from src.utils.exception import CustomException
import sys

def format_docs(docs):
    try:
        """ Formats retrived chunks with page metadata for citation."""
        formatted = []
        for doc in docs :
            page = doc.metadata.get("page","Unknown")
            formatted.append(f"[Source Page{page}]:\n {doc.page_content}")
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
            raise CustomException(e,sys)
    
class RAGChain:
    """ Constructs the RAG pipeline using GROQ for fast inference"""
    def __init__(self, retriever, model_name: str ="gemini-3.6-flash", temperature : float = 0.1  ):
        try:
            self.retriever = retriever 
            
            load_dotenv()
            self.api_key =  os.getenv("GOOGLE_API_KEY")
            
            self.llm =  ChatOpenAI(model = model_name , temperature=temperature ,
                                api_key=self.api_key,
                                base_url="https://generativelanguage.googleapis.com/v1beta/openai/" )
            self.output_parser = StrOutputParser()
        except Exception as e:
            raise CustomException(e ,sys)
        
    def build(self, Prompt_template):
        try:
            chain = (
                {
                    "context": self.retriever | format_docs ,
                    "question": RunnablePassthrough()
                }
                | Prompt_template
                | self.llm
                | self.output_parser
            )
            return chain
        except Exception as e:
            raise CustomException(e,sys)