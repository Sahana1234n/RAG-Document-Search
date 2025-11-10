"""vector store for document embedding and retrieval"""

from typing import List 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStore:
    """manges vector store application"""
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self , documents:List[Document]):
        """
        Create vector store from documents
        
        Args:
            documents: List of documents to embed

        """
        self.vectorstore = FAISS.from_documents(documents , self.embeddings)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        """
        Get the retriever instance
        
        Returns:
            Retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self , query:str , k:int = 4)->List[Document]:
        """Retrieve document for query

        Args:
            query: Search query ,
            k: number of documents to retrieve

        Returns:
            List of relevant documents       
        """
        if self.retriever is None:
            raise ValueError("vectorStore is not initialized. Call create_vectorstore first")
        return self.retriever._get_relevant_documents(query)
    