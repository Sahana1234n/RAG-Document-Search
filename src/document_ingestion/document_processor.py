"""Document processing module for loading and splitting documents"""

from typing import List, Union
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyMuPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)


class DocumentProcessor:
    """Handles document loading and processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size : Size of text chunks
            chunk_overlap : Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize the text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_text(self, filepath: Union[str, Path]) -> List[Document]:
        """Load documents from a text file"""
        loader = TextLoader(str(filepath), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a document from a single PDF file"""
        loader = PyMuPDFLoader(str(file_path))
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, or text/PDF files.
        
        Args:
            sources: List of document sources
        
        Returns:
            List of loaded documents
        """
        docs: List[Document] = []

        for src in sources:
            # If the source is a URL
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
                continue

            path = Path(src)

            # If path is a directory → assume PDF directory
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))

            # If path is a text file
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_text(path))

            # If path is a single PDF file
            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))

            else:
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use URL, PDF file, PDF directory, or .txt file."
                )

        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        """
        return self.splitter.split_documents(documents)

    def process(self, sources: List[str]) -> List[Document]:
        """
        Complete pipeline: load and split documents
        
        Args:
            sources: List of URLs/paths
        
        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(sources)
        return self.split_documents(docs)
 