import os
import glob
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """
    A class for loading and processing PDF documents for semantic search.
    """
    
    def __init__(self, documents_dir="Policy+Documents", chunk_size=1000, chunk_overlap=200):
        """
        Initialize the DocumentProcessor.
        
        Args:
            documents_dir (str): Directory containing the PDF documents
            chunk_size (int): Size of text chunks for splitting documents
            chunk_overlap (int): Overlap between text chunks
        """
        self.documents_dir = documents_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def load_documents(self):
        """
        Load PDF documents from the specified directory.
        
        Returns:
            list: List of processed document chunks with metadata
        """
        # Check if documents directory exists
        if not os.path.exists(self.documents_dir):
            raise FileNotFoundError(f"Documents directory not found: {self.documents_dir}")
        
        # Create PDF directory loader
        loader = PyPDFDirectoryLoader(self.documents_dir)
        
        # Load documents
        documents = loader.load()
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        # Split documents into chunks
        split_documents = text_splitter.split_documents(documents)
        
        print(f"Loaded {len(documents)} PDF documents and split into {len(split_documents)} chunks")
        
        return split_documents
    
    def load_single_document(self, pdf_path):
        """
        Load a single PDF document.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of processed document chunks with metadata
        """
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Create PDF loader
        loader = PyPDFLoader(pdf_path)
        
        # Load document
        document = loader.load()
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        # Split document into chunks
        split_document = text_splitter.split_documents(document)
        
        print(f"Loaded PDF document and split into {len(split_document)} chunks")
        
        return split_document
