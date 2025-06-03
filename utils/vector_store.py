from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

class VectorStoreManager:
    """
    A class for managing vector stores to enable semantic search functionality.
    """
    
    def __init__(self, vector_store_path="./faiss_index", api_key=None):
        """
        Initialize the VectorStoreManager.
        
        Args:
            vector_store_path (str): Path where the vector store will be saved
            api_key (str, optional): OpenAI API key. If None, will try to get from environment.
        """
        self.vector_store_path = vector_store_path
        
        # Initialize OpenAIEmbeddings with the API key
        if api_key:
            # If API key is provided directly
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=api_key
            )
        else:
            # Otherwise rely on environment variables
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002"
            )
        
    def create_vector_store(self, documents):
        """
        Create a vector store from document chunks.
        
        Args:
            documents (list): List of document chunks
            
        Returns:
            VectorStore: The created vector store
        """
        # Create FAISS vector store
        vector_store = FAISS.from_documents(
            documents, 
            self.embeddings
        )
        print('passed documents and embeddings')
        # Save vector store to disk
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        print(f"Saving vector store to {self.vector_store_path}")
        vector_store.save_local(self.vector_store_path)
        
        print(f"Vector store created and saved to {self.vector_store_path}")
        
        return vector_store
    
    def load_vector_store(self):
        """
        Load an existing vector store from disk.
        
        Returns:
            VectorStore: The loaded vector store
        """
        # Check if vector store exists
        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError(f"Vector store not found at {self.vector_store_path}")
        
        # Load vector store
        vector_store = FAISS.load_local(
            self.vector_store_path, 
            self.embeddings
        )
        
        print(f"Vector store loaded from {self.vector_store_path}")
        
        return vector_store
