# Explicitly export classes to make importing more reliable
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.query_engine import QueryEngine

__all__ = ['DocumentProcessor', 'VectorStoreManager', 'QueryEngine']
