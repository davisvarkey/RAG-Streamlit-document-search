"""QueryEngine module for semantic search on insurance documents using LangChain."""
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

class QueryEngine:
    """
    A class for handling semantic search queries and providing answers using RAG.
    """
    
    def __init__(self, vector_store, top_k=50):
        """
        Initialize the QueryEngine.
        
        Args:
            vector_store: The vector store containing document embeddings
            top_k (int): Number of documents to retrieve for each query
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.llm = ChatOpenAI(temperature=0)
        self.prompt = hub.pull("rlm/rag-prompt")
        self.retriever = self._create_retriever()
        self.rag_chain = self._create_rag_chain()
        
    def _create_retriever(self):
        """
        Create a retriever for the vector store.
        
        Returns:
            retriever: The configured retriever
        """
        # Initialize base retriever from vector store
        search_kwargs = {"k": self.top_k, "score_threshold": 0.8}
        return self.vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs=search_kwargs
        )
    
    def _format_docs(self, docs):
        """
        Format a list of documents into a single string.
        
        Args:
            docs (list): List of document objects
            
        Returns:
            str: Formatted string of document contents
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _create_rag_chain(self):
        """
        Create a RAG chain for query processing.
        
        Returns:
            chain: The configured RAG chain
        """
        return (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def answer_query(self, query):
        """
        Answer a user query using RAG.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: The answer to the query
        """
        return self.rag_chain.invoke(query)
    
    def get_relevant_documents(self, query):
        """
        Get documents relevant to a query.
        
        Args:
            query (str): The user's query
            
        Returns:
            list: List of relevant documents
        """
        return self.retriever.invoke(query)
