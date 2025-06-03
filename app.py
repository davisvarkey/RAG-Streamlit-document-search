import streamlit as st
import os
from dotenv import load_dotenv
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.query_engine import QueryEngine

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Semantic Spotter - Insurance Policy Search",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("Semantic Spotter 🔍")
st.subheader("Insurance Policy Semantic Search Engine")
st.markdown("""
This application allows you to search through insurance policy documents using semantic search technology.
Simply enter your question about insurance policies, and the app will find the most relevant information.
""")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.documents_loaded = False
    st.session_state.vector_store_created = False

# Sidebar for configurations and information
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", 
                            value=os.getenv("OPENAI_API_KEY", ""), 
                            type="password",
                            help="Enter your OpenAI API key to enable the semantic search functionality")
    print(api_key)
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Document loading section
    st.subheader("Document Processing")
    if st.button("Load Documents"):
        with st.spinner("Loading and processing documents..."):
            try:
                doc_processor = DocumentProcessor()
                documents = doc_processor.load_documents()
                st.session_state.documents = documents
                st.session_state.documents_loaded = True
                st.success(f"Successfully loaded {len(documents)} document chunks")
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")
    
    # Vector store creation
    if st.session_state.documents_loaded:
        if st.button("Create Vector Store"):
            with st.spinner("Creating vector store..."):
                try:
                    # Pass the API key directly to VectorStoreManager
                    vector_store_manager = VectorStoreManager(api_key=api_key)
                    print('1')
                    vector_store = vector_store_manager.create_vector_store(st.session_state.documents)
                    print('2')
                    st.session_state.vector_store = vector_store
                    print('3')
                    st.session_state.vector_store_created = True
                    st.success("Vector store created successfully")
                except Exception as e:
                    st.error(f"Error creating vector store: {str(e)}")
    
    # App info
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses LangChain and OpenAI to provide semantic search capabilities 
    over insurance policy documents. It extracts text, generates embeddings, 
    and uses retrieval-augmented generation to answer your questions.
    """)

# Main app area
# Debug session state
st.sidebar.markdown("---")
with st.sidebar.expander("Debug Info"):
    st.write("Session State Variables:")
    st.write(f"- Documents Loaded: {st.session_state.get('documents_loaded', False)}")
    st.write(f"- Vector Store Created: {st.session_state.get('vector_store_created', False)}")

# Always show the query input
st.markdown("## Ask a Question")
user_query = st.text_input("Enter your insurance policy question:", 
                          placeholder="e.g., What is the coverage for disability?")

# Search functionality
if user_query:
    search_button = st.button("Search")
    
    # Show appropriate guidance based on session state
    if not st.session_state.get('documents_loaded', False):
        st.warning("⚠️ Please load documents first using the sidebar before searching.")
    elif not st.session_state.get('vector_store_created', False):
        st.warning("⚠️ Please create the vector store using the sidebar before searching.")
    
    # Process search if documents and vector store are ready
    if search_button and st.session_state.get('vector_store_created', False):
        with st.spinner("Searching for relevant information..."):
            try:
                # Initialize query engine
                query_engine = QueryEngine(st.session_state.vector_store)
                
                # Get answer using RAG
                answer = query_engine.answer_query(user_query)
                
                # Display answer
                st.markdown("### Answer")
                st.write(answer)
                
                # Get relevant documents
                relevant_docs = query_engine.get_relevant_documents(user_query)
                
                # Display source documents
                st.markdown("### Source Documents")
                for i, doc in enumerate(relevant_docs[:3]):  # Show top 3 documents
                    with st.expander(f"Document {i+1} - {doc.metadata['source']} (Page {doc.metadata['page']})"): 
                        st.write(doc.page_content)
                        
            except Exception as e:
                st.error(f"Error processing your query: {str(e)}")
    
    # If button is clicked but conditions aren't met, show helpful message
    elif search_button:
        st.error("Cannot process query until documents are loaded and vector store is created.")
else:
    # Display a helpful message when no query is entered
    st.info("Enter your question about insurance policies above and click 'Search' to get an answer.")
    
# Display instructions if first-time setup is needed
if not st.session_state.get('documents_loaded', False) or not st.session_state.get('vector_store_created', False):
    st.markdown("---")
    st.markdown("### Getting Started")
    st.markdown("""
    1. Enter your OpenAI API key in the sidebar
    2. Click "Load Documents" to process the insurance PDFs
    3. Click "Create Vector Store" to build the search index
    4. Ask your question in the search box above
    """)

