"""
Test script to diagnose OpenAIEmbeddings initialization issues
"""
from langchain_openai import OpenAIEmbeddings
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print(f"Python version: {sys.version}")
print(f"Testing OpenAIEmbeddings from langchain_openai")

# Print package versions
print(f"langchain_openai version: {__import__('pkg_resources').get_distribution('langchain-openai').version}")
print(f"openai version: {__import__('pkg_resources').get_distribution('openai').version}")
print(f"pydantic version: {__import__('pkg_resources').get_distribution('pydantic').version}")

try:
    # First try with all the basic parameters
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    print("✅ Successfully initialized OpenAIEmbeddings with model and api_key parameters")
    
    # Test embedding
    test_text = "This is a test document."
    embedding = embeddings.embed_query(test_text)
    print(f"✅ Successfully generated embedding. Dimension: {len(embedding)}")
    
except Exception as e:
    print(f"❌ Error initializing OpenAIEmbeddings: {e}")

# Try with the most basic initialization
try:
    embeddings_basic = OpenAIEmbeddings()
    print("✅ Successfully initialized OpenAIEmbeddings with no parameters")
except Exception as e:
    print(f"❌ Error initializing basic OpenAIEmbeddings: {e}")
