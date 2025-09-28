#!/usr/bin/env python3
"""Test RAG engine initialization with the new fallback system."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=== Testing RAG Engine LLM Initialization ===")

try:
    from config import Config
    from embedding_index import ProfileEmbeddingIndex
    from rag_engine import OceanographyRAG
    
    print("✅ All modules imported successfully")
    
    # Create a dummy embedding index for testing
    class DummyEmbeddingIndex:
        def search_similar_profiles(self, query, top_k=3):
            return []
    
    # Initialize RAG engine
    dummy_index = DummyEmbeddingIndex()
    rag = OceanographyRAG(dummy_index, Config)
    
    print(f"OpenAI available: {rag.openai_available}")
    print(f"API key present: {bool(Config.OPENAI_API_KEY)}")
    
    # Try to initialize LLM
    print("\nAttempting LLM initialization...")
    rag.initialize_llm()
    
    if rag.llm:
        print("✅ LLM initialized successfully!")
        
        # Test a simple query
        result = rag.query("Hello, are you working?")
        print(f"Test query result: {result['answer'][:100]}...")
    else:
        print("⚠️ LLM not initialized - using rule-based mode")
        
        # Test rule-based response
        result = rag.query("Hello")
        print(f"Rule-based response: {result['answer'][:100]}...")
        
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
