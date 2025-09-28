#!/usr/bin/env python3
"""Test script to verify LangChain imports and OpenAI initialization."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing LangChain imports...")

try:
    from langchain_openai import ChatOpenAI
    print("✅ ChatOpenAI imported successfully")
    
    from langchain_core.messages import HumanMessage, SystemMessage
    print("✅ Message classes imported successfully")
    
    # Test OpenAI initialization
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OpenAI API key found: {api_key[:10]}...")
        
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.2,
                max_tokens=100,
                api_key=api_key
            )
            print("✅ ChatOpenAI initialized successfully")
            
            # Test a simple query
            test_message = [HumanMessage(content="Hello, can you respond with 'LLM is working'?")]
            response = llm.invoke(test_message)
            print(f"✅ LLM test response: {response.content}")
            
        except Exception as e:
            print(f"❌ LLM initialization failed: {e}")
    else:
        print("❌ OpenAI API key not found in environment")
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Installing required packages...")
