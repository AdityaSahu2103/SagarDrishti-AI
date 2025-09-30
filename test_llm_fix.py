#!/usr/bin/env python3
"""Test script to verify LLM initialization fix."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=== LLM Initialization Test ===")

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    print("✅ LangChain packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Installing required packages...")
    os.system("pip install langchain-openai langchain-core openai")
    sys.exit(1)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OpenAI API key not found in environment")
    sys.exit(1)
print(f"✅ OpenAI API key found: {api_key[:10]}...")

try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=50,
        api_key=api_key
    )
    print("✅ ChatOpenAI initialized successfully")
except Exception as e:
    print(f"❌ LLM initialization failed: {e}")
    sys.exit(1)

try:
    test_message = [HumanMessage(content="Say 'LLM working' if you can respond")]
    response = llm.invoke(test_message)
    print(f"✅ LLM test successful: {response.content}")
except Exception as e:
    print(f"❌ LLM test failed: {e}")
    sys.exit(1)

print("🎉 All LLM tests passed! The LLM should now work properly.")
