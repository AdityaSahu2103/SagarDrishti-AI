#!/usr/bin/env python3
"""Debug script to identify LLM initialization issues."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=== LLM Debug Analysis ===")

# Step 1: Check Python environment
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Step 2: Check environment variables
api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API key from env: {'Found' if api_key else 'NOT FOUND'}")
if api_key:
    print(f"API key starts with: {api_key[:10]}...")

# Step 3: Test imports
print("\n--- Testing Imports ---")
try:
    import langchain_openai
    print(f"✅ langchain_openai version: {langchain_openai.__version__}")
except ImportError as e:
    print(f"❌ langchain_openai import failed: {e}")
    
try:
    import langchain_core
    print(f"✅ langchain_core version: {langchain_core.__version__}")
except ImportError as e:
    print(f"❌ langchain_core import failed: {e}")

try:
    import openai
    print(f"✅ openai version: {openai.__version__}")
except ImportError as e:
    print(f"❌ openai import failed: {e}")

# Step 4: Test specific imports
print("\n--- Testing Specific Classes ---")
try:
    from langchain_openai import ChatOpenAI
    print("✅ ChatOpenAI imported successfully")
except ImportError as e:
    print(f"❌ ChatOpenAI import failed: {e}")
    sys.exit(1)

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    print("✅ Message classes imported successfully")
except ImportError as e:
    print(f"❌ Message classes import failed: {e}")
    sys.exit(1)

# Step 5: Test LLM initialization
print("\n--- Testing LLM Initialization ---")
if not api_key:
    print("❌ Cannot test LLM - no API key")
    sys.exit(1)

try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=50,
        api_key=api_key
    )
    print("✅ ChatOpenAI object created successfully")
    
    # Test invoke
    test_message = [HumanMessage(content="Say 'working' if you can respond")]
    response = llm.invoke(test_message)
    print(f"✅ LLM response: {response.content}")
    
except Exception as e:
    print(f"❌ LLM initialization/test failed: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

print("\n--- Testing Config ---")
try:
    from config import Config
    print(f"✅ Config imported")
    print(f"Config API key present: {bool(Config.OPENAI_API_KEY)}")
    print(f"Config LLM model: {Config.LLM_MODEL}")
except Exception as e:
    print(f"❌ Config test failed: {e}")

print("\n=== Debug Complete ===")
