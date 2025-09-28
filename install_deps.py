#!/usr/bin/env python3
"""Install required dependencies for LLM functionality."""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    packages = [
        "langchain==0.1.20",
        "langchain-openai==0.1.8", 
        "langchain-core==0.1.52",
        "langchain-community==0.0.38",
        "openai>=1.0.0"
    ]
    
    print("Installing LangChain packages...")
    
    for package in packages:
        install_package(package)
    
    print("\nTesting imports...")
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        print("✅ All imports successful!")
    except ImportError as e:
        print(f"❌ Import still failing: {e}")

if __name__ == "__main__":
    main()
