"""
Debug script to check langchain installations and imports
"""

def check_imports():
    print("🔍 Checking langchain installations...")
    
    # Check installed packages
    try:
        import pkg_resources
        installed_packages = [d.project_name for d in pkg_resources.working_set]
        langchain_packages = [pkg for pkg in installed_packages if 'langchain' in pkg.lower()]
        print(f"📦 Installed langchain packages: {langchain_packages}")
    except:
        print("❌ Could not check installed packages")
    
    # Test different import paths
    import_tests = [
        ("langchain_openai", "from langchain_openai import ChatOpenAI"),
        ("langchain.chat_models", "from langchain.chat_models import ChatOpenAI"),
        ("langchain_community.chat_models", "from langchain_community.chat_models import ChatOpenAI"),
        ("langchain.schema", "from langchain.schema import HumanMessage, SystemMessage"),
    ]
    
    successful_imports = []
    failed_imports = []
    
    for module_name, import_statement in import_tests:
        try:
            exec(import_statement)
            successful_imports.append(f"✅ {import_statement}")
        except ImportError as e:
            failed_imports.append(f"❌ {import_statement} - {str(e)}")
        except Exception as e:
            failed_imports.append(f"⚠️ {import_statement} - Unexpected error: {str(e)}")
    
    print("\n🟢 Successful imports:")
    for success in successful_imports:
        print(f"  {success}")
    
    print("\n🔴 Failed imports:")
    for failure in failed_imports:
        print(f"  {failure}")
    
    # Check if ChatOpenAI is available
    ChatOpenAI = None
    try:
        from langchain_openai import ChatOpenAI
        print("\n✅ ChatOpenAI available from langchain_openai")
    except ImportError:
        try:
            from langchain.chat_models import ChatOpenAI
            print("\n✅ ChatOpenAI available from langchain.chat_models")
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOpenAI
                print("\n✅ ChatOpenAI available from langchain_community.chat_models")
            except ImportError:
                print("\n❌ ChatOpenAI not available from any known location")
                return
    
    if ChatOpenAI:
        print(f"🎯 ChatOpenAI class found: {ChatOpenAI}")
        print(f"📍 Location: {ChatOpenAI.__module__}")

if __name__ == "__main__":
    check_imports()