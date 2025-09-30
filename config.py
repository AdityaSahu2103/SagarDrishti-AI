"""
Configuration module for ARGO Float Chat system.
Centralized settings for data paths, API tokens, and model configurations.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    DATA_ROOT = os.getenv("DATA_ROOT", "argo_data")
    INDIAN_OCEAN_PATH = os.path.join(DATA_ROOT, "indian_ocean")
    
    ARGO_BASE_URL = "https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian"
    DEFAULT_YEAR = "2019"
    DEFAULT_MONTH = "01"
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
    LLM_MODEL = "gpt-4o-mini"  
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
    
    DEFAULT_TOP_K = 3
    FAISS_INDEX_DIMENSION = 384  
    
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_TITLE = "FloatChat - ARGO Oceanographic Data Assistant"
    API_VERSION = "1.0.0"
    
    DEFAULT_MAPBOX_STYLE = "carto-positron"
    DEFAULT_COLORSCALE = "Viridis"
    
    MCP_ENABLED = os.getenv("MCP_ENABLED", "true").lower() in ("1", "true", "yes")
    MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
    MCP_PORT = int(os.getenv("MCP_PORT", "8765"))
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"ws://{MCP_HOST}:{MCP_PORT}")


    POSTGRES_ENABLED = os.getenv("POSTGRES_ENABLED", "false").lower() in ("1", "true", "yes")
    POSTGRES_URL = os.getenv("POSTGRES_URL")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "argo")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_SSLMODE = os.getenv("POSTGRES_SSLMODE", "prefer")

    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss").lower()
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(DATA_ROOT, "chroma"))
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "argo_profiles")
    
    AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() in ("1", "true", "yes")
    AUTH_USERS = [u.strip() for u in os.getenv("AUTH_USERS", "admin:admin").split(",") if u.strip()]

    DB_DEFAULT_START_DATE = os.getenv("DB_DEFAULT_START_DATE", "2018-01-01")
    
    @classmethod
    def ensure_data_dirs(cls):
        """Create necessary data directories if they don't exist."""
        os.makedirs(cls.DATA_ROOT, exist_ok=True)
        os.makedirs(cls.INDIAN_OCEAN_PATH, exist_ok=True)
        if cls.VECTOR_BACKEND == "chroma":
            os.makedirs(cls.CHROMA_PERSIST_DIR, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate critical configuration settings."""
        if not cls.OPENAI_API_KEY or "sk-proj-" not in cls.OPENAI_API_KEY:
            print("⚠️  Warning: OPENAI_API_KEY not found or invalid in config.py")
            print("   Please add your valid OpenAI API key.")
        
        cls.ensure_data_dirs()
        return True

