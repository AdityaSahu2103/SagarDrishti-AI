"""
Configuration module for ARGO Float Chat system.
Centralized settings for data paths, API tokens, and model configurations.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Data paths
    DATA_ROOT = os.getenv("DATA_ROOT", "argo_data")
    INDIAN_OCEAN_PATH = os.path.join(DATA_ROOT, "indian_ocean")
    
    # ARGO data source
    ARGO_BASE_URL = "https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian"
    DEFAULT_YEAR = "2019"
    DEFAULT_MONTH = "01"
    
    # --- OpenAI API Configuration ---
    # Using OpenAI for the Language Model (LLM)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model configurations
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Stays the same for embeddings
    LLM_MODEL = "gpt-4o-mini"  # Switch to OpenAI's 4o mini model for faster, cost-effective performance
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
    
    # Search settings
    DEFAULT_TOP_K = 3
    FAISS_INDEX_DIMENSION = 384  # all-MiniLM-L6-v2 embedding size
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_TITLE = "FloatChat - ARGO Oceanographic Data Assistant"
    API_VERSION = "1.0.0"
    
    # Visualization settings
    DEFAULT_MAPBOX_STYLE = "carto-positron"
    DEFAULT_COLORSCALE = "Viridis"
    
    # --- MCP (Model Context Protocol) Configuration ---
    # Enable/disable MCP integration and basic connection details
    MCP_ENABLED = os.getenv("MCP_ENABLED", "true").lower() in ("1", "true", "yes")
    MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
    MCP_PORT = int(os.getenv("MCP_PORT", "8765"))
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"ws://{MCP_HOST}:{MCP_PORT}")

    # --- PostgreSQL Configuration ---
    POSTGRES_ENABLED = os.getenv("POSTGRES_ENABLED", "false").lower() in ("1", "true", "yes")
    # Prefer a full URL if provided, else build from discrete parts
    POSTGRES_URL = os.getenv("POSTGRES_URL")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "argo")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_SSLMODE = os.getenv("POSTGRES_SSLMODE", "prefer")

    # --- Vector Store Options ---
    # Supported: 'faiss' (in-memory) or 'chroma' (persistent)
    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss").lower()
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(DATA_ROOT, "chroma"))
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "argo_profiles")
    
    # --- Basic Authentication (for Streamlit Dashboard) ---
    # Enable/disable simple username:password authentication gate for the UI
    AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() in ("1", "true", "yes")
    # Comma-separated list of username:password pairs, e.g. "admin:admin,user2:pass2"
    AUTH_USERS = [u.strip() for u in os.getenv("AUTH_USERS", "admin:admin").split(",") if u.strip()]

    # --- UI Defaults ---
    # Default start date for DB filter widgets in dashboard (ISO format)
    DB_DEFAULT_START_DATE = os.getenv("DB_DEFAULT_START_DATE", "2018-01-01")
    
    @classmethod
    def ensure_data_dirs(cls):
        """Create necessary data directories if they don't exist."""
        os.makedirs(cls.DATA_ROOT, exist_ok=True)
        os.makedirs(cls.INDIAN_OCEAN_PATH, exist_ok=True)
        # Ensure chroma directory if selected
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

