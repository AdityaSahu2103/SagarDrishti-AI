"""
Streamlit dashboard for interactive exploration of ARGO oceanographic data.
Provides a web interface for querying, visualization, and data analysis,
powered by a full Retrieval-Augmented Generation (RAG) pipeline.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import base64
from datetime import datetime, timedelta
import io
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# Optional audio recorder (streamlit-audiorec)
try:
    from st_audiorec import st_audiorec  # type: ignore
except Exception:
    st_audiorec = None  # type: ignore
# Import components from your RAG pipeline
# Ensure these files are in the same directory:
# config.py, data_ingest.py, embedding_index.py, rag_engine.py
try:
    from config import Config
    from data_ingest import ARGODataIngestor
    from embedding_index import ProfileEmbeddingIndex
    from rag_engine import OceanographyRAG
    # Optional DB support
    try:
        from db import get_engine, init_db
    except Exception:
        get_engine = None  # type: ignore
        init_db = None  # type: ignore
    try:
        from mcp_client import MCPClient
    except Exception:
        MCPClient = None  # type: ignore
except ImportError as e:
    st.error(f"Failed to import a necessary module: {e}")
    st.error("Please ensure config.py, data_ingest.py, embedding_index.py, and rag_engine.py are in the same directory as this dashboard script.")
    st.stop()


# --- Page Configuration and Styling ---

st.set_page_config(
    page_title="FloatChat ARGO Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Unified Modern CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Styling */
.stApp {
    font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, #0a1628 0%, #1e3c56 50%, #2d5a7b 100%) !important;
    color: #ffffff !important;
}

/* Force text visibility */
h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown, .stText {
    color: #ffffff !important;
}

.stSubheader {
    color: #42C2FF !important;
    font-weight: 600 !important;
}

/* Main content area */
.main .block-container {
    background: rgba(26, 31, 46, 0.9) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    margin: 1rem !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3) !important;
    padding-top: 2rem !important;
}

/* Header Styling */
.main-header {
    font-size: 3rem !important;
    background: linear-gradient(135deg, #42C2FF 0%, #ffffff 50%, #4ECDC4 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
    font-weight: 700 !important;
}

/* SagarDrishti AI App Header */
.app-header {
    text-align: center !important;
    padding: 1.5rem 0 2rem 0 !important;
    margin-bottom: 1rem !important;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(30, 41, 59, 0.1)) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(148, 163, 184, 0.08) !important;
}

.app-title {
    font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    margin: 0 !important;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(66, 194, 255, 0.15), rgba(78, 205, 196, 0.1)) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(66, 194, 255, 0.3) !important;
    border-radius: 1rem !important;
    padding: 1.5rem !important;
    margin: 0.5rem 0 !important;
    box-shadow: 0 8px 32px rgba(14, 46, 92, 0.3) !important;
    transition: all 0.3s ease !important;
}

.metric-card:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 12px 40px rgba(66, 194, 255, 0.4) !important;
}

.metric-card h3 {
    color: #42C2FF !important;
    margin: 0;
    font-size: 1.1rem;
}

.metric-card h2 {
    color: #ffffff !important;
    margin: 0.5rem 0 0 0;
    font-size: 2.2rem;
}

/* Data at a Glance Container */
.data-at-glance-container {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin: 1rem 0 2rem 0 !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

.data-at-glance-header {
    display: flex !important;
    align-items: center !important;
    margin-bottom: 1.5rem !important;
    padding-bottom: 0.75rem !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2) !important;
}

.glance-icon {
    font-size: 1.25rem !important;
    margin-right: 0.5rem !important;
}

.glance-title {
    font-size: 1.125rem !important;
    font-weight: 600 !important;
    color: #f8fafc !important;
}

.metrics-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    gap: 2rem !important;
}

.glance-metric {
    flex: 1 !important;
    text-align: center !important;
    padding: 0.5rem !important;
}

.metric-label {
    font-size: 0.875rem !important;
    color: #94a3b8 !important;
    margin-bottom: 0.5rem !important;
    font-weight: 500 !important;
}

.metric-value {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #60a5fa !important;
    line-height: 1 !important;
}

/* Responsive design for metrics */
@media (max-width: 768px) {
    .metrics-row {
        flex-direction: column !important;
        gap: 1rem !important;
    }

    .glance-metric {
        width: 100% !important;
    }
}

/* Enhanced Tabs */
.stTabs [data-baseweb="tab"] {
    background: rgba(26, 31, 46, 0.9) !important;
    border: 2px solid #3182ce !important;
    border-radius: 15px !important;
    color: #ffffff !important;
    padding: 1rem 2rem !important;
    margin: 0.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    min-height: 60px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(49, 130, 206, 0.2) !important;
    border-color: #63b3ed !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(49, 130, 206, 0.4) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #3182ce, #63b3ed) !important;
    color: white !important;
    border-color: #63b3ed !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(49, 130, 206, 0.5) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 1rem !important;
    padding: 1rem 0 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(11, 47, 92, 0.95), rgba(46, 134, 171, 0.9)) !important;
    border-right: 1px solid rgba(66, 194, 255, 0.3) !important;
    display: block !important;
    visibility: visible !important;
    width: 300px !important;
}

section[data-testid="stSidebar"] > div {
    display: block !important;
    visibility: visible !important;
}

section[data-testid="stSidebar"] h2 {
    color: #42C2FF !important;
}

section[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff !important;
}

.data-source-badge {
    display: inline-block !important;
    background: #4ECDC4;
    color: white !important;
    padding: 0.3rem 0.8rem !important;
    border-radius: 1rem !important;
    font-size: 0.8rem !important;
    margin: 0.2rem !important;
}

/* Force sidebar visibility */
.css-1d391kg, .css-1lcbmhc, .css-1outpf7 {
    display: block !important;
    visibility: visible !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: linear-gradient(135deg, rgba(66, 194, 255, 0.12), rgba(78, 205, 196, 0.10)) !important;
    border: 1px solid rgba(66, 194, 255, 0.45) !important;
    border-radius: 0.9rem !important;
    color: #ffffff !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 24px rgba(14, 46, 92, 0.35) inset !important;
    padding: 0.75rem !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #42C2FF !important;
    box-shadow: 0 0 0 2px rgba(66, 194, 255, 0.35) !important;
}

/* Textarea wrapper to avoid dark boxes */
.stTextArea > div {
    background: transparent !important;
}
/* Chat Interface */
.query-box {
    background: linear-gradient(135deg, rgba(66, 194, 255, 0.15), rgba(78, 205, 196, 0.1)) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(66, 194, 255, 0.3) !important;
    border-radius: 1rem !important;
    padding: 2rem !important;
    margin: 1rem 0 !important;
    box-shadow: 0 8px 32px rgba(14, 46, 92, 0.3) !important;
}
/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #2E86AB, #42C2FF) !important;
    border: none !important;
    border-radius: 0.8rem !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    padding: 0.5rem 1.5rem !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(66, 194, 255, 0.4) !important;
}

/* Sidebar Refresh Button - match theme */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, rgba(66, 194, 255, 0.25), rgba(78, 205, 196, 0.2)) !important;
    border: 1px solid rgba(66, 194, 255, 0.45) !important;
    color: #ffffff !important;
    width: 100% !important;
    border-radius: 0.9rem !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 24px rgba(14, 46, 92, 0.35) !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #42C2FF, #4ECDC4) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 28px rgba(66, 194, 255, 0.5) !important;
    border-color: rgba(66, 194, 255, 0.6) !important;
}

/* Suggestion Buttons */
div[data-testid="column"] .stButton > button {
    background: linear-gradient(135deg, rgba(66, 194, 255, 0.2), rgba(78, 205, 196, 0.15)) !important;
    border: 1px solid rgba(66, 194, 255, 0.3) !important;
    color: #42C2FF !important;
    font-weight: 500 !important;
    width: 100% !important;
    margin: 0.25rem 0 !important;
    min-height: 70px !important;
    white-space: normal !important;
    line-height: 1.25 !important;
    text-align: center !important;
    padding: 10px 14px !important;
}

div[data-testid="column"] .stButton > button:hover {
    background: linear-gradient(135deg, #42C2FF, #4ECDC4) !important;
    color: white !important;
    transform: translateY(-1px) !important;
}

.response-message {
    background: linear-gradient(135deg, rgba(66,194,255,0.12), rgba(78,205,196,0.10)) !important;
    border-left: 5px solid #42C2FF !important;
    border-radius: 0.8rem !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    backdrop-filter: blur(10px) !important;
}

/* Expander Styling */
details[data-testid="stExpander"] {
    background: linear-gradient(135deg, rgba(66, 194, 255, 0.12), rgba(78, 205, 196, 0.10)) !important;
    border: 1px solid rgba(66, 194, 255, 0.35) !important;
    border-radius: 0.9rem !important;
    overflow: hidden !important;
}

details[data-testid="stExpander"] > summary {
    background: linear-gradient(135deg, rgba(66, 194, 255, 0.18), rgba(78, 205, 196, 0.12)) !important;
    border-bottom: 1px solid rgba(66, 194, 255, 0.25) !important;
    padding: 0.8rem 1rem !important;
    color: #ffffff !important;
}

details[data-testid="stExpander"] > div[role="region"] {
    padding: 0.8rem 1rem !important;
}

details[data-testid="stExpander"]:hover {
    box-shadow: 0 6px 22px rgba(66,194,255,0.25) !important;
}

/* Code blocks and Tables */
.stCode, .stMarkdown pre, .stMarkdown code {
    background: #0f2032 !important;
    border: 1px solid rgba(66,194,255,0.25) !important;
    border-radius: 10px !important;
}

.stDataFrame, .stTable {
    border: 1px solid rgba(66, 194, 255, 0.25) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 6px 22px rgba(14, 46, 92, 0.25) !important;
}

.stDataFrame table, .stTable table {
    background: rgba(10, 22, 40, 0.35) !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: rgba(10,22,40,0.2); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #2E86AB, #42C2FF);
    border-radius: 10px;
    border: 2px solid rgba(10,22,40,0.3);
}

/* Section Styling */
.stMarkdown h2 {
    color: #63b3ed !important;
    border-bottom: 2px solid #3182ce !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 1.5rem !important;
}

.stMarkdown h3 {
    color: #ffffff !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header, .stDeployButton {
    visibility: hidden !important;
    display: none !important;
}

/* Responsive design */
@media (max-width: 900px) {
    .main .block-container {
        padding-top: 1rem !important;
    }
    .stButton > button {
        width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)


# --- RAG Pipeline Initialization ---

@st.cache_resource
def initialize_system():
    """
    Initialize and cache the full RAG pipeline: Config, Ingestor, Embeddings, and RAG Engine.
    The loaded data is stored in memory to accelerate visualizations.
    """
    try:
        with st.spinner("🚀 Launching FloatChat Engine... This may take a moment on the first run."):
            import gc  # For garbage collection
            config = Config()
            config.validate_config()

            ingestor = ARGODataIngestor(config)

            # Process a limited number of NetCDF files to prevent memory issues
            # Limit to first 200 files for better performance
            profiles = ingestor.process_all_files()
            if len(profiles) > 100:
                profiles = profiles[:100]  # Limit to first 100 profiles
            if not profiles:
                st.error(f"⚠️ No ARGO data files found in the specified directory: `{config.INDIAN_OCEAN_PATH}`. Please add NetCDF files to this folder.")
                return None

            embedding_index = ProfileEmbeddingIndex(config)
            embedding_index.create_embeddings(profiles)
            # If cache was loaded but profile count changed since last run, rebuild embeddings
            try:
                stats_now = embedding_index.get_statistics()
                if stats_now.get('total_profiles', 0) != len(profiles):
                    # Force rebuild to synchronize with current data
                    embedding_index.create_embeddings(profiles, force_rebuild=True)
            except Exception:
                # Best-effort safeguard; proceed if comparison fails
                pass

            # Initialize MCP client (optional)
            mcp_client = None
            try:
                if getattr(config, "MCP_ENABLED", False) and MCPClient is not None:
                    mcp_client = MCPClient(getattr(config, "MCP_SERVER_URL", None))
            except Exception:
                mcp_client = None

            rag_system = OceanographyRAG(embedding_index, config, mcp_client=mcp_client)
            rag_system.initialize_llm()
            if rag_system.llm is None:
                 st.warning("⚠️ LLM (Language Model) didnt initialize. The chat interface will use a simplified, rule-based response mode.")

            # Force garbage collection to free memory
            gc.collect()

        return {
            "config": config,
            "ingestor": ingestor,
            "embedding_index": embedding_index,
            "rag_system": rag_system,
            "all_profiles_data": profiles  # Store loaded dataframes for visualizations
        }
    except Exception as e:
        st.error(f"❌ Critical Error during initialization: {e}")
        st.error("Please check your configuration, API tokens, and data paths.")
        return None

# --- Helper Functions ---

def get_region(latitude, longitude):
    """Classify ocean basin globally based on lat/lon.

    Returns labels like 'North Pacific Ocean', 'South Atlantic Ocean',
    'Indian Ocean', 'Southern Ocean', or 'Arctic Ocean'.
    """
    try:
        lat = float(latitude)
        lon = float(longitude)
    except Exception:
        return "Unknown"

    # Normalize longitude to [-180, 180]
    if lon > 180:
        lon = ((lon + 180) % 360) - 180
    if lon < -180:
        lon = ((lon - 180) % 360) + 180

    # Polar oceans
    if lat >= 66.0:
        return "Arctic Ocean"
    if lat <= -50.0:
        return "Southern Ocean"

    # Main basins based on broad longitude bands
    # Atlantic: [-70, 20]
    # Indian: [20, 146]
    # Pacific: lon <= -70 or lon >= 146
    if -70.0 <= lon <= 20.0:
        basin = "Atlantic Ocean"
    elif 20.0 < lon < 146.0:
        basin = "Indian Ocean"
    else:
        basin = "Pacific Ocean"

    # Hemisphere prefix for Atlantic/Pacific
    if basin in ("Atlantic Ocean", "Pacific Ocean"):
        hemi = "North" if lat >= 0 else "South"
        return f"{hemi} {basin}"
    else:
        return basin

# --- UI Display Functions ---

# --- Authentication Helpers ---
def _parse_auth_users(cfg: Config):
    """Parse Config.AUTH_USERS into a dict {username: password}."""
    users = {}
    try:
        for pair in getattr(cfg, 'AUTH_USERS', []) or []:
            if ":" in pair:
                u, p = pair.split(":", 1)
                users[u.strip()] = p.strip()
    except Exception:
        pass
    return users

def _ensure_auth_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'auth_user' not in st.session_state:
        st.session_state.auth_user = None

def _auth_gate(cfg: Config) -> bool:
    """Render a login form when AUTH is enabled. Returns True if user is authenticated."""
    _ensure_auth_state()
    if not getattr(cfg, 'AUTH_ENABLED', True):
        # Auth disabled
        st.session_state.authenticated = True
        st.session_state.auth_user = "anonymous"
        return True

    if st.session_state.authenticated:
        return True

    st.markdown("## 🔐 Sign in to FloatChat")
    st.caption("Access is restricted. Please sign in.")
    users = _parse_auth_users(cfg)
    with st.form("login_form"):
        colu1, colu2 = st.columns(2)
        with colu1:
            username = st.text_input("Username", key="login_user")
        with colu2:
            password = st.text_input("Password", type="password", key="login_pass")
        submit = st.form_submit_button("Sign In")

    if submit:
        if username in users and users[username] == password:
            st.session_state.authenticated = True
            st.session_state.auth_user = username
            st.success("Signed in successfully. Redirecting…")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

    # Stop rendering rest of the app
    return False

def display_header():
    """Display the main page header."""
    st.markdown(
        '''
        <div class="app-header">
            <h1 class="app-title">🌊 SagarDrishti AI</h1>
        </div>
        ''',
        unsafe_allow_html=True
    )

def render_theme_toggle():
    """Theme toggle removed as requested"""
    pass

def render_data_bubbles(system_components):
    """Render floating data bubbles"""
    embedding_index = system_components["embedding_index"]
    metadata = embedding_index.profile_metadata
    
    if not metadata:
        return
    
    total_profiles = len(metadata)
    unique_regions = len(set(get_region(m['latitude'], m['longitude']) for m in metadata))
    
    times = [m['time'] for m in metadata]
    time_span_days = (max(times) - min(times)).days if times else 0
    
    all_temps = [item for df in system_components["all_profiles_data"] for item in df['temperature'].tolist()]
    avg_temp = np.nanmean(all_temps) if all_temps else 0
    
    st.markdown("""
    
    """.format(total_profiles, unique_regions, time_span_days, avg_temp), unsafe_allow_html=True)

def create_3d_globe(system_components):
    """Create interactive 3D globe with ARGO float locations"""
    profiles_data = system_components["all_profiles_data"]
    if not profiles_data:
        return None
    
    globe_data = []
    for df in profiles_data:
        if not df.empty:
            globe_data.append({
                'lat': df['latitude'].iloc[0],
                'lon': df['longitude'].iloc[0],
                'temp': df['temperature'].mean(),
                'salinity': df['salinity'].mean(),
                'depth': df['pressure'].max(),
                'region': get_region(df['latitude'].iloc[0], df['longitude'].iloc[0]),
                'file_source': df['file_source'].iloc[0]
            })
    
    df_globe = pd.DataFrame(globe_data)
    
    fig = go.Figure()
    
    # Create custom colorscale for better visibility
    custom_colorscale = [
        [0.0, '#1f4e79'],  # Dark blue for cold
        [0.2, '#2e5984'],  # Medium blue
        [0.4, '#4a90a4'],  # Teal
        [0.6, '#7bb3d3'],  # Light blue
        [0.8, '#a8d8ea'],  # Very light blue
        [1.0, '#ff6b6b']   # Red for warm
    ]
    
    fig.add_trace(go.Scattergeo(
        mode='markers',
        lon=df_globe['lon'],
        lat=df_globe['lat'],
        marker=dict(
            size=12,
            color=df_globe['temp'],
            colorscale=custom_colorscale,
            colorbar=dict(
                title=dict(
                    text="Temperature (°C)",
                    font=dict(color='white', size=14)
                ),
                tickfont=dict(color='white'),
                bgcolor='rgba(26, 31, 46, 0.8)',
                bordercolor='#3182ce',
                borderwidth=2
            ),
            line=dict(width=2, color='white'),
            opacity=0.9,
            symbol='circle'
        ),
        text=df_globe.apply(lambda x: f"Region: {x['region']}<br>Temp: {x['temp']:.2f}°C<br>Salinity: {x['salinity']:.2f} PSU<br>Max Depth: {x['depth']:.0f} dbar", axis=1),
        hovertemplate='%{text}<extra></extra>',
        name='ARGO Floats'
    ))
    
    # Add trajectory lines with better visibility
    for region in df_globe['region'].unique():
        region_data = df_globe[df_globe['region'] == region]
        if len(region_data) > 1:
            fig.add_trace(go.Scattergeo(
                mode='lines',
                lon=region_data['lon'],
                lat=region_data['lat'],
                line=dict(width=3, color='rgba(99, 179, 237, 0.6)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title=dict(
            text="ARGO Float Locations - Interactive 3D Globe",
            font=dict(color='white', size=20)
        ),
        geo=dict(
            projection_type='orthographic',
            showocean=True,
            oceancolor='rgba(10, 14, 26, 0.8)',
            showland=True,
            landcolor='rgba(100, 100, 100, 0.9)',
            showlakes=True,
            lakecolor='rgba(49, 130, 206, 0.7)',
            showrivers=True,
            rivercolor='rgba(99, 179, 237, 0.5)',
            showframe=True,
            framecolor='#3182ce',
            framewidth=2,
            bgcolor='rgba(0,0,0,0)',
            coastlinecolor='#63b3ed',
            coastlinewidth=1
        ),
        height=600,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def render_enhanced_voice_input():
    """Render enhanced voice input with circular button"""
    st.markdown("""
    <div class="voice-recorder">
        <button class="record-button" id="recordBtn">
            🎤
        </button>
    </div>
    """, unsafe_allow_html=True)
    
    if st_audiorec is not None:
        audio_data = st_audiorec()
        if audio_data:
            return audio_data
    return None

def render_project_info_section():
    """Render ARGO Project information section"""
    st.markdown("""
    <div class="project-info">
        <h2 style="color: #63b3ed; margin-bottom: 2rem; text-align: center;">
            About The ARGO Project
        </h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: center;">
            <div>
                <h3 style="color: #ffffff; margin-bottom: 1rem;">Global Ocean Observing Network</h3>
                <p style="color: #a0aec0; line-height: 1.6; margin-bottom: 1rem;">
                    The ARGO program is an international collaboration that maintains a global array of autonomous 
                    profiling floats to monitor the temperature, salinity, and velocity of the upper ocean.
                </p>
                <p style="color: #a0aec0; line-height: 1.6; margin-bottom: 1rem;">
                    With over 3,900 active floats worldwide, ARGO provides real-time data that is essential for 
                    climate research, weather prediction, and ocean monitoring.
                </p>
                <div style="display: flex; gap: 1rem; margin-top: 2rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #63b3ed;">3,900+</div>
                        <div style="color: #a0aec0;">Active Floats</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #63b3ed;">2000+</div>
                        <div style="color: #a0aec0;">Profiles/Day</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #63b3ed;">30+</div>
                        <div style="color: #a0aec0;">Countries</div>
                    </div>
                </div>
            </div>
            <div class="video-placeholder">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📹</div>
                <h4 style="color: #ffffff; margin-bottom: 1rem;">Project Showcase Video</h4>
                <p style="color: #a0aec0;">
                    YouTube video embedding will be added here to showcase our project film.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar(system_components):
    """Display the sidebar with system information and controls."""
    # Ensure sidebar is visible
    with st.sidebar:
        # Account / Authentication controls
        st.markdown("## 👤 Account")
        if st.session_state.get("authenticated", False):
            user_lbl = st.session_state.get("auth_user") or "anonymous"
            st.markdown(f"Signed in as: `{user_lbl}`")
            if st.button("Logout", help="Sign out and clear session", use_container_width=True):
                try:
                    # Clear cached resources to avoid leaking state across users
                    st.cache_resource.clear()
                except Exception:
                    pass
                st.session_state.authenticated = False
                st.session_state.auth_user = None
                st.rerun()
        else:
            st.info("Not signed in")

        st.markdown("## 🔧 System Status")
        
        embedding_index = system_components["embedding_index"]
        stats = embedding_index.get_statistics()

        st.metric("Total Profiles Loaded", stats.get('total_profiles', 0))
        st.markdown(f"**Embedding Model:** `{stats.get('embedding_model', 'N/A')}`")

        st.markdown("---")
        st.markdown("## 📊 Data Sources")
        try:
            metadata = embedding_index.profile_metadata or []
            regions = []
            for m in metadata:
                try:
                    lat = m.get('latitude'); lon = m.get('longitude')
                    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                        regions.append(get_region(lat, lon))
                except Exception:
                    pass
            regions = sorted(set([r for r in regions if r]))
            # Build dynamic badges for live regions
            badges_html = '<span class="data-source-badge">ARGO Program</span>'
            for r in regions:
                badges_html += f' <span class="data-source-badge">{r}</span>'
            if not regions:
                badges_html += ' <span class="data-source-badge">No regions detected</span>'
            st.markdown(badges_html, unsafe_allow_html=True)
        except Exception:
            st.markdown("<span class=\"data-source-badge\">ARGO Program</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## ⚡ Quick Actions")
        if st.button("🔄 Refresh Data & System", help="Clears cache and reloads all data and models."):
            try:
                config = system_components.get("config")
                if config:
                    cache_path = os.path.join(config.DATA_ROOT, "embeddings_cache.pkl")
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                        st.toast("Cleared embeddings cache.", icon="🗑️")
            except Exception as e:
                st.warning(f"Could not clear custom cache file: {e}")
                
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        search_results = st.slider("Search Results (Top K)", 1, 10, 3, help="Number of profiles to retrieve for each query.", key="search_results_slider")

        # LLM Max Tokens control
        try:
            cfg = system_components.get("config")
            rag_system = system_components.get("rag_system")
            current_max = int(getattr(cfg, "LLM_MAX_TOKENS", 1200)) if cfg else 1200
            new_max = st.slider(
                "LLM Max Tokens (answer length)",
                min_value=256,
                max_value=4096,
                value=current_max,
                step=128,
                help="Higher values allow longer answers but cost more tokens.",
                key="llm_max_tokens_slider"
            )
            if cfg and new_max != current_max:
                cfg.LLM_MAX_TOKENS = int(new_max)
                # Reinitialize LLM with new token limit
                if rag_system:
                    rag_system.llm = None
                    rag_system.initialize_llm()
                st.toast(f"Updated LLM max tokens to {new_max}.", icon="✅")
        except Exception as e:
            st.caption(f"LLM token control unavailable: {e}")

        # Add LLM status indicator
        st.markdown("---")
        st.markdown("## 🤖 LLM Status")
        rag_system = system_components.get("rag_system")
        if rag_system and rag_system.llm:
            st.success("✅ LLM Active")
        else:
            st.warning("⚠️ Rule-based Mode")

        # MCP status
        st.markdown("---")
        st.markdown("## 🧩 MCP Status")
        try:
            cfg = system_components.get("config")
            mcp_enabled = getattr(cfg, "MCP_ENABLED", False)
            if mcp_enabled and rag_system and getattr(rag_system, "mcp_client", None) is not None:
                st.success("✅ MCP Enabled")
            elif mcp_enabled:
                st.warning("⚠️ MCP Enabled but client not initialized")
            else:
                st.info("ℹ️ MCP Disabled")
        except Exception:
            st.info("ℹ️ MCP status unavailable")

    return {'search_results': search_results}

# --- Database Utilities ---
@st.cache_resource
def get_db_connection(_cfg: Config):
    """Return a SQLAlchemy engine for PostgreSQL if enabled, else None."""
    try:
        if not getattr(_cfg, 'POSTGRES_ENABLED', False):
            return None
        if get_engine is None:
            return None
        if init_db is not None:
            try:
                init_db(_cfg)
            except Exception:
                pass
        engine = get_engine(_cfg)
        return engine
    except Exception as e:
        st.warning(f"PostgreSQL not available: {e}")
        return None

def _run_sql_df(engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    try:
        return pd.read_sql(sql, con=engine, params=params)
    except Exception as e:
        st.error(f"SQL error: {e}")
        return pd.DataFrame()

def display_database_explorer(system_components):
    """Database tab reflecting queries against Postgres schema."""
    cfg: Config = system_components["config"]
    engine = get_db_connection(cfg)

    st.subheader("🗄️ Database Explorer (PostgreSQL)")
    if engine is None:
        st.info("PostgreSQL is disabled or not reachable. Set POSTGRES_ENABLED=true and configure connection in .env to use this tab.")
        return

    # Quick metrics
    colA, colB, colC = st.columns(3)
    with colA:
        df_counts = _run_sql_df(engine, "SELECT COUNT(*) AS profiles FROM public.argo_profiles")
        st.markdown(f'<div class="metric-card"><h3>Profiles</h3><h2>{int(df_counts.get("profiles", pd.Series([0])).iloc[0]) if not df_counts.empty else 0}</h2></div>', unsafe_allow_html=True)
    with colB:
        df_meas = _run_sql_df(engine, "SELECT COUNT(*) AS measurements FROM public.argo_measurements")
        st.markdown(f'<div class="metric-card"><h3>Measurements</h3><h2>{int(df_meas.get("measurements", pd.Series([0])).iloc[0]) if not df_meas.empty else 0}</h2></div>', unsafe_allow_html=True)
    with colC:
        df_time = _run_sql_df(engine, "SELECT MIN(time) AS min_t, MAX(time) AS max_t FROM public.argo_profiles")
        if not df_time.empty and pd.notna(df_time.loc[0, 'min_t']) and pd.notna(df_time.loc[0, 'max_t']):
            span = (pd.to_datetime(df_time.loc[0, 'max_t']) - pd.to_datetime(df_time.loc[0, 'min_t'])).days
        else:
            span = 0
        st.markdown(f'<div class="metric-card"><h3>Time Span</h3><h2>{span} days</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔎 Filter Profiles")
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        # Default to Config.DB_DEFAULT_START_DATE if available
        try:
            _default_start = pd.to_datetime(system_components["config"].DB_DEFAULT_START_DATE).date()
        except Exception:
            _default_start = (datetime.utcnow() - timedelta(days=365)).date()
        start_date = st.date_input("Start Date", value=_default_start, key="start_date_input")
    with col2:
        end_date = st.date_input("End Date", value=datetime.utcnow().date(), key="end_date_input")
    with col3:
        lat_band = st.slider("Latitude band", -90.0, 90.0, (-10.0, 10.0), key="lat_band_slider")

    col4, col5 = st.columns(2)
    with col4:
        lon_band = st.slider("Longitude band", -180.0, 180.0, (60.0, 120.0), key="lon_band_slider")
    with col5:
        min_n = st.number_input("Min measurements", value=20, step=10, key="min_measurements_input")

    sql_profiles = (
        """
        SELECT id, file_source, profile_idx, latitude, longitude, time,
               n_measurements, depth_range, temp_range, summary
        FROM public.argo_profiles
        WHERE time BETWEEN %(start)s AND %(end)s
          AND latitude BETWEEN %(lat_min)s AND %(lat_max)s
          AND longitude BETWEEN %(lon_min)s AND %(lon_max)s
          AND n_measurements >= %(min_n)s
        ORDER BY time ASC
        """
    )
    params = {
        "start": pd.Timestamp(start_date),
        "end": pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
        "lat_min": float(lat_band[0]),
        "lat_max": float(lat_band[1]),
        "lon_min": float(lon_band[0]),
        "lon_max": float(lon_band[1]),
        "min_n": int(min_n),
    }
    df_profiles = _run_sql_df(engine, sql_profiles, params)

    with st.expander("View Profiles (from DB)", expanded=True):
        st.dataframe(df_profiles, use_container_width=True)
        # Show copy-pastable SQL for the above filter with inlined parameters
        try:
            start_iso = pd.Timestamp(params["start"]).strftime('%Y-%m-%d %H:%M:%S')
            end_iso = pd.Timestamp(params["end"]).strftime('%Y-%m-%d %H:%M:%S')
            copy_sql_profiles = f"""
SELECT id, file_source, profile_idx, latitude, longitude, time,
       n_measurements, depth_range, temp_range, summary
FROM public.argo_profiles
WHERE time BETWEEN '{start_iso}' AND '{end_iso}'
  AND latitude BETWEEN {params['lat_min']} AND {params['lat_max']}
  AND longitude BETWEEN {params['lon_min']} AND {params['lon_max']}
  AND n_measurements >= {params['min_n']}
ORDER BY time ASC;
"""
            st.markdown("#### Copy-paste SQL for this result")
            st.code(copy_sql_profiles.strip(), language="sql")
        except Exception:
            pass

    # Map from DB
    if not df_profiles.empty:
        try:
            # Compute regions for legend entries
            try:
                df_profiles["Region"] = df_profiles.apply(lambda r: get_region(r["latitude"], r["longitude"]), axis=1)
            except Exception:
                df_profiles["Region"] = "Unknown"

            fig = px.scatter_mapbox(
                df_profiles.rename(columns={"latitude": "Latitude", "longitude": "Longitude"}),
                lat='Latitude', lon='Longitude', color='n_measurements',
                hover_data=['file_source', 'profile_idx', 'time', 'Region'],
                mapbox_style='carto-positron', zoom=2,
                title="Profiles (from PostgreSQL)"
            )
            # Add legend entries for ocean names (regions) without changing the color scale
            try:
                for region in sorted(df_profiles["Region"].dropna().unique().tolist()):
                    sample = df_profiles[df_profiles["Region"] == region].head(1)
                    if not sample.empty:
                        fig.add_trace(
                            go.Scattermapbox(
                                lat=[float(sample.iloc[0]["latitude"])],
                                lon=[float(sample.iloc[0]["longitude"])],
                                mode="markers",
                                marker=dict(size=1, opacity=0),
                                name=str(region),
                                hoverinfo="skip",
                                showlegend=True,
                                legendgroup="Region"
                            )
                        )
            except Exception:
                pass
            fig.update_layout(height=450, margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True, key="database_map_chart")
        except Exception as e:
            st.warning(f"Map rendering failed: {e}")

    # Profile detail and measurements
    st.markdown("### 📈 Profile Detail")
    if not df_profiles.empty:
        sel = st.selectbox(
            "Select a profile to view measurements",
            df_profiles[['id','file_source','profile_idx']].apply(lambda r: f"ID {r['id']} - {r['file_source']} (idx {r['profile_idx']})", axis=1),
            index=0,
            key="profile_detail_selectbox"
        )
        # Extract chosen id
        chosen_id = int(df_profiles.iloc[0].id) if not df_profiles.empty else None
        try:
            # Attempt to parse ID from selection
            chosen_id = int(sel.split(' ')[1])
        except Exception:
            pass

        if chosen_id is not None:
            df_meas_sel = _run_sql_df(engine, "SELECT pressure, temperature, salinity FROM public.argo_measurements WHERE profile_id = %(pid)s ORDER BY pressure ASC", {"pid": chosen_id})
            colm1, colm2 = st.columns(2)
            with colm1:
                st.dataframe(df_meas_sel.head(500), use_container_width=True)
            with colm2:
                try:
                    figm = go.Figure()
                    if 'temperature' in df_meas_sel and 'pressure' in df_meas_sel:
                        figm.add_trace(go.Scatter(x=df_meas_sel['temperature'], y=df_meas_sel['pressure'], mode='lines', name='Temp'))
                    if 'salinity' in df_meas_sel and 'pressure' in df_meas_sel:
                        figm.add_trace(go.Scatter(x=df_meas_sel['salinity'], y=df_meas_sel['pressure'], mode='lines', name='Salinity'))
                    figm.update_layout(title="Temp/Salinity vs Depth", xaxis_title="Value", yaxis_title="Pressure (dbar)", yaxis=dict(autorange='reversed'))
                    st.plotly_chart(figm, use_container_width=True, key="profile_measurements_chart")
                except Exception as e:
                    st.warning(f"Plot failed: {e}")

    st.markdown("---")
    st.markdown("### 🧰 Prebuilt Queries")
    query_options = {
        "Average temperature/salinity by profile": (
            """
            SELECT p.id AS profile_id, p.file_source, p.profile_idx,
                   AVG(m.temperature) AS avg_temp, AVG(m.salinity) AS avg_salinity,
                   MIN(m.pressure) AS min_pressure, MAX(m.pressure) AS max_pressure
            FROM public.argo_profiles p
            JOIN public.argo_measurements m ON m.profile_id = p.id
            GROUP BY p.id, p.file_source, p.profile_idx
            ORDER BY avg_temp DESC
            """
        ),
        "Deep-water high-salinity (>35 PSU) at >1000 dbar": (
            """
            SELECT DISTINCT p.id, p.file_source, p.profile_idx, p.latitude, p.longitude, p.time
            FROM public.argo_profiles p
            JOIN public.argo_measurements m ON m.profile_id = p.id
            WHERE m.pressure >= 1000 AND m.salinity >= 35
            ORDER BY p.time
            """
        ),
        "Month-over-month mean temperature trend": (
            """
            SELECT DATE_TRUNC('month', p.time) AS month, AVG(m.temperature) AS mean_temp_c
            FROM public.argo_profiles p
            JOIN public.argo_measurements m ON m.profile_id = p.id
            GROUP BY 1
            ORDER BY 1
            """
        ),
        "Monthly profile counts": (
            """
            SELECT DATE_TRUNC('month', time) AS month, COUNT(*) AS profiles
            FROM public.argo_profiles
            GROUP BY 1
            ORDER BY 1
            """
        ),
        "Warmest surface temps (pressure<=10 dbar)": (
            """
            SELECT p.id, p.file_source, p.profile_idx, p.latitude, p.longitude, p.time,
                   MAX(m.temperature) AS max_surface_temp
            FROM public.argo_profiles p
            JOIN public.argo_measurements m ON m.profile_id = p.id
            WHERE m.pressure <= 10
            GROUP BY p.id, p.file_source, p.profile_idx, p.latitude, p.longitude, p.time
            ORDER BY max_surface_temp DESC
            LIMIT 50
            """
        ),
        "High salinity 36+ at 500–1500 dbar": (
            """
            SELECT DISTINCT p.id, p.file_source, p.profile_idx, p.latitude, p.longitude, p.time
            FROM public.argo_profiles p
            JOIN public.argo_measurements m ON m.profile_id = p.id
            WHERE m.pressure BETWEEN 500 AND 1500
              AND m.salinity >= 36
            ORDER BY p.time DESC
            LIMIT 100
            """
        ),
        "Profiles in Arabian Sea bbox": (
            """
            SELECT id, file_source, profile_idx, latitude, longitude, time
            FROM public.argo_profiles
            WHERE latitude BETWEEN 8 AND 25
              AND longitude BETWEEN 60 AND 78
            ORDER BY time DESC
            LIMIT 200
            """
        ),
        "Latest 100 profiles": (
            """
            SELECT id, file_source, profile_idx, latitude, longitude, time, n_measurements
            FROM public.argo_profiles
            ORDER BY time DESC
            LIMIT 100
            """
        ),
        "T-S sample (measurements)": (
            """
            SELECT p.id AS profile_id, p.time, p.latitude, p.longitude,
                   m.temperature, m.salinity, m.pressure
            FROM public.argo_profiles p
            JOIN public.argo_measurements m ON m.profile_id = p.id
            WHERE m.temperature IS NOT NULL AND m.salinity IS NOT NULL
            ORDER BY p.time DESC
            LIMIT 1000
            """
        ),
    }
    qlabel = st.selectbox("Select a prebuilt query", list(query_options.keys()), index=0, key="prebuilt_query_selectbox")
    # Show the raw SQL so it can be copy–pasted into pgAdmin
    st.markdown("#### Copy-paste SQL")
    st.code(query_options[qlabel].strip(), language="sql")
    if st.button("Run Query", key="run_query_btn"):
        dfq = _run_sql_df(engine, query_options[qlabel])
        st.dataframe(dfq, use_container_width=True)

def display_overview_metrics(system_components):
    """Display key data metrics in cards using real data."""
    embedding_index = system_components["embedding_index"]
    metadata = embedding_index.profile_metadata

    if not metadata:
        st.warning("No profile metadata available.")
        return

    total_profiles = len(metadata)
    unique_regions = len(set(get_region(m['latitude'], m['longitude']) for m in metadata))

    times = [m['time'] for m in metadata]
    time_span_days = (max(times) - min(times)).days if times else 0

    all_temps = [item for df in system_components["all_profiles_data"] for item in df['temperature'].tolist()]
    avg_temp = np.nanmean(all_temps) if all_temps else 0

    # Create the Data at a Glance container with dark background
    st.markdown(f'''
    <div class="data-at-glance-container">
        <div class="data-at-glance-header">
            <span class="glance-icon">📊</span>
            <span class="glance-title">Data at a Glance</span>
        </div>
        <div class="metrics-row">
            <div class="glance-metric">
                <div class="metric-label">Total Profiles</div>
                <div class="metric-value">{total_profiles}</div>
            </div>
            <div class="glance-metric">
                <div class="metric-label">Regions Covered</div>
                <div class="metric-value">{unique_regions}</div>
            </div>
            <div class="glance-metric">
                <div class="metric-label">Time Span</div>
                <div class="metric-value">{time_span_days} days</div>
            </div>
            <div class="glance-metric">
                <div class="metric-label">Avg. Temp</div>
                <div class="metric-value">{avg_temp:.2f}°C</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def display_geographic_map(system_components):
    """Display an interactive map of ARGO float locations."""
    st.subheader("🗺️ Geographic Distribution of Floats")
    
    profiles_data = system_components["all_profiles_data"]
    if not profiles_data:
        st.warning("No profile data to display on the map.")
        return

    map_df_data = []
    for df in profiles_data:
        if not df.empty:
            map_df_data.append({
                'Latitude': df['latitude'].iloc[0],
                'Longitude': df['longitude'].iloc[0],
                'Profile ID': df['file_source'].iloc[0],
                'Mean Temperature': df['temperature'].mean(),
                'Max Depth': df['pressure'].max(),
                'Region': get_region(df['latitude'].iloc[0], df['longitude'].iloc[0])
            })
            
    df = pd.DataFrame(map_df_data)
    # Ensure Max Depth is numeric and use absolute values (depths can be negative in source data)
    if not df.empty:
        df['Max Depth'] = pd.to_numeric(df['Max Depth'], errors='coerce').fillna(0.0).abs()
    
    with st.expander("View Raw Map Data"):
        st.dataframe(df, use_container_width=True)

    fig = px.scatter_mapbox(
        df, lat='Latitude', lon='Longitude', color='Mean Temperature', size='Max Depth',
        hover_data=['Profile ID', 'Region'], color_continuous_scale=px.colors.sequential.Viridis,
        mapbox_style='carto-positron', zoom=3, center={'lat': 10, 'lon': 75},
        title="ARGO Float Locations (Color: Mean Temp, Size: Max Depth)"
    )
    # Add legend entries for each ocean/region
    try:
        for region in sorted(df['Region'].dropna().unique().tolist()):
            sample = df[df['Region'] == region].head(1)
            if not sample.empty:
                fig.add_trace(
                    go.Scattermapbox(
                        lat=[float(sample.iloc[0]['Latitude'])],
                        lon=[float(sample.iloc[0]['Longitude'])],
                        mode='markers',
                        marker=dict(size=1, opacity=0),
                        name=str(region),
                        hoverinfo='skip',
                        showlegend=True,
                        legendgroup='Region'
                    )
                )
    except Exception:
        pass
    fig.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True, key="geographic_map_chart")

# --- Insights Computation Helpers ---
def _compute_profile_metrics(profile_df: pd.DataFrame) -> dict:
    """Compute key metrics from a single ARGO profile DataFrame."""
    if profile_df is None or profile_df.empty:
        return {}
    df = profile_df.sort_values('pressure')
    sst = float(df['temperature'].iloc[0]) if 'temperature' in df else np.nan
    deep_t = float(df['temperature'].iloc[-1]) if 'temperature' in df else np.nan
    mean_t = float(df['temperature'].mean()) if 'temperature' in df else np.nan
    mean_s = float(df['salinity'].mean()) if 'salinity' in df else np.nan
    std_s = float(df['salinity'].std()) if 'salinity' in df else np.nan
    std_t = float(df['temperature'].std()) if 'temperature' in df else np.nan
    max_depth = float(df['pressure'].max()) if 'pressure' in df else np.nan
    # Estimate thermocline depth: first depth where temperature drops >= 2C from surface
    thermocline_depth = np.nan
    try:
        if 'temperature' in df and 'pressure' in df and not df['temperature'].isna().all():
            surface_t = df['temperature'].iloc[0]
            drop_mask = (surface_t - df['temperature']) >= 2.0
            if drop_mask.any():
                thermocline_depth = float(df.loc[drop_mask, 'pressure'].iloc[0])
    except Exception:
        pass
    stratified = (sst - deep_t) >= 2.0 if (not np.isnan(sst) and not np.isnan(deep_t)) else False
    return {
        'sst': sst,
        'deep_t': deep_t,
        'mean_t': mean_t,
        'mean_s': mean_s,
        'std_s': std_s,
        'std_t': std_t,
        'max_depth': max_depth,
        'thermocline_depth': thermocline_depth,
        'stratified': stratified,
    }

def _aggregate_metrics(context_dfs: list[pd.DataFrame]) -> dict:
    """Aggregate metrics across multiple retrieved profile DataFrames."""
    if not context_dfs:
        return {}
    metrics = [_compute_profile_metrics(df) for df in context_dfs if df is not None and not df.empty]
    if not metrics:
        return {}
    def avg(key):
        vals = [m[key] for m in metrics if key in m and not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else np.nan
    agg = {
        'profiles_count': len(metrics),
        'sst_mean': avg('sst'),
        'deep_t_mean': avg('deep_t'),
        'mean_t_mean': avg('mean_t'),
        'mean_s_mean': avg('mean_s'),
        'std_s_mean': avg('std_s'),
        'std_t_mean': avg('std_t'),
        'max_depth_mean': avg('max_depth'),
        'thermocline_depth_mean': avg('thermocline_depth'),
        'stratified_share': float(np.mean([1.0 if m.get('stratified') else 0.0 for m in metrics])) if metrics else 0.0,
    }
    return agg

# --- UI Utility: Image Gallery Renderer ---
def render_image_gallery(gallery: list[dict], columns: int = 3):
    """Render a responsive, aesthetic image gallery with fixed-size images and captions.
    Each item in gallery should be a dict with keys 'url' and 'caption'.
    """
    if not gallery:
        return
    for i in range(0, len(gallery), columns):
        cols = st.columns(columns)
        for j, col in enumerate(cols):
            if i + j < len(gallery):
                item = gallery[i + j]
                with col:
                    st.markdown(
                        f'<img class="gallery-img img-card" src="{item["url"]}" alt="{item.get("caption", "image")}">',
                        unsafe_allow_html=True
                    )
                    st.caption(item.get('caption', ''))

def generate_industry_insights_from_profiles(retrieved_profiles: list[dict], all_profiles_data: list[pd.DataFrame]) -> list[tuple[str, str]]:
    """Map retrieved profile data to industry-specific insights.
    Returns a list of (industry, insight) tuples.
    """
    if not retrieved_profiles:
        return []
    # Match retrieved profiles to full DataFrames by file_source
    retrieved_files = {p.get('file_source') for p in retrieved_profiles if p.get('file_source')}
    context_dfs = [df for df in all_profiles_data if not df.empty and df['file_source'].iloc[0] in retrieved_files]
    agg = _aggregate_metrics(context_dfs)
    if not agg:
        return []
    sst = agg.get('sst_mean', np.nan)
    deep_t = agg.get('deep_t_mean', np.nan)
    mean_s = agg.get('mean_s_mean', np.nan)
    max_depth = agg.get('max_depth_mean', np.nan)
    thermo = agg.get('thermocline_depth_mean', np.nan)
    strat_share = agg.get('stratified_share', 0.0)

    # Qualitative descriptors (no numbers in the output)
    def temp_feel(val: float) -> str:
        if np.isnan(val):
            return "stable"
        if val >= 28:
            return "very warm"
        if val >= 24:
            return "warm"
        if val >= 20:
            return "mild"
        return "cool"

    sst_desc = temp_feel(sst)
    strat_desc = "strong" if strat_share >= 0.5 else ("moderate" if strat_share >= 0.2 else "weak")
    thermo_desc = "a clear layering between surface and deeper waters" if not np.isnan(thermo) else "little layering between surface and deeper waters"
    sal_desc = "typical ocean salinity" if np.isnan(mean_s) else ("saltier than average" if mean_s > 35 else ("slightly fresher" if mean_s < 34 else "typical ocean salinity"))

    insights: list[tuple[str, str]] = []
    # Climate & Meteorology
    insights.append((
        "🌦️ Climate & Meteorology",
        f"Surface waters are {sst_desc} with {thermo_desc} and {strat_desc} layering — this helps refine climate models and improve hurricane/monsoon prediction."
    ))
    # Fisheries & Aquaculture
    insights.append((
        "🐟 Fisheries & Aquaculture",
        "The water conditions look broadly suitable for marine life, aiding fish stock planning and tracking seasonal movements (even when some readings are missing)."
    ))
    # Shipping & Transport
    insights.append((
        "🚢 Shipping & Transport",
        f"Surface conditions and {strat_desc} layering guide smarter routes and can help save fuel by avoiding challenging ocean zones."
    ))
    # Energy (Oil, Gas, Offshore Wind)
    insights.append((
        "⚡ Energy (Oil, Gas, Offshore Wind)",
        "The observed water-column structure and layering support safer offshore operations and site selection, serving as a reliable fallback when detailed numbers aren’t available."
    ))
    # Insurance & Risk
    insights.append((
        "🛡️ Insurance & Risk",
        "Thermal structure and surface conditions feed into storm and flood forecasting, strengthening overall risk models even with partial data."
    ))
    # Tourism & Coastal Communities
    reef_phrase = "; monitor coral reefs for heat stress" if (not np.isnan(sst) and sst >= 28) else ""
    insights.append((
        "🏖️ Tourism & Coastal Communities",
        f"Surface waters are {sst_desc}{reef_phrase}. This information supports early preparedness and coastal monitoring for the public."
    ))
    # Defense & Security
    insights.append((
        "🛥️ Defense & Security",
        f"{thermo_desc.capitalize()} can influence sonar and underwater navigation, useful for planning naval operations under different ocean states."
    ))
    # Environmental NGOs
    insights.append((
        "🌍 Environmental NGOs",
        f"Salinity appears {sal_desc}, and the vertical temperature structure helps assess ocean health and guide conservation policies, even when exact figures aren’t shown."
    ))
    return insights

def plot_contextual_temp_vs_depth(context_dfs):
    """Generates a Temperature vs. Depth plot for the given dataframes."""
    fig = go.Figure()
    for profile_df in context_dfs:
        profile_name = f"{profile_df['file_source'].iloc[0]}"
        try:
            region = get_region(profile_df['latitude'].iloc[0], profile_df['longitude'].iloc[0])
        except Exception:
            region = "Unknown"
        fig.add_trace(go.Scatter(
            x=profile_df['temperature'], y=profile_df['pressure'], mode='lines+markers',
            name=f"{profile_name} — {region}"
        ))
    fig.update_layout(title="Temperature vs. Depth for Retrieved Profiles", xaxis_title="Temperature (°C)", yaxis_title="Pressure (dbar)", yaxis=dict(autorange='reversed'), height=400)
    st.plotly_chart(fig, use_container_width=True, key="contextual_temp_chart")

def plot_contextual_salinity_vs_depth(context_dfs):
    """Generates a Salinity vs. Depth plot for the given dataframes."""
    fig = go.Figure()
    for profile_df in context_dfs:
        profile_name = f"{profile_df['file_source'].iloc[0]}"
        try:
            region = get_region(profile_df['latitude'].iloc[0], profile_df['longitude'].iloc[0])
        except Exception:
            region = "Unknown"
        fig.add_trace(go.Scatter(
            x=profile_df['salinity'], y=profile_df['pressure'], mode='lines+markers',
            name=f"{profile_name} — {region}"
        ))
    fig.update_layout(title="Salinity vs. Depth for Retrieved Profiles", xaxis_title="Salinity (PSU)", yaxis_title="Pressure (dbar)", yaxis=dict(autorange='reversed'), height=400)
    st.plotly_chart(fig, use_container_width=True, key="contextual_sal_chart")

def plot_contextual_ts_plot(context_dfs):
    """Generates a T-S plot for the given dataframes."""
    fig = go.Figure()
    for df in context_dfs:
        if not df.empty:
            profile_name = f"{df['file_source'].iloc[0]}"
            try:
                region = get_region(df['latitude'].iloc[0], df['longitude'].iloc[0])
            except Exception:
                region = "Unknown"
            fig.add_trace(go.Scatter(
                x=df['salinity'], y=df['temperature'], mode='markers',
                name=f"{profile_name} — {region}", marker=dict(size=5, opacity=0.7)
            ))
    fig.update_layout(title="Temperature-Salinity (T-S) Plot for Retrieved Profiles", xaxis_title="Salinity (PSU)", yaxis_title="Temperature (°C)", height=400)
    st.plotly_chart(fig, use_container_width=True, key="contextual_ts_chart")

def display_contextual_visualization(query, response, all_profiles_data):
    """Analyzes the query and displays the most relevant visualization."""
    st.subheader("📊 Contextual Visualization")
    query_lower = query.lower()
    
    retrieved_files = [p.get('file_source', 'N/A') for p in response.get('retrieved_profiles', [])]
    context_dfs = [df for df in all_profiles_data if not df.empty and df['file_source'].iloc[0] in retrieved_files]

    if not context_dfs:
        st.info("No specific profiles were retrieved to generate a contextual plot.")
        return

    # Logic to decide which plot to show
    if "salinity" in query_lower and "temperature" not in query_lower:
        plot_contextual_salinity_vs_depth(context_dfs)
    elif "t-s" in query_lower or "temperature-salinity" in query_lower:
        plot_contextual_ts_plot(context_dfs)
    else: # Default to temperature plot
        plot_contextual_temp_vs_depth(context_dfs)

def display_chat_interface(system_components, settings):
    """Display the AI-powered chat interface and handle query processing."""
    st.subheader("💬 AI-Powered Data Query")

    if 'query_response' not in st.session_state:
        st.session_state.query_response = None
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    
    def process_query(query_text):
        if query_text and query_text.strip():
            with st.spinner("🧠 Thinking... Searching for profiles and generating analysis..."):
                rag_system = system_components["rag_system"]
                response = rag_system.query(query_text, top_k=settings['search_results'])
                st.session_state.query_response = response
                st.session_state.last_query = query_text

    # --- Query Input Form ---
    suggestions = [
        # New samples to add
        "What dataset do you have?",
        "Salinity in the North Pacific Ocean at latitude 53.56° and longitude −140.41° in September 2020",
        "Temperature in the South Pacific Ocean at latitude −33.06° and longitude −154.69° in September 2020",
        # Existing helpful samples
        "Show me temperature profiles in the Indian Ocean",
        "Generate salinity vs depth plot",
        "Generate a T-S plot for the Indian ocean",
        "Compare temperatures between equatorial and southern regions",
        "What ARGO profiles do you have?",
        "What visualizations can you show?",
    ]

    st.markdown("💡 **Try these sample queries:**")
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 3].button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            st.session_state.current_query = suggestion
            process_query(suggestion)
    
    # Input area (container removed per request)
    query = st.text_area(
        "Enter your question about the oceanographic data:",
        value=st.session_state.get('current_query', ''),
        height=100,
        key="ai_query_textarea"
    )

    if st.button("🔍 Search", type="primary"):
        process_query(query)

    # --- Display Response Area ---
    if st.session_state.query_response:
        response = st.session_state.query_response
        if response['success']:
            st.subheader("🤖 AI Analysis")
            st.markdown(f"<div class='response-message'>{response['answer']}</div>", unsafe_allow_html=True)

            # Show MCP plan (SQL/filters) if available
            mcp_plan = response.get('mcp')
            if mcp_plan:
                with st.expander("🔧 MCP Plan (SQL & Filters)"):
                    st.markdown('<div class="mcp-plan">', unsafe_allow_html=True)
                    sql = mcp_plan.get('sql')
                    if sql:
                        st.code(sql, language='sql')
                    filters = mcp_plan.get('filters', [])
                    if filters:
                        st.json(filters)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader("📚 Retrieved Profiles")
            if response['retrieved_profiles']:
                for i, profile in enumerate(response['retrieved_profiles'], 1):
                    with st.expander(f"**Profile {i}:** {profile.get('file_source', 'N/A')} (Score: {profile['similarity_score']:.2f})", expanded=i<=2):
                        st.markdown(f"**Summary:** *{profile['summary']}*")
                        st.markdown(f"**Location:** {profile.get('latitude', 0):.2f}°, {profile.get('longitude', 0):.2f}°")
                        st.markdown(f"**Depth Range:** {profile.get('depth_range', 'N/A')}")
            else:
                st.info("The AI generated an answer without retrieving specific profiles for context.")
            
            # First show contextual visualization
            display_contextual_visualization(st.session_state.last_query, response, system_components["all_profiles_data"])

            # Then show Industry Insights below the visualization
            insights = generate_industry_insights_from_profiles(
                response.get('retrieved_profiles', []),
                system_components["all_profiles_data"]
            )
            if insights:
                st.subheader("🏭 Industry Insights (from retrieved profiles)")
                with st.expander("View industry-specific insights", expanded=True):
                    for industry, text in insights:
                        st.markdown(f"- **{industry}** → {text}")
            else:
                st.info("No industry insights could be derived from the current retrieved profiles.")
        else:
            st.error(f"❌ Query failed: {response['answer']}")

def display_data_visualization(system_components):
    """Display the main data visualization section with selectors."""
    st.subheader("📈 Interactive Visualizations")
    
    viz_type = st.selectbox(
        "Choose visualization type:",
        [
            "Temperature & Salinity vs. Depth", 
            "Regional Data Comparison", 
            "Temperature-Salinity (T-S) Plots", 
            "3D Ocean Properties", 
            "Mean Temperature Over Time",
            "Property Distribution by Region",
            "Property Range Over Time",
            "Temperature & Salinity Over Time"
        ],
        key="viz_type_selectbox"
    )

    if viz_type == "Temperature & Salinity vs. Depth":
        display_depth_profiles(system_components)
    elif viz_type == "Regional Data Comparison":
        display_regional_comparison(system_components)
    elif viz_type == "Temperature-Salinity (T-S) Plots":
        display_ts_plots(system_components)
    elif viz_type == "3D Ocean Properties":
        display_3d_plot(system_components)
    elif viz_type == "Mean Temperature Over Time":
        display_time_series_analysis(system_components)
    elif viz_type == "Property Distribution by Region":
        display_property_distribution(system_components)
    elif viz_type == "Property Range Over Time":
        display_property_range_over_time(system_components)
    elif viz_type == "Temperature & Salinity Over Time":
        display_temp_sal_over_time(system_components)


def display_overview_depth_profiles(system_components):
    """Display temperature and salinity vs. depth profiles for overview."""
    all_profiles = system_components["all_profiles_data"]
    profile_options = {f"{df['file_source'].iloc[0]} ({get_region(df['latitude'].iloc[0], df['longitude'].iloc[0])})": i 
                       for i, df in enumerate(all_profiles) if not df.empty}
    
    selected_profiles = st.multiselect("Select profiles to display:", options=list(profile_options.keys()), default=list(profile_options.keys())[:3], key="overview_depth_profiles_multiselect")
    
    if selected_profiles:
        fig = go.Figure()
        for profile_name in selected_profiles:
            idx = profile_options[profile_name]
            profile_df = all_profiles[idx]
            fig.add_trace(go.Scatter(x=profile_df['temperature'], y=profile_df['pressure'], mode='lines', name=f"Temp - {profile_name}"))
        
        fig.update_layout(title="Temperature vs. Depth", xaxis_title="Temperature (°C)", yaxis_title="Pressure (dbar)", yaxis=dict(autorange='reversed'), height=500)
        st.plotly_chart(fig, use_container_width=True, key="depth_profiles_chart")

def display_depth_profiles(system_components):
    """Display temperature and salinity vs. depth profiles."""
    all_profiles = system_components["all_profiles_data"]
    profile_options = {f"{df['file_source'].iloc[0]} ({get_region(df['latitude'].iloc[0], df['longitude'].iloc[0])})": i 
                       for i, df in enumerate(all_profiles) if not df.empty}
    
    selected_profiles = st.multiselect("Select profiles to display:", options=list(profile_options.keys()), default=list(profile_options.keys())[:3], key="depth_profiles_multiselect")
    
    if selected_profiles:
        fig = go.Figure()
        for profile_name in selected_profiles:
            idx = profile_options[profile_name]
            profile_df = all_profiles[idx]
            fig.add_trace(go.Scatter(x=profile_df['temperature'], y=profile_df['pressure'], mode='lines', name=f"Temp - {profile_name}"))
        
        fig.update_layout(title="Temperature vs. Depth", xaxis_title="Temperature (°C)", yaxis_title="Pressure (dbar)", yaxis=dict(autorange='reversed'), height=500)
        st.plotly_chart(fig, use_container_width=True, key="overview_depth_profiles_chart")

def display_overview_regional_comparison(system_components):
    """Display bar charts comparing average metrics by region for overview."""
    all_profiles_data = system_components["all_profiles_data"]
    
    regional_stats = []
    for df in all_profiles_data:
        if not df.empty:
            lat, lon = df['latitude'].iloc[0], df['longitude'].iloc[0]
            regional_stats.append({
                'Region': get_region(lat, lon),
                'Avg Temperature': df['temperature'].mean(),
                'Avg Salinity': df['salinity'].mean()
            })
    
    df_regional = pd.DataFrame(regional_stats).groupby('Region').mean().reset_index()

    st.dataframe(df_regional, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df_regional, x='Region', y='Avg Temperature', title="Average Temperature by Region")
        st.plotly_chart(fig1, use_container_width=True, key="overview_regional_temp_chart")
    with col2:
        fig2 = px.bar(df_regional, x='Region', y='Avg Salinity', title="Average Salinity by Region")
        st.plotly_chart(fig2, use_container_width=True, key="overview_regional_sal_chart")

def display_regional_comparison(system_components):
    """Display bar charts comparing average metrics by region."""
    all_profiles_data = system_components["all_profiles_data"]
    
    regional_stats = []
    for df in all_profiles_data:
        if not df.empty:
            lat, lon = df['latitude'].iloc[0], df['longitude'].iloc[0]
            regional_stats.append({
                'Region': get_region(lat, lon),
                'Avg Temperature': df['temperature'].mean(),
                'Avg Salinity': df['salinity'].mean()
            })
    
    df_regional = pd.DataFrame(regional_stats).groupby('Region').mean().reset_index()

    st.dataframe(df_regional, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df_regional, x='Region', y='Avg Temperature', title="Average Temperature by Region")
        st.plotly_chart(fig1, use_container_width=True, key="regional_temp_chart")
    with col2:
        fig2 = px.bar(df_regional, x='Region', y='Avg Salinity', title="Average Salinity by Region")
        st.plotly_chart(fig2, use_container_width=True, key="regional_sal_chart")

def display_overview_ts_plots(system_components):
    """Display Temperature-Salinity (T-S) relationship plots for overview."""
    all_profiles = system_components["all_profiles_data"]
    
    fig = go.Figure()
    for df in all_profiles:
        if not df.empty:
            region = get_region(df['latitude'].iloc[0], df['longitude'].iloc[0])
            fig.add_trace(go.Scatter(x=df['salinity'], y=df['temperature'], mode='markers', name=region,
                                     marker=dict(size=5, opacity=0.7)))
    
    fig.update_layout(title="Temperature-Salinity (T-S) Plot", xaxis_title="Salinity (PSU)", yaxis_title="Temperature (°C)", height=500)
    st.plotly_chart(fig, use_container_width=True, key="overview_ts_plot")

def display_ts_plots(system_components):
    """Display Temperature-Salinity (T-S) relationship plots."""
    all_profiles = system_components["all_profiles_data"]
    
    fig = go.Figure()
    for df in all_profiles:
        if not df.empty:
            region = get_region(df['latitude'].iloc[0], df['longitude'].iloc[0])
            fig.add_trace(go.Scatter(x=df['salinity'], y=df['temperature'], mode='markers', name=region,
                                     marker=dict(size=5, opacity=0.7)))
    
    fig.update_layout(title="Temperature-Salinity (T-S) Plot", xaxis_title="Salinity (PSU)", yaxis_title="Temperature (°C)", height=500)
    st.plotly_chart(fig, use_container_width=True, key="ts_plot")

def display_3d_plot(system_components):
    """Display a 3D scatter plot of Temp, Salinity, and Pressure."""
    all_profiles = system_components["all_profiles_data"]
    
    plot_data = []
    for df in all_profiles:
        if not df.empty:
            region = get_region(df['latitude'].iloc[0], df['longitude'].iloc[0])
            for _, row in df.iterrows():
                plot_data.append({
                    'Temperature': row['temperature'],
                    'Salinity': row['salinity'],
                    'Pressure': row['pressure'],
                    'Region': region
                })
    df_3d = pd.DataFrame(plot_data).dropna()

    fig = px.scatter_3d(df_3d, x='Salinity', y='Temperature', z='Pressure',
                        color='Temperature', color_continuous_scale='Viridis',
                        title="3D Ocean Properties (Temp-Salinity-Depth)")
    
    fig.update_layout(scene=dict(zaxis=dict(autorange="reversed")), height=600)
    st.plotly_chart(fig, use_container_width=True, key="3d_plot")

def display_time_series_analysis(system_components):
    """Display a time series plot of mean temperature."""
    metadata = system_components["embedding_index"].profile_metadata
    all_profiles = system_components["all_profiles_data"]

    if not metadata:
        st.warning("No data available for time series analysis.")
        return

    time_series_data = []
    for meta, df in zip(metadata, all_profiles):
        if not df.empty:
            time_series_data.append({
                'Date': meta['time'],
                'Mean Temperature': df['temperature'].mean(),
                'Region': get_region(meta['latitude'], meta['longitude'])
            })
    
    df_ts = pd.DataFrame(time_series_data).sort_values(by='Date')

    fig = px.line(df_ts, x='Date', y='Mean Temperature', color='Region', markers=True,
                  title="Mean Profile Temperature Over Time")
    st.plotly_chart(fig, use_container_width=True, key="time_series_chart")

def display_property_distribution(system_components):
    """Display histograms and box plots for temperature and salinity by region."""
    all_profiles = system_components["all_profiles_data"]
    
    dist_data = []
    for df in all_profiles:
        if not df.empty:
            region = get_region(df['latitude'].iloc[0], df['longitude'].iloc[0])
            for _, row in df.iterrows():
                dist_data.append({
                    'Temperature': row['temperature'],
                    'Salinity': row['salinity'],
                    'Region': region
                })
    df_dist = pd.DataFrame(dist_data).dropna()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Temperature Distribution")
        fig_temp_hist = px.histogram(df_dist, x='Temperature', color='Region', marginal='box', title="Temperature Distribution by Region")
        st.plotly_chart(fig_temp_hist, use_container_width=True, key="temp_distribution_chart")
    with col2:
        st.markdown("##### Salinity Distribution")
        fig_sal_hist = px.histogram(df_dist, x='Salinity', color='Region', marginal='box', title="Salinity Distribution by Region")
        st.plotly_chart(fig_sal_hist, use_container_width=True, key="sal_distribution_chart")

def display_property_range_over_time(system_components):
    """Display the range of temperature and salinity over time."""
    metadata = system_components["embedding_index"].profile_metadata
    all_profiles = system_components["all_profiles_data"]

    range_data = []
    for meta, df in zip(metadata, all_profiles):
        if not df.empty:
            range_data.append({
                'Date': meta['time'],
                'Min Temperature': df['temperature'].min(),
                'Max Temperature': df['temperature'].max(),
                'Min Salinity': df['salinity'].min(),
                'Max Salinity': df['salinity'].max(),
            })
    df_range = pd.DataFrame(range_data).sort_values(by='Date')

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Min Temperature'], mode='lines', name='Min Temperature', line=dict(width=0)))
    fig_temp.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Max Temperature'], mode='lines', name='Max Temperature', fill='tonexty'))
    fig_temp.update_layout(title="Temperature Range Over Time", xaxis_title="Date", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig_temp, use_container_width=True, key="temp_range_chart")
    
    fig_sal = go.Figure()
    fig_sal.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Min Salinity'], mode='lines', name='Min Salinity', line=dict(width=0)))
    fig_sal.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Max Salinity'], mode='lines', name='Max Salinity', fill='tonexty'))
    fig_sal.update_layout(title="Salinity Range Over Time", xaxis_title="Date", yaxis_title="Salinity (PSU)")
    st.plotly_chart(fig_sal, use_container_width=True, key="sal_range_chart")

def display_temp_sal_over_time(system_components):
    """Display mean temperature and salinity over time on a dual-axis chart."""
    metadata = system_components["embedding_index"].profile_metadata
    all_profiles = system_components["all_profiles_data"]

    ts_data = []
    for meta, df in zip(metadata, all_profiles):
        if not df.empty:
            ts_data.append({
                'Date': meta['time'],
                'Mean Temperature': df['temperature'].mean(),
                'Mean Salinity': df['salinity'].mean()
            })
    df_ts = pd.DataFrame(ts_data).sort_values(by='Date')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Mean Temperature'], name='Mean Temperature', yaxis='y1'))
    fig.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Mean Salinity'], name='Mean Salinity', yaxis='y2'))

    fig.update_layout(
        title="Mean Temperature and Salinity Over Time",
        xaxis_title="Date",
        yaxis=dict(title="Mean Temperature (°C)"),
        yaxis2=dict(title="Mean Salinity (PSU)", overlaying='y', side='right'),
        legend=dict(x=0, y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True, key="temp_sal_over_time_chart")

# --- Main Application ---

def _b64_image_url(file_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _get_openai_client(cfg: Config):
    if OpenAI is None:
        return None
    try:
        if not getattr(cfg, 'OPENAI_API_KEY', None):
            return None
        return OpenAI(api_key=cfg.OPENAI_API_KEY)
    except Exception:
        return None

def _transcribe_audio_bytes(cfg: Config, audio_bytes: bytes, filename: str = "audio.wav") -> str | None:
    """Transcribe audio bytes using OpenAI Whisper. Returns text or None."""
    try:
        client = _get_openai_client(cfg)
        if client is None:
            return None
        bio = io.BytesIO(audio_bytes)
        try:
            bio.name = filename  # type: ignore[attr-defined]
        except Exception:
            pass
        resp = client.audio.transcriptions.create(
            model=getattr(cfg, 'WHISPER_MODEL', 'whisper-1'),
            file=bio,
            response_format='text',
            temperature=0
        )
        return str(resp).strip() if resp else None
    except Exception:
        return None

def display_multimodal_analyzer(cfg: Config, system_components, settings):
    """Multimodal (image + text) analysis using GPT-4o-mini vision."""
    st.subheader("🖼️ Multimodal Analyzer (GPT-4o-mini)")

    client = _get_openai_client(cfg)
    if client is None:
        st.warning("OpenAI client not available. Please set OPENAI_API_KEY in .env and restart.")

    # Enhanced Voice Query section
    with st.expander("🎙️ Voice Query", expanded=False):
        st.caption("Record a short question (5–20s). We'll transcribe it with Whisper and run the same AI Query.")
        
        # Enhanced voice input with circular button
        audio_data = render_enhanced_voice_input()
        
        # Fallback if st_audiorec is not available
        if audio_data is None and st_audiorec is None:
            st.info("streamlit-audiorec not installed or failed to load. You can upload a WAV/MP3/M4A file instead.")
            uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg"], accept_multiple_files=False, key="multimodal_audio_file_uploader")
            if uploaded is not None:
                audio_data = uploaded.read()

        transcript = None
        if audio_data:
            with st.spinner("Transcribing audio with Whisper…"):
                transcript = _transcribe_audio_bytes(cfg, audio_data)
            if transcript:
                st.text_area("Transcribed Text", value=transcript, height=80, key="transcribed_text_textarea")
                if st.button("Run Voice Query", key="run_voice_query_btn"):
                    try:
                        rag_system = system_components["rag_system"]
                        response = rag_system.query(transcript, top_k=settings['search_results'])
                        if response and response.get('success', True):
                            st.markdown("### 🧠 Answer")
                            st.markdown(f"<div class='response-message'>{response.get('answer','')}</div>", unsafe_allow_html=True)
                            # Optional context viz
                            try:
                                display_contextual_visualization(transcript, response, system_components["all_profiles_data"])
                            except Exception:
                                pass
                        else:
                            st.error(f"❌ Query failed: {response.get('answer','Unknown error')}")
                    except Exception as e:
                        st.error(f"Voice query failed: {e}")
            else:
                st.warning("Could not transcribe audio. Ensure OpenAI key is set and audio is clear.")

    st.caption("Upload an image (PNG/JPEG) and choose what you want the model to do. Your image is sent to OpenAI for analysis.")

    colu1, colu2 = st.columns([2,1])
    with colu1:
        file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="multimodal_image_file_uploader")
        prompt_mode = st.radio("Choose task", [
            "Describe image",
            "Extract text (OCR)",
            "Read chart/table",
            "Custom"
        ], horizontal=True, key="prompt_mode_radio")
        custom_prompt = ""
        if prompt_mode == "Custom":
            custom_prompt = st.text_area("Custom prompt", value="Provide a concise analysis of this image. If there are numbers or labels, list them clearly.", height=80, key="custom_prompt_textarea")

    with colu2:
        quality = st.select_slider("Detail level", options=["low", "medium", "high"], value="high", key="quality_select_slider")
        include_boxes = st.checkbox("Ask for structured bullets", value=True, key="include_boxes_checkbox")
        max_tokens = st.slider("Max tokens", 128, 1000, 400, step=32, key="max_tokens_slider")

    if file is None:
        st.info("Please upload an image to proceed.")
        return

    mime = file.type or "image/png"
    img_bytes = file.read()
    st.image(img_bytes, caption=f"Preview: {file.name}", use_column_width=True)

    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing image with GPT…"):
            try:
                if prompt_mode == "Describe image":
                    base_prompt = "Describe this image thoroughly. If graphs or maps are present, summarize key trends and values."
                elif prompt_mode == "Extract text (OCR)":
                    base_prompt = "Extract all readable text from the image. Preserve line breaks where helpful and list key numbers."
                elif prompt_mode == "Read chart/table":
                    base_prompt = (
                        "This is likely a chart or table. Summarize axes, units, series, and key values. "
                        "Report notable peaks/lows and trends in clear bullets."
                    )
                else:
                    base_prompt = custom_prompt.strip() or "Provide a concise analysis of this image."

                if include_boxes:
                    base_prompt += "\nReturn results as short bullet points first, then a brief paragraph." 

                img_url = _b64_image_url(img_bytes, mime)

                # OpenAI Chat Completions with image input (data URL)
                resp = client.chat.completions.create(
                    model=getattr(cfg, 'LLM_MODEL', 'gpt-4o-mini'),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": base_prompt},
                                {"type": "image_url", "image_url": {"url": img_url, "detail": quality}},
                            ],
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                text = resp.choices[0].message.content
                st.markdown("### 📄 Result")
                st.markdown(f"<div class='response-message'>{text}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Image analysis failed: {e}")

def display_3d_earth_visualization(system_components):
    """Display live 3D earth animation with ARGO float locations."""
    st.subheader("🌍 Live 3D Earth - ARGO Float Network")
    st.caption("Interactive spinning globe showing real-time oceanographic data collection points")

    profiles_data = system_components["all_profiles_data"]
    if not profiles_data:
        st.warning("No profile data to display on the 3D earth.")
        return

    # Prepare data for 3D earth visualization
    map_df_data = []
    for df in profiles_data:
        if not df.empty:
            map_df_data.append({
                'latitude': df['latitude'].iloc[0],
                'longitude': df['longitude'].iloc[0],
                'file_source': df['file_source'].iloc[0],
                'time': df.get('time', pd.Series(['Unknown'])).iloc[0] if 'time' in df.columns else 'Unknown',
                'n_measurements': len(df),
                'mean_temperature': df['temperature'].mean() if 'temperature' in df.columns else 0,
                'mean_salinity': df['salinity'].mean() if 'salinity' in df.columns else 0
            })

    df_profiles = pd.DataFrame(map_df_data)

    if df_profiles.empty:
        st.warning("No valid profile data for 3D earth visualization.")
        return

    # Display 3D Earth if available
    try:
        from earth import display_home_page_globe
        display_home_page_globe(df_profiles)
    except Exception as e:
        st.error(f"Error displaying 3D Earth: {e}")
        # Fallback to regular map
        display_fallback_map(df_profiles)

def display_fallback_map(df_profiles):
    """Fallback 2D map if 3D earth is not available."""
    fig = px.scatter_mapbox(
        df_profiles, lat='latitude', lon='longitude',
        color='mean_temperature', size='n_measurements',
        hover_data=['file_source', 'time'],
        color_continuous_scale=px.colors.sequential.Viridis,
        mapbox_style='carto-positron', zoom=2,
        title="ARGO Float Locations (Fallback Map)"
    )
    fig.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

def display_oceanographic_graphs(system_components):
    """Display temperature, salinity, and pressure graphs below 3D earth."""
    profiles_data = system_components["all_profiles_data"]
    if not profiles_data:
        st.warning("No profile data available for oceanographic graphs.")
        return

    st.markdown("### 📊 Oceanographic Data Analysis")
    st.caption("Real-time measurements from ARGO float network")

    # Combine all profile data
    all_temp_data = []
    all_sal_data = []
    all_pressure_data = []

    for df in profiles_data:
        if not df.empty:
            # Temperature data
            temp_data = df[['pressure', 'temperature']].dropna()
            if not temp_data.empty:
                temp_data['profile_id'] = df['file_source'].iloc[0]
                all_temp_data.append(temp_data)

            # Salinity data
            sal_data = df[['pressure', 'salinity']].dropna()
            if not sal_data.empty:
                sal_data['profile_id'] = df['file_source'].iloc[0]
                all_sal_data.append(sal_data)

            # Pressure distribution
            pressure_data = df[['pressure']].dropna()
            if not pressure_data.empty:
                pressure_data['profile_id'] = df['file_source'].iloc[0]
                all_pressure_data.append(pressure_data)

    # Create three columns for the graphs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🌡️ Temperature Profiles")
        if all_temp_data:
            combined_temp = pd.concat(all_temp_data, ignore_index=True)
            fig_temp = px.line(combined_temp, x='temperature', y='pressure',
                             color='profile_id',
                             title="Temperature vs Depth",
                             labels={'temperature': 'Temperature (°C)', 'pressure': 'Pressure (dbar)'})
            fig_temp.update_layout(yaxis=dict(autorange="reversed"))  # Depth increases downward
            fig_temp.update_layout(height=400, showlegend=False,
                                 paper_bgcolor='rgba(0,0,0,0)',
                                 plot_bgcolor='rgba(0,0,0,0)',
                                 font_color='white')
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("No temperature data available")

    with col2:
        st.markdown("#### 🧂 Salinity Profiles")
        if all_sal_data:
            combined_sal = pd.concat(all_sal_data, ignore_index=True)
            fig_sal = px.line(combined_sal, x='salinity', y='pressure',
                            color='profile_id',
                            title="Salinity vs Depth",
                            labels={'salinity': 'Salinity (PSU)', 'pressure': 'Pressure (dbar)'})
            fig_sal.update_layout(yaxis=dict(autorange="reversed"))  # Depth increases downward
            fig_sal.update_layout(height=400, showlegend=False,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='white')
            st.plotly_chart(fig_sal, use_container_width=True)
        else:
            st.info("No salinity data available")

    with col3:
        st.markdown("#### 📊 Pressure Distribution")
        if all_pressure_data:
            combined_pressure = pd.concat(all_pressure_data, ignore_index=True)
            fig_pressure = px.histogram(combined_pressure, x='pressure',
                                      title="Pressure Distribution",
                                      labels={'pressure': 'Pressure (dbar)', 'count': 'Frequency'},
                                      nbins=30)
            fig_pressure.update_layout(height=400,
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)',
                                     font_color='white')
            st.plotly_chart(fig_pressure, use_container_width=True)
        else:
            st.info("No pressure data available")

def main():
    """Main dashboard function to orchestrate the UI."""
    # Add session timeout to prevent hanging
    import time
    if 'session_start' not in st.session_state:
        st.session_state.session_start = time.time()

    # Check if session has been running too long (30 minutes)
    if time.time() - st.session_state.session_start > 1800:  # 30 minutes
        st.warning("⚠️ Session timeout. Please refresh the page to continue.")
        st.stop()

    # Theme is already initialized at the top of the file
    
    # Auth gate BEFORE any heavy initialization
    cfg = Config()
    if not _auth_gate(cfg):
        return

    system_components = initialize_system()
    
    if not system_components:
        st.warning("System initialization failed. The dashboard cannot be displayed.")
        st.stop()
    
    # Render matrix header and theme toggle
    display_header()
    # Theme toggle removed
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Render floating data bubbles
    render_data_bubbles(system_components)
    
    # Get settings from sidebar (but don't display sidebar)
    settings = {'search_results': 3}  # Default value
    
    # Use keys for tabs to prevent accidental navigation on rerun
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "AI Query", "3D Globe", "Visualization", "Database", "Multimodal", "About"])
    
    with tab1:
        display_overview_metrics(system_components)
        st.markdown("---")

        # Add 3D Earth visualization above geographic map
        display_3d_earth_visualization(system_components)
        st.markdown("---")

        display_geographic_map(system_components)

        # Add main visualizations to overview
        st.markdown("---")
        st.subheader("📊 Key Data Visualizations")

        # Temperature vs Depth profiles
        st.markdown("### 🌡️ Temperature vs Depth Profiles")
        display_overview_depth_profiles(system_components)

        # Regional comparison
        st.markdown("### 🌍 Regional Data Comparison")
        display_overview_regional_comparison(system_components)

        # T-S plots
        st.markdown("### 🔬 Temperature-Salinity Analysis")
        display_overview_ts_plots(system_components)

        # Add oceanographic graphs (the 3 profile charts)
        st.markdown("---")
        display_oceanographic_graphs(system_components)
    
    with tab2:
        display_chat_interface(system_components, settings)
    
    with tab3:
        st.markdown('<div class="globe-container">', unsafe_allow_html=True)
        st.subheader("Interactive 3D Globe - ARGO Float Locations")
        
        globe_fig = create_3d_globe(system_components)
        if globe_fig:
            st.plotly_chart(globe_fig, use_container_width=True, key="3d_globe_chart")
        else:
            st.warning("No data available for 3D globe visualization.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        display_data_visualization(system_components)

    with tab5:
        display_database_explorer(system_components)

    with tab6:
        display_multimodal_analyzer(cfg, system_components, settings)

    with tab7:
        render_project_info_section()
        
        st.markdown("""
        ## About FloatChat - ARGO Data Explorer
        This interactive dashboard provides AI-powered exploration of oceanographic data 
        from ARGO floats. The system combines semantic search with data visualization 
        to make ocean data more accessible and interpretable.
        
        ### Key Technologies
        - **Backend & API:** FastAPI (for the underlying API)
        - **Frontend:** Streamlit
        - **Embeddings:** `sentence-transformers`
        - **Vector Search:** FAISS (Facebook AI Similarity Search)
        - **LLM Integration:** `langchain` with Hugging Face models
        - **Data Processing:** `xarray`, `pandas`, `netCDF4`
        """)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
