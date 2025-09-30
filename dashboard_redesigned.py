"""
FloatChat ARGO Data Explorer - Redesigned Dashboard
A modern, professional interface for oceanographic data exploration
with 3D visualizations, dual themes, and enhanced user experience.
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
import json
from typing import Dict, List, Optional, Tuple
import time

# 3D Globe imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error("Please install plotly: pip install plotly")

# Audio recording
try:
    from st_audiorec import st_audiorec
except ImportError:
    st_audiorec = None

# OpenAI for multimodal
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Backend imports
try:
    from config import Config
    from data_ingest import ARGODataIngestor
    from embedding_index import ProfileEmbeddingIndex
    from rag_engine import OceanographyRAG
    try:
        from db import get_engine, init_db
    except ImportError:
        get_engine = None
        init_db = None
    try:
        from mcp_client import MCPClient
    except ImportError:
        MCPClient = None
except ImportError as e:
    st.error(f"Failed to import backend modules: {e}")
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="FloatChat ARGO Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Modern Design
def load_custom_css():
    """Load comprehensive custom CSS for modern design"""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Reset and Base Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
    }
    
    /* CSS Variables for Theme System */
    :root {
        /* Light Theme */
        --bg-primary: #fafbfc;
        --bg-secondary: #ffffff;
        --bg-tertiary: #f8f9fa;
        --text-primary: #1a1a1a;
        --text-secondary: #6c757d;
        --text-accent: #0d6efd;
        --border-color: #e9ecef;
        --shadow-light: rgba(0, 0, 0, 0.1);
        --shadow-medium: rgba(0, 0, 0, 0.15);
        --ocean-primary: #0066cc;
        --ocean-secondary: #004499;
        --ocean-accent: #00aaff;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --error-color: #dc3545;
        --gradient-ocean: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-surface: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    [data-theme="dark"] {
        /* Dark Theme */
        --bg-primary: #0a0e1a;
        --bg-secondary: #1a1f2e;
        --bg-tertiary: #2d3748;
        --text-primary: #ffffff;
        --text-secondary: #a0aec0;
        --text-accent: #63b3ed;
        --border-color: #4a5568;
        --shadow-light: rgba(0, 0, 0, 0.3);
        --shadow-medium: rgba(0, 0, 0, 0.5);
        --ocean-primary: #3182ce;
        --ocean-secondary: #2c5282;
        --ocean-accent: #63b3ed;
        --gradient-ocean: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        --gradient-surface: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Matrix-style Header */
    .matrix-header {
        font-family: 'JetBrains Mono', monospace !important;
        background: linear-gradient(45deg, #000000, #1a1a1a, #000000);
        color: #00ff00 !important;
        padding: 2rem 0;
        text-align: center;
        border-bottom: 2px solid #00ff00;
        position: relative;
        overflow: hidden;
    }
    
    .matrix-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 0, 0.1), transparent);
        animation: matrix-scan 3s infinite;
    }
    
    @keyframes matrix-scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .matrix-text {
        font-size: 2.5rem;
        font-weight: 600;
        text-shadow: 0 0 10px #00ff00;
        letter-spacing: 0.1em;
        animation: matrix-glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes matrix-glow {
        from { text-shadow: 0 0 10px #00ff00; }
        to { text-shadow: 0 0 20px #00ff00, 0 0 30px #00ff00; }
    }
    
    /* Main Container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
        background: var(--bg-secondary);
        border-radius: 20px;
        box-shadow: 0 20px 40px var(--shadow-light);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Theme Toggle */
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background: var(--bg-secondary);
        border: 2px solid var(--border-color);
        border-radius: 50px;
        padding: 0.5rem 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px var(--shadow-light);
    }
    
    .theme-toggle:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px var(--shadow-medium);
    }
    
    /* Floating Data Bubbles */
    .data-bubbles-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding: 3rem 0;
        background: var(--bg-tertiary);
        border-radius: 20px;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .data-bubbles-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 80%, rgba(0, 102, 204, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(0, 170, 255, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .data-bubble {
        background: var(--bg-secondary);
        border: 2px solid var(--ocean-primary);
        border-radius: 50%;
        width: 150px;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        position: relative;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px var(--shadow-light);
        animation: float 6s ease-in-out infinite;
    }
    
    .data-bubble:nth-child(2) { animation-delay: -2s; }
    .data-bubble:nth-child(3) { animation-delay: -4s; }
    .data-bubble:nth-child(4) { animation-delay: -6s; }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
    }
    
    .data-bubble:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 15px 35px var(--shadow-medium);
        border-color: var(--ocean-accent);
    }
    
    .bubble-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--ocean-primary);
        margin-bottom: 0.5rem;
    }
    
    .bubble-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Navigation Tabs */
    .custom-tabs {
        background: var(--bg-secondary);
        border-radius: 15px;
        padding: 1rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px var(--shadow-light);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        border-radius: 10px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        padding: 1rem 2rem !important;
        margin: 0 0.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--gradient-ocean) !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px var(--shadow-medium) !important;
    }
    
    /* Query Interface */
    .query-container {
        background: var(--bg-secondary);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px var(--shadow-light);
        border: 1px solid var(--border-color);
    }
    
    .query-input {
        background: var(--bg-tertiary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 15px !important;
        color: var(--text-primary) !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .query-input:focus {
        border-color: var(--ocean-primary) !important;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1) !important;
    }
    
    /* Voice Recording Button */
    .voice-recorder {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }
    
    .record-button {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: var(--gradient-ocean);
        border: none;
        color: white;
        font-size: 2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px var(--shadow-medium);
        position: relative;
        overflow: hidden;
    }
    
    .record-button:hover {
        transform: scale(1.1);
        box-shadow: 0 15px 40px var(--shadow-medium);
    }
    
    .record-button.recording {
        background: var(--error-color);
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    /* Response Cards */
    .response-card {
        background: var(--bg-secondary);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 4px solid var(--ocean-primary);
        box-shadow: 0 8px 25px var(--shadow-light);
        transition: all 0.3s ease;
    }
    
    .response-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px var(--shadow-medium);
    }
    
    /* 3D Globe Container */
    .globe-container {
        background: var(--bg-secondary);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 15px 40px var(--shadow-light);
        position: relative;
        overflow: hidden;
    }
    
    .globe-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at center, rgba(0, 102, 204, 0.05) 0%, transparent 70%);
        pointer-events: none;
    }
    
    /* Visualization Cards */
    .viz-card {
        background: var(--bg-secondary);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px var(--shadow-light);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .viz-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px var(--shadow-medium);
    }
    
    /* Project Info Section */
    .project-info {
        background: var(--bg-secondary);
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem 0;
        box-shadow: 0 15px 40px var(--shadow-light);
        position: relative;
        overflow: hidden;
    }
    
    .project-info::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(0, 102, 204, 0.1) 0%, rgba(0, 170, 255, 0.1) 100%);
        pointer-events: none;
    }
    
    /* Video Placeholder */
    .video-placeholder {
        background: var(--bg-tertiary);
        border: 2px dashed var(--border-color);
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .video-placeholder:hover {
        border-color: var(--ocean-primary);
        background: rgba(0, 102, 204, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--gradient-ocean) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px var(--shadow-light) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px var(--shadow-medium) !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu, footer, header, .stDeployButton {
        visibility: hidden !important;
        display: none !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-container {
            padding: 1rem;
            margin: 1rem;
        }
        
        .data-bubbles-container {
            flex-direction: column;
            gap: 1rem;
        }
        
        .data-bubble {
            width: 120px;
            height: 120px;
        }
        
        .matrix-text {
            font-size: 1.8rem;
        }
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid var(--border-color);
        border-radius: 50%;
        border-top-color: var(--ocean-primary);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Success/Error Messages */
    .success-message {
        background: rgba(40, 167, 69, 0.1);
        border: 1px solid var(--success-color);
        border-radius: 10px;
        padding: 1rem;
        color: var(--success-color);
        margin: 1rem 0;
    }
    
    .error-message {
        background: rgba(220, 53, 69, 0.1);
        border: 1px solid var(--error-color);
        border-radius: 10px;
        padding: 1rem;
        color: var(--error-color);
        margin: 1rem 0;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-tertiary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--ocean-primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--ocean-secondary);
    }
    </style>
    """

# Theme Management
def initialize_theme():
    """Initialize theme system"""
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    
    # Apply theme to body
    theme_script = f"""
    <script>
    document.documentElement.setAttribute('data-theme', '{st.session_state.theme}');
    </script>
    """
    st.markdown(theme_script, unsafe_allow_html=True)

def toggle_theme():
    """Toggle between light and dark themes"""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    st.rerun()

# Backend Initialization (unchanged)
@st.cache_resource
def initialize_system():
    """Initialize and cache the full RAG pipeline"""
    try:
        with st.spinner("🚀 Initializing FloatChat Engine..."):
            config = Config()
            config.validate_config()

            ingestor = ARGODataIngestor(config)
            profiles = ingestor.process_all_files()
            
            if not profiles:
                st.error(f"⚠️ No ARGO data files found in: `{config.INDIAN_OCEAN_PATH}`")
                return None

            embedding_index = ProfileEmbeddingIndex(config)
            embedding_index.create_embeddings(profiles)
            
            try:
                stats_now = embedding_index.get_statistics()
                if stats_now.get('total_profiles', 0) != len(profiles):
                    embedding_index.create_embeddings(profiles, force_rebuild=True)
            except Exception:
                pass

            mcp_client = None
            try:
                if getattr(config, "MCP_ENABLED", False) and MCPClient is not None:
                    mcp_client = MCPClient(getattr(config, "MCP_SERVER_URL", None))
            except Exception:
                mcp_client = None

            rag_system = OceanographyRAG(embedding_index, config, mcp_client=mcp_client)
            rag_system.initialize_llm()
            
            if rag_system.llm is None:
                st.warning("⚠️ LLM not initialized. Using rule-based responses.")

            return {
                "config": config,
                "ingestor": ingestor,
                "embedding_index": embedding_index,
                "rag_system": rag_system,
                "all_profiles_data": profiles
            }
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        return None

# Helper Functions (unchanged)
def get_region(latitude, longitude):
    """Classify ocean basin based on lat/lon"""
    try:
        lat = float(latitude)
        lon = float(longitude)
    except Exception:
        return "Unknown"

    if lon > 180:
        lon = ((lon + 180) % 360) - 180
    if lon < -180:
        lon = ((lon - 180) % 360) + 180

    if lat >= 66.0:
        return "Arctic Ocean"
    if lat <= -50.0:
        return "Southern Ocean"

    if -70.0 <= lon <= 20.0:
        basin = "Atlantic Ocean"
    elif 20.0 < lon < 146.0:
        basin = "Indian Ocean"
    else:
        basin = "Pacific Ocean"

    if basin in ("Atlantic Ocean", "Pacific Ocean"):
        hemi = "North" if lat >= 0 else "South"
        return f"{hemi} {basin}"
    else:
        return basin

# UI Components
def render_matrix_header():
    """Render the matrix-style header"""
    st.markdown("""
    <div class="matrix-header">
        <div class="matrix-text">FLOATCHAT ARGO DATA EXPLORER</div>
        <div style="font-size: 1rem; margin-top: 1rem; opacity: 0.8;">
            ADVANCED OCEANOGRAPHIC DATA ANALYSIS SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_theme_toggle():
    """Render theme toggle button"""
    theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
    if st.button(theme_icon, key="theme_toggle", help="Toggle theme"):
        toggle_theme()

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
    <div class="data-bubbles-container">
        <div class="data-bubble">
            <div class="bubble-value">{}</div>
            <div class="bubble-label">Total Profiles</div>
        </div>
        <div class="data-bubble">
            <div class="bubble-value">{}</div>
            <div class="bubble-label">Ocean Regions</div>
        </div>
        <div class="data-bubble">
            <div class="bubble-value">{} days</div>
            <div class="bubble-label">Time Span</div>
        </div>
        <div class="data-bubble">
            <div class="bubble-value">{:.1f}°C</div>
            <div class="bubble-label">Avg Temperature</div>
        </div>
    </div>
    """.format(total_profiles, unique_regions, time_span_days, avg_temp), unsafe_allow_html=True)

def create_3d_globe(system_components):
    """Create interactive 3D globe with ARGO float locations"""
    profiles_data = system_components["all_profiles_data"]
    if not profiles_data:
        return None
    
    # Prepare data for globe
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
    
    # Create 3D globe
    fig = go.Figure()
    
    # Add globe surface
    fig.add_trace(go.Scattergeo(
        mode='markers',
        lon=df_globe['lon'],
        lat=df_globe['lat'],
        marker=dict(
            size=8,
            color=df_globe['temp'],
            colorscale='Viridis',
            colorbar=dict(title="Temperature (°C)"),
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        text=df_globe.apply(lambda x: f"Region: {x['region']}<br>Temp: {x['temp']:.2f}°C<br>Salinity: {x['salinity']:.2f} PSU<br>Max Depth: {x['depth']:.0f} dbar", axis=1),
        hovertemplate='%{text}<extra></extra>',
        name='ARGO Floats'
    ))
    
    # Add trajectory lines (simplified)
    for region in df_globe['region'].unique():
        region_data = df_globe[df_globe['region'] == region]
        if len(region_data) > 1:
            fig.add_trace(go.Scattergeo(
                mode='lines',
                lon=region_data['lon'],
                lat=region_data['lat'],
                line=dict(width=2, color='rgba(0, 102, 204, 0.3)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title="ARGO Float Locations - 3D Globe View",
        geo=dict(
            projection_type='orthographic',
            showocean=True,
            oceancolor='rgba(0, 102, 204, 0.3)',
            showland=True,
            landcolor='rgba(200, 200, 200, 0.8)',
            showlakes=True,
            lakecolor='rgba(0, 102, 204, 0.5)',
            showrivers=True,
            rivercolor='rgba(0, 102, 204, 0.3)',
            showframe=False,
            bgcolor='rgba(0,0,0,0)'
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
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
    
    # Voice recording logic
    if st_audiorec is not None:
        audio_data = st_audiorec()
        if audio_data:
            return audio_data
    return None

def render_project_info_section():
    """Render ARGO Project information section"""
    st.markdown("""
    <div class="project-info">
        <h2 style="color: var(--ocean-primary); margin-bottom: 2rem; text-align: center;">
            About The ARGO Project
        </h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: center;">
            <div>
                <h3 style="color: var(--text-primary); margin-bottom: 1rem;">Global Ocean Observing Network</h3>
                <p style="color: var(--text-secondary); line-height: 1.6; margin-bottom: 1rem;">
                    The ARGO program is an international collaboration that maintains a global array of autonomous 
                    profiling floats to monitor the temperature, salinity, and velocity of the upper ocean.
                </p>
                <p style="color: var(--text-secondary); line-height: 1.6; margin-bottom: 1rem;">
                    With over 3,900 active floats worldwide, ARGO provides real-time data that is essential for 
                    climate research, weather prediction, and ocean monitoring.
                </p>
                <div style="display: flex; gap: 1rem; margin-top: 2rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: var(--ocean-primary);">3,900+</div>
                        <div style="color: var(--text-secondary);">Active Floats</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: var(--ocean-primary);">2000+</div>
                        <div style="color: var(--text-secondary);">Profiles/Day</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: var(--ocean-primary);">30+</div>
                        <div style="color: var(--text-secondary);">Countries</div>
                    </div>
                </div>
            </div>
            <div class="video-placeholder">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📹</div>
                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">Project Showcase Video</h4>
                <p style="color: var(--text-secondary);">
                    YouTube video embedding will be added here to showcase our project film.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Application
def main():
    """Main application function"""
    # Load CSS and initialize theme
    st.markdown(load_custom_css(), unsafe_allow_html=True)
    initialize_theme()
    
    # Render header and theme toggle
    render_matrix_header()
    render_theme_toggle()
    
    # Initialize system
    system_components = initialize_system()
    if not system_components:
        st.error("System initialization failed.")
        st.stop()
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Data bubbles
    render_data_bubbles(system_components)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 AI Query", "🌍 3D Globe", "📊 Visualizations", "🗄️ Database", "ℹ️ About"
    ])
    
    with tab1:
        st.markdown('<div class="query-container">', unsafe_allow_html=True)
        st.subheader("AI-Powered Ocean Data Query")
        
        # Query input
        query = st.text_area(
            "Ask questions about ARGO oceanographic data:",
            height=100,
            key="query_input",
            help="Enter your question about temperature, salinity, ocean regions, or any oceanographic data"
        )
        
        # Voice input
        st.markdown("### Voice Input")
        audio_data = render_enhanced_voice_input()
        
        if st.button("🔍 Search", type="primary"):
            if query.strip():
                with st.spinner("Analyzing ocean data..."):
                    rag_system = system_components["rag_system"]
                    response = rag_system.query(query, top_k=3)
                    
                    if response['success']:
                        st.markdown(f"""
                        <div class="response-card">
                            <h4 style="color: var(--ocean-primary); margin-bottom: 1rem;">AI Analysis</h4>
                            <p style="color: var(--text-primary); line-height: 1.6;">{response['answer']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show retrieved profiles
                        if response['retrieved_profiles']:
                            st.subheader("Retrieved Profiles")
                            for i, profile in enumerate(response['retrieved_profiles'], 1):
                                with st.expander(f"Profile {i}: {profile.get('file_source', 'N/A')} (Score: {profile['similarity_score']:.2f})"):
                                    st.write(f"**Summary:** {profile['summary']}")
                                    st.write(f"**Location:** {profile.get('latitude', 0):.2f}°, {profile.get('longitude', 0):.2f}°")
                                    st.write(f"**Region:** {get_region(profile.get('latitude', 0), profile.get('longitude', 0))}")
                    else:
                        st.error(f"Query failed: {response['answer']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="globe-container">', unsafe_allow_html=True)
        st.subheader("Interactive 3D Globe - ARGO Float Locations")
        
        globe_fig = create_3d_globe(system_components)
        if globe_fig:
            st.plotly_chart(globe_fig, use_container_width=True)
        else:
            st.warning("No data available for 3D globe visualization.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.subheader("Advanced Data Visualizations")
        
        # Visualization options
        viz_type = st.selectbox(
            "Select visualization type:",
            [
                "Temperature vs Depth Profiles",
                "Salinity vs Depth Profiles", 
                "Temperature-Salinity Plots",
                "Regional Comparisons",
                "Time Series Analysis"
            ]
        )
        
        # Render selected visualization
        all_profiles = system_components["all_profiles_data"]
        
        if viz_type == "Temperature vs Depth Profiles":
            fig = go.Figure()
            for i, df in enumerate(all_profiles[:5]):  # Show first 5 profiles
                if not df.empty:
                    region = get_region(df['latitude'].iloc[0], df['longitude'].iloc[0])
                    fig.add_trace(go.Scatter(
                        x=df['temperature'], 
                        y=df['pressure'], 
                        mode='lines+markers',
                        name=f"{df['file_source'].iloc[0]} - {region}",
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title="Temperature vs Depth Profiles",
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (dbar)",
                yaxis=dict(autorange='reversed'),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Temperature-Salinity Plots":
            fig = go.Figure()
            for df in all_profiles:
                if not df.empty:
                    region = get_region(df['latitude'].iloc[0], df['longitude'].iloc[0])
                    fig.add_trace(go.Scatter(
                        x=df['salinity'], 
                        y=df['temperature'], 
                        mode='markers',
                        name=region,
                        marker=dict(size=6, opacity=0.7)
                    ))
            
            fig.update_layout(
                title="Temperature-Salinity (T-S) Plot",
                xaxis_title="Salinity (PSU)",
                yaxis_title="Temperature (°C)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.subheader("Database Explorer")
        
        # Database connection and queries (simplified version)
        cfg = system_components["config"]
        if getattr(cfg, 'POSTGRES_ENABLED', False):
            st.info("PostgreSQL integration available. Database queries can be implemented here.")
        else:
            st.info("PostgreSQL is disabled. Enable in configuration to access database features.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        render_project_info_section()
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
