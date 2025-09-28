"""
FastAPI endpoints for the FloatChat ARGO oceanographic data assistant.
Provides REST API for querying, visualization, and data exploration.
"""

import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json

from config import Config
from data_ingest import ARGODataIngestor
from embedding_index import ProfileEmbeddingIndex
from rag_engine import OceanographyRAG, QueryAnalyzer
from utils import setup_logging
from typing import Any
try:
    # Optional import; only used when POSTGRES is enabled or endpoint is called
    from db import save_profiles_to_postgres
except Exception:
    save_profiles_to_postgres = None  # type: ignore

# Initialize logging
logger = setup_logging(__name__)

# Initialize configuration
config = Config()
config.validate_config()

# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="RAG-powered assistant for exploring ARGO oceanographic float data"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global components (initialized on startup)
ingestor: Optional[ARGODataIngestor] = None
embedding_index: Optional[ProfileEmbeddingIndex] = None
rag_system: Optional[OceanographyRAG] = None

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about oceanographic data")
    top_k: Optional[int] = Field(3, description="Number of similar profiles to retrieve", ge=1, le=10)
    include_metadata: bool = Field(True, description="Include detailed profile metadata")

class QueryResponse(BaseModel):
    answer: str
    retrieved_profiles: List[Dict]
    query: str
    context_profiles_count: int
    success: bool
    suggestions: Optional[List[str]] = None

class PlotRequest(BaseModel):
    question: str = Field(..., description="Query to find relevant profile for plotting")
    plot_type: str = Field("temperature", description="Type of plot: temperature, salinity, or both")

class DataSummary(BaseModel):
    total_profiles: int
    date_range: str
    geographic_coverage: Dict[str, float]
    available_parameters: List[str]

class SystemStatus(BaseModel):
    status: str
    components: Dict[str, bool]
    data_loaded: bool
    total_profiles: int
    embedding_model: str
    vector_backend: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup."""
    global ingestor, embedding_index, rag_system
    
    logger.info("🚀 Starting FloatChat API...")
    
    try:
        # Initialize data ingestor
        ingestor = ARGODataIngestor(config)
        logger.info("✅ Data ingestor initialized")
        
        # Initialize embedding index
        embedding_index = ProfileEmbeddingIndex(config)
        logger.info("✅ Embedding index initialized")
        
        # Load existing data if available
        profiles = ingestor.process_all_files()
        if profiles:
            logger.info(f"📊 Loading {len(profiles)} existing profiles...")
            embedding_index.create_embeddings(profiles)
            
            # Initialize RAG system
            rag_system = OceanographyRAG(embedding_index, config)
            logger.info("✅ RAG system initialized")
        else:
            logger.warning("⚠️  No profile data found. API will run in limited mode.")
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Continue startup but mark components as failed

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """API root with basic information and links."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{config.API_TITLE}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ color: #2E86AB; }}
            .endpoint {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .status {{ color: #28a745; }}
        </style>
    </head>
    <body>
        <h1 class="header">🌊 {config.API_TITLE}</h1>
        <p>RAG-powered assistant for exploring ARGO oceanographic float data</p>
        
        <h2>Available Endpoints:</h2>
        <div class="endpoint"><strong>GET /status</strong> - System status and health check</div>
        <div class="endpoint"><strong>POST /chat</strong> - Ask questions about oceanographic data</div>
        <div class="endpoint"><strong>POST /plot</strong> - Generate profile visualizations</div>
        <div class="endpoint"><strong>GET /data/summary</strong> - Data collection summary</div>
        <div class="endpoint"><strong>GET /docs</strong> - Interactive API documentation</div>
        
        <p class="status">🟢 API is running</p>
        
        <h3>Example Query:</h3>
        <pre>
curl -X POST "http://localhost:8000/chat" \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "Show me temperature profiles near the equator"}}'
        </pre>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status and component health."""
    status = "healthy"
    components = {
        "data_ingestor": ingestor is not None,
        "embedding_index": embedding_index is not None,
        "rag_system": rag_system is not None,
    }
    
    if not all(components.values()):
        status = "degraded"
    
    total_profiles = 0
    if embedding_index:
        stats = embedding_index.get_statistics()
        total_profiles = stats.get('total_profiles', 0)
    
    return SystemStatus(
        status=status,
        components=components,
        data_loaded=total_profiles > 0,
        total_profiles=total_profiles,
        embedding_model=config.EMBEDDING_MODEL,
        vector_backend=config.VECTOR_BACKEND,
    )

# Main chat endpoint
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """Process natural language queries about oceanographic data."""
    if not rag_system:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not initialized. Please check system status."
        )
    
    try:
        logger.info(f"Processing query: {request.question}")
        
        # Process query through RAG system
        response = rag_system.query(
            request.question,
            top_k=request.top_k,
            include_metadata=request.include_metadata
        )
        
        # Generate suggestions
        suggestions = rag_system.suggest_related_queries(request.question)
        response['suggestions'] = suggestions
        
        return QueryResponse(**response)
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# Profile plotting endpoint
@app.post("/plot")
async def plot_profile(request: PlotRequest):
    """Generate oceanographic profile plots based on query."""
    if not rag_system or not ingestor:
        raise HTTPException(
            status_code=503,
            detail="System components not initialized"
        )
    
    try:
        # Find relevant profile
        search_results = embedding_index.search_similar_profiles(request.question, top_k=1)
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No matching profiles found")
        
        # Get profile metadata
        profile_metadata = search_results[0]['metadata']
        file_source = profile_metadata['file_source']
        
        # Load the actual profile data
        file_path = os.path.join(config.INDIAN_OCEAN_PATH, file_source)
        df = ingestor.parse_netcdf_profile(file_path)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="Profile data not available")
        
        # Create plot based on request type
        if request.plot_type == "temperature":
            fig = px.line(
                df, x="temperature", y="pressure",
                title=f"Temperature Profile - {profile_metadata.get('file_source', 'Unknown')}",
                labels={"temperature": "Temperature (°C)", "pressure": "Pressure (dbar)"}
            )
        elif request.plot_type == "salinity":
            fig = px.line(
                df, x="salinity", y="pressure",
                title=f"Salinity Profile - {profile_metadata.get('file_source', 'Unknown')}",
                labels={"salinity": "Salinity (PSU)", "pressure": "Pressure (dbar)"}
            )
        elif request.plot_type == "both":
            # Create subplot with both temperature and salinity
            fig = go.Figure()
            
            # Add temperature trace
            fig.add_trace(go.Scatter(
                x=df['temperature'], y=df['pressure'],
                mode='lines+markers',
                name='Temperature (°C)',
                line=dict(color='red')
            ))
            
            # Add salinity trace on secondary x-axis
            fig.add_trace(go.Scatter(
                x=df['salinity'], y=df['pressure'],
                mode='lines+markers',
                name='Salinity (PSU)',
                line=dict(color='blue'),
                xaxis='x2'
            ))
            
            # Update layout for dual x-axis
            fig.update_layout(
                title=f"Temperature & Salinity Profile - {profile_metadata.get('file_source', 'Unknown')}",
                xaxis=dict(title="Temperature (°C)", side="bottom"),
                xaxis2=dict(title="Salinity (PSU)", side="top", overlaying="x"),
                yaxis=dict(title="Pressure (dbar)", autorange="reversed")
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid plot_type. Use 'temperature', 'salinity', or 'both'")
        
        # Reverse y-axis (depth increases downward)
        if request.plot_type != "both":
            fig.update_yaxes(autorange="reversed")
        
        # Convert to HTML
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Also provide JSON for programmatic use
        plot_json = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return {
            "plot_html": plot_html,
            "plot_json": plot_json,
            "profile_metadata": profile_metadata,
            "plot_type": request.plot_type,
            "data_points": len(df)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {str(e)}")

# Data summary endpoint
@app.get("/data/summary", response_model=DataSummary)
async def get_data_summary():
    """Get summary statistics of the loaded oceanographic data."""
    if not embedding_index:
        raise HTTPException(status_code=503, detail="Embedding index not initialized")
    
    stats = embedding_index.get_statistics()
    
    if stats['total_profiles'] == 0:
        return DataSummary(
            total_profiles=0,
            date_range="No data loaded",
            geographic_coverage={"lat_min": 0, "lat_max": 0, "lon_min": 0, "lon_max": 0},
            available_parameters=[]
        )
    
    # Extract geographic and temporal info
    metadata_list = embedding_index.profile_metadata
    latitudes = [m['latitude'] for m in metadata_list if m['latitude']]
    longitudes = [m['longitude'] for m in metadata_list if m['longitude']]
    times = [m['time'] for m in metadata_list if m['time']]
    
    geographic_coverage = {
        "lat_min": min(latitudes) if latitudes else 0,
        "lat_max": max(latitudes) if latitudes else 0,
        "lon_min": min(longitudes) if longitudes else 0,
        "lon_max": max(longitudes) if longitudes else 0
    }
    
    date_range = "Unknown"
    if times:
        min_date = min(times).strftime("%Y-%m-%d")
        max_date = max(times).strftime("%Y-%m-%d")
        date_range = f"{min_date} to {max_date}"
    
    return DataSummary(
        total_profiles=stats['total_profiles'],
        date_range=date_range,
        geographic_coverage=geographic_coverage,
        available_parameters=["temperature", "salinity", "pressure", "latitude", "longitude", "time"]
    )

# Data management endpoints
@app.post("/data/refresh")
async def refresh_data(background_tasks: BackgroundTasks):
    """Refresh data by downloading new files and rebuilding index."""
    if not ingestor or not embedding_index:
        raise HTTPException(status_code=503, detail="System components not initialized")
    
    def refresh_task():
        try:
            logger.info("Starting data refresh...")
            
            # Download new files
            downloaded = ingestor.download_netcdf_files(max_files=10)
            logger.info(f"Downloaded {len(downloaded)} files")
            
            # Reprocess all files
            profiles = ingestor.process_all_files()
            
            if profiles:
                # Rebuild embeddings
                embedding_index.create_embeddings(profiles, force_rebuild=True)
                logger.info(f"Rebuilt index with {len(profiles)} profiles")
            
        except Exception as e:
            logger.error(f"Data refresh failed: {e}")
    
    background_tasks.add_task(refresh_task)
    
    return {"message": "Data refresh started in background", "status": "processing"}

# Save parsed profiles and measurements to PostgreSQL
@app.post("/data/save_postgres")
async def save_to_postgres():
    """Persist parsed ARGO profiles (and level measurements) into PostgreSQL."""
    if not ingestor or not embedding_index:
        raise HTTPException(status_code=503, detail="System components not initialized")

    if not config.POSTGRES_ENABLED:
        raise HTTPException(status_code=400, detail="POSTGRES_ENABLED is False in config/env")

    if save_profiles_to_postgres is None:
        raise HTTPException(status_code=500, detail="Database module not available")

    try:
        # Rebuild current profile DataFrames from files
        profiles = ingestor.process_all_files()
        if not profiles:
            raise HTTPException(status_code=404, detail="No profile data available to save")

        if not embedding_index.profile_metadata or not embedding_index.profile_summaries:
            # Ensure embeddings/metadata exist to align with profiles
            embedding_index.create_embeddings(profiles)

        saved_count = save_profiles_to_postgres(
            profiles=profiles,
            summaries=embedding_index.profile_summaries,
            metadata_list=embedding_index.profile_metadata,
            cfg=config,
        )
        return {"saved_profiles": saved_count, "database": "postgres"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Saving to PostgreSQL failed: {e}")
        raise HTTPException(status_code=500, detail=f"Save to PostgreSQL failed: {str(e)}")

# Export current vectors to Chroma persistent collection
@app.post("/vectors/export_chroma")
async def export_vectors_to_chroma():
    """Export embeddings, documents, and metadata to a persistent Chroma collection."""
    if not embedding_index:
        raise HTTPException(status_code=503, detail="Embedding index not initialized")

    try:
        result = embedding_index.export_to_chroma()
        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"Chroma export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chroma export failed: {str(e)}")

# Search endpoint for direct vector search
@app.get("/search")
async def search_profiles(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results", ge=1, le=20)
):
    """Direct vector search without LLM generation."""
    if not embedding_index:
        raise HTTPException(status_code=503, detail="Embedding index not initialized")
    
    try:
        results = embedding_index.search_similar_profiles(q, top_k=top_k)
        
        return {
            "query": q,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "available_endpoints": [
            "/", "/status", "/chat", "/plot", "/data/summary", "/search", "/health", "/docs"
        ]}
    )

# Development helper - run with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting FloatChat API on {config.API_HOST}:{config.API_PORT}")
    
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )