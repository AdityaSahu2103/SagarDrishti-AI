"""
Vector embedding and FAISS indexing module for ARGO profile similarity search.
Creates semantic embeddings of oceanographic profiles for RAG retrieval.
"""

import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
from config import Config
from data_ingest import create_profile_summary
from utils import setup_logging, timing_decorator

logger = setup_logging(__name__)

class ProfileEmbeddingIndex:
    def __init__(self, config: Config = Config):
        self.config = config
        self.model = None
        self.index = None
        self.profile_metadata = []
        self.profile_summaries = []
        self.embeddings: Optional[np.ndarray] = None
        
    def initialize_model(self):
        """Initialize the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
    
    @timing_decorator
    def create_embeddings(self, profiles: List[pd.DataFrame], 
                         force_rebuild: bool = False) -> None:
        """
        Create embeddings for a list of profile DataFrames.
        
        Args:
            profiles: List of ARGO profile DataFrames
            force_rebuild: Whether to rebuild even if cache exists
        """
        cache_path = os.path.join(self.config.DATA_ROOT, "embeddings_cache.pkl")
        
        # Check if cached embeddings exist
        if not force_rebuild and os.path.exists(cache_path):
            logger.info("Loading cached embeddings...")
            self._load_cache(cache_path)
            return
        
        self.initialize_model()
        
        logger.info(f"Creating embeddings for {len(profiles)} profiles...")
        
        # Generate summaries and metadata
        self.profile_summaries = []
        self.profile_metadata = []
        
        for i, df in enumerate(profiles):
            if df.empty:
                continue
                
            # Create human-readable summary
            summary = create_profile_summary(df)
            self.profile_summaries.append(summary)
            
            # Store metadata
            metadata = {
                'profile_idx': i,
                'latitude': df['latitude'].iloc[0],
                'longitude': df['longitude'].iloc[0],
                'time': df['time'].iloc[0],
                'file_source': df['file_source'].iloc[0],
                'n_measurements': len(df),
                'depth_range': f"{df['pressure'].min():.1f}-{df['pressure'].max():.1f}m",
                'temp_range': f"{df['temperature'].min():.1f}-{df['temperature'].max():.1f}°C"
            }
            self.profile_metadata.append(metadata)
        
        # Generate embeddings
        logger.info("Encoding summaries to embeddings...")
        embeddings = self.model.encode(
            self.profile_summaries, 
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        self.embeddings = embeddings
        
        # Create FAISS index
        self._build_faiss_index(embeddings)
        
        # Cache results
        self._save_cache(cache_path, embeddings)
        logger.info(f"Created and cached {len(embeddings)} embeddings")
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        
        # Use L2 distance for semantic similarity
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, {dimension}D")
    
    def _save_cache(self, cache_path: str, embeddings: np.ndarray):
        """Save embeddings and metadata to cache file."""
        cache_data = {
            'embeddings': embeddings,
            'summaries': self.profile_summaries,
            'metadata': self.profile_metadata,
            'model_name': self.config.EMBEDDING_MODEL
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Embeddings cached to {cache_path}")
    
    def _load_cache(self, cache_path: str):
        """Load embeddings and metadata from cache file."""
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify model compatibility
            if cache_data.get('model_name') != self.config.EMBEDDING_MODEL:
                logger.warning("Cached embeddings use different model, rebuilding...")
                return False
            
            # Load data
            embeddings = cache_data['embeddings']
            self.profile_summaries = cache_data['summaries']
            self.profile_metadata = cache_data['metadata']
            self.embeddings = embeddings
            
            # Rebuild FAISS index
            self._build_faiss_index(embeddings)
            
            logger.info(f"Loaded {len(self.profile_summaries)} cached embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False
    
    def search_similar_profiles(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for profiles similar to the given query.
        
        Args:
            query: Natural language search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing profile metadata and similarity scores
        """
        if self.model is None:
            self.initialize_model()
        
        if self.index is None:
            raise ValueError("No index available. Call create_embeddings() first.")
        
        top_k = top_k or self.config.DEFAULT_TOP_K
        
        # Encode query
        logger.info(f"Searching for: '{query}'")
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), top_k
        )
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.profile_metadata):  # Bounds check
                result = {
                    'similarity_score': float(1.0 / (1.0 + distance)),  # Convert distance to similarity
                    'distance': float(distance),
                    'summary': self.profile_summaries[idx],
                    'metadata': self.profile_metadata[idx]
                }
                results.append(result)
        
        logger.info(f"Found {len(results)} similar profiles")
        return results
    
    def get_profile_by_index(self, index: int) -> Optional[Dict]:
        """Get profile data by index."""
        if 0 <= index < len(self.profile_metadata):
            return {
                'summary': self.profile_summaries[index],
                'metadata': self.profile_metadata[index]
            }
        return None
    
    def get_statistics(self) -> Dict:
        """Get index statistics."""
        if not self.profile_metadata:
            return {'total_profiles': 0}
        
        latitudes = [m['latitude'] for m in self.profile_metadata]
        longitudes = [m['longitude'] for m in self.profile_metadata]
        
        return {
            'total_profiles': len(self.profile_metadata),
            'latitude_range': f"{min(latitudes):.2f} to {max(latitudes):.2f}",
            'longitude_range': f"{min(longitudes):.2f} to {max(longitudes):.2f}",
            'embedding_model': self.config.EMBEDDING_MODEL,
            'index_type': 'FAISS IndexFlatL2'
        }

    def export_to_chroma(self) -> Dict:
        """Export current embeddings, documents, and metadata to a Chroma persistent collection."""
        if self.embeddings is None or self.index is None:
            raise ValueError("No embeddings available. Call create_embeddings() first.")

        if self.config.VECTOR_BACKEND != 'chroma':
            logger.warning("VECTOR_BACKEND is not 'chroma'; proceeding to write to Chroma anyway.")

        try:
            # Lazy import chromadb
            import chromadb
            from chromadb.utils import embedding_functions

            os.makedirs(self.config.CHROMA_PERSIST_DIR, exist_ok=True)
            client = chromadb.PersistentClient(path=self.config.CHROMA_PERSIST_DIR)

            # Create or get collection
            coll = client.get_or_create_collection(name=self.config.CHROMA_COLLECTION)

            # Prepare payloads
            ids = []
            docs = []
            metas = []
            embs = []
            for i, (summary, meta) in enumerate(zip(self.profile_summaries, self.profile_metadata)):
                ids.append(f"profile-{i}")
                docs.append(summary)
                m = dict(meta)
                # Ensure JSON-serializable metadata
                t = m.get('time')
                if hasattr(t, 'isoformat'):
                    m['time'] = t.isoformat()
                metas.append(m)
                embs.append(self.embeddings[i].astype(float).tolist())

            # Clear existing ids (optional): Chroma upserts by id; keep it simple and upsert
            coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

            logger.info(f"Exported {len(ids)} vectors to Chroma collection '{self.config.CHROMA_COLLECTION}' at {self.config.CHROMA_PERSIST_DIR}")
            return {"exported": len(ids), "collection": self.config.CHROMA_COLLECTION, "path": self.config.CHROMA_PERSIST_DIR}
        except Exception as e:
            logger.error(f"Failed to export to Chroma: {e}")
            raise

# Utility functions
def search_profiles_by_criteria(embedding_index: ProfileEmbeddingIndex, 
                               latitude_range: Tuple[float, float] = None,
                               longitude_range: Tuple[float, float] = None,
                               month: int = None) -> List[Dict]:
    """
    Filter profiles by geographic/temporal criteria.
    
    Args:
        embedding_index: ProfileEmbeddingIndex instance
        latitude_range: (min_lat, max_lat) tuple
        longitude_range: (min_lon, max_lon) tuple  
        month: Target month (1-12)
        
    Returns:
        List of matching profile metadata
    """
    results = []
    
    for i, metadata in enumerate(embedding_index.profile_metadata):
        match = True
        
        if latitude_range:
            lat = metadata['latitude']
            if not (latitude_range[0] <= lat <= latitude_range[1]):
                match = False
        
        if longitude_range and match:
            lon = metadata['longitude']
            if not (longitude_range[0] <= lon <= longitude_range[1]):
                match = False
        
        if month and match:
            profile_month = metadata['time'].month
            if profile_month != month:
                match = False
        
        if match:
            result = {
                'index': i,
                'summary': embedding_index.profile_summaries[i],
                'metadata': metadata
            }
            results.append(result)
    
    return results

# Example usage
if __name__ == "__main__":
    from data_ingest import ARGODataIngestor
    
    # Create some sample data
    ingestor = ARGODataIngestor()
    profiles = ingestor.process_all_files()
    
    if profiles:
        # Create embedding index
        embedding_index = ProfileEmbeddingIndex()
        embedding_index.create_embeddings(profiles)
        
        # Test searches
        queries = [
            "Show me temperature profiles near the equator",
            "Find salinity measurements in deep water",
            "Profiles from January with warm temperatures"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            results = embedding_index.search_similar_profiles(query, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result['similarity_score']:.3f}")
                print(f"   {result['summary']}")
    
        # Print statistics
        stats = embedding_index.get_statistics()
        print(f"\n📊 Index Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    else:
        print("No profiles found. Download some data first!")