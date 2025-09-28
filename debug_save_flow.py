# debug_save_flow.py
import pprint
import pandas as pd
from config import Config
from data_ingest import ARGODataIngestor
from embedding_index import ProfileEmbeddingIndex
from db import save_profiles_to_postgres
from utils import setup_logging

logger = setup_logging(__name__)
cfg = Config()
cfg.validate_config()

print("CONFIG POSTGRES_ENABLED:", cfg.POSTGRES_ENABLED)
print("DATA ROOT:", cfg.DATA_ROOT)
print("INDIAN_OCEAN_PATH:", cfg.INDIAN_OCEAN_PATH)

ingestor = ARGODataIngestor(cfg)
print("Calling ingestor.process_all_files() ...")
profiles = ingestor.process_all_files()

if not profiles:
    print("=> process_all_files() returned empty or None.")
else:
    print("=> process_all_files() returned", len(profiles), "items.")
    # Print brief info about first 3 profiles
    for idx, p in enumerate(profiles[:3]):
        print(f"\n--- profile index {idx} type: {type(p)} ---")
        if isinstance(p, pd.DataFrame):
            print("DataFrame shape:", p.shape)
            print("Columns:", list(p.columns))
            print("dtypes:\n", p.dtypes)
            print("head:\n", p.head(3))
            print("null counts:\n", p.isnull().sum())
            # try to get file_source from df if present
            if "file_source" in p.columns:
                print("file_source sample:", p["file_source"].iloc[0])
        elif isinstance(p, dict):
            print("dict keys:", list(p.keys()))
            # if dict-of-arrays, convert sample to df
            try:
                df = pd.DataFrame(p)
                print("Converted dict -> DataFrame shape:", df.shape)
                print(df.head(2))
            except Exception as e:
                print("Could not convert dict to DataFrame:", e)
        else:
            # print repr for unknown types
            print("repr:", repr(p)[:400])

# If embedding index holds metadata, print a sample (optional)
try:
    embedding_idx = ProfileEmbeddingIndex(cfg)
    if getattr(embedding_idx, "profile_metadata", None):
        print("\nEmbedding index metadata count:", len(embedding_idx.profile_metadata))
        pprint.pprint(embedding_idx.profile_metadata[:3])
except Exception as e:
    print("Could not initialize embedding index (not fatal):", e)

# Attempt to save using the same function used in API
print("\nCalling save_profiles_to_postgres(...)")
try:
    saved = save_profiles_to_postgres(
        profiles=profiles,
        summaries=getattr(ingestor, "summaries", None) or [],
        metadata_list=getattr(ingestor, "metadata_list", None) or [],
        cfg=cfg
    )
    print("save_profiles_to_postgres returned:", saved)
except Exception as e:
    print("save_profiles_to_postgres raised exception:", e)
