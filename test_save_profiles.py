# test_save_profiles.py
import pandas as pd
from db import save_profiles_to_postgres
from config import Config

cfg = Config()
cfg.validate_config()

# Build a simple DataFrame that matches expected columns
df = pd.DataFrame({
    "pressure": [0.5, 10.0, 20.0],
    "temperature": [28.5, 26.1, 24.3],
    "salinity": [35.2, 35.1, 35.0],
    "time": [pd.Timestamp("2023-01-01T00:00:00")] * 3,
    "file_source": ["testfile.nc"] * 3,
})

profiles = [df]
summaries = ["Test profile summary"]
metadata_list = [{"file_source": "testfile.nc", "profile_idx": 0, "latitude": 10.0, "longitude": 70.0, "time": "2023-01-01"}]

saved = save_profiles_to_postgres(profiles=profiles, summaries=summaries, metadata_list=metadata_list, cfg=cfg)
print("Saved count:", saved)
