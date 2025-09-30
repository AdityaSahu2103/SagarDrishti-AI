"""
Ingest ARGO NetCDF-derived profile data into PostgreSQL.

This script uses the project's existing configuration and data ingestor to:
1) Read profiles from the configured data directory (e.g., Config.INDIAN_OCEAN_PATH)
2) Create PostgreSQL tables if they do not exist
3) Insert either a demo subset (default 1 profile) or all profiles

Usage examples:
  - Demo (one profile):
      python ingest_to_postgres.py --demo-count 1

  - Ingest all profiles:
      python ingest_to_postgres.py --all

Environment/Config:
  - Prefers using project's `db.get_engine(Config)` if available and POSTGRES_ENABLED=true.
  - Otherwise falls back to DATABASE_URL env var for SQLAlchemy.
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    from config import Config
    from data_ingest import ARGODataIngestor
except Exception as e:  # pragma: no cover
    print(f"Failed to import project modules: {e}", file=sys.stderr)
    sys.exit(1)

# Optional DB helpers from the project
try:
    from db import get_engine as project_get_engine, init_db as project_init_db  # type: ignore
except Exception:
    project_get_engine = None  # type: ignore
    project_init_db = None  # type: ignore

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


PROFILES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.argo_profiles (
    id BIGSERIAL PRIMARY KEY,
    file_source TEXT,
    profile_idx INTEGER,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    time TIMESTAMPTZ,
    n_measurements INTEGER,
    depth_range TEXT,
    temp_range TEXT,
    summary TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_profile_file_idx UNIQUE (file_source, profile_idx)
);
"""

MEASUREMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.argo_measurements (
    id BIGSERIAL PRIMARY KEY,
    profile_id BIGINT REFERENCES public.argo_profiles(id) ON DELETE CASCADE,
    pressure DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    salinity DOUBLE PRECISION
);
"""


def _get_engine(cfg: Config) -> Engine:
    """Return a SQLAlchemy engine, preferring the project's db helpers when enabled."""
    # Prefer project DB integration if enabled and available
    if getattr(cfg, "POSTGRES_ENABLED", False) and project_get_engine is not None:
        if project_init_db is not None:
            try:
                project_init_db(cfg)
            except Exception:
                pass
        return project_get_engine(cfg)  # type: ignore

    # Fallback: use DATABASE_URL
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set and project db.get_engine is unavailable.")
    return create_engine(db_url)


def _ensure_schema(engine: Engine) -> None:
    """Create required tables if they do not exist."""
    with engine.begin() as conn:
        conn.execute(text(PROFILES_TABLE_SQL))
        conn.execute(text(MEASUREMENTS_TABLE_SQL))
        # Helpful indexes to speed up joins and lookups
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_profiles_file_idx ON public.argo_profiles(file_source, profile_idx)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_measurements_profile_id ON public.argo_measurements(profile_id)"))


def _safe_float(val) -> Optional[float]:
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def _infer_time(df: pd.DataFrame) -> Optional[datetime]:
    """Attempt to infer a representative timestamp for the profile."""
    for col in ("time", "timestamp", "date", "datetime"):
        if col in df.columns:
            try:
                series = pd.to_datetime(df[col], errors="coerce")
                if series.notna().any():
                    return pd.to_datetime(series.dropna().iloc[0]).to_pydatetime()
            except Exception:
                pass
    return None


def _summarize_profile(df: pd.DataFrame) -> tuple[str, str, str, int]:
    n = int(len(df))
    p_min = df["pressure"].min() if "pressure" in df else np.nan
    p_max = df["pressure"].max() if "pressure" in df else np.nan
    t_min = df["temperature"].min() if "temperature" in df else np.nan
    t_max = df["temperature"].max() if "temperature" in df else np.nan

    depth_range = (
        f"{float(p_min):.0f}–{float(p_max):.0f} dbar" if not (pd.isna(p_min) or pd.isna(p_max)) else ""
    )
    temp_range = (
        f"{float(t_min):.1f}–{float(t_max):.1f} °C" if not (pd.isna(t_min) or pd.isna(t_max)) else ""
    )

    lat = df["latitude"].iloc[0] if "latitude" in df else None
    lon = df["longitude"].iloc[0] if "longitude" in df else None
    summary = f"Profile at {lat:.2f}°, {lon:.2f}° with {n} measurements" if lat is not None and lon is not None else f"Profile with {n} measurements"

    return depth_range, temp_range, summary, n


def _insert_profile(engine: Engine, profile_idx: int, df: pd.DataFrame) -> int:
    """Insert one profile and its measurements. Returns inserted profile id."""
    if df is None or df.empty:
        raise ValueError("Empty profile DataFrame")

    file_source = str(df["file_source"].iloc[0]) if "file_source" in df else None
    latitude = _safe_float(df["latitude"].iloc[0]) if "latitude" in df else None
    longitude = _safe_float(df["longitude"].iloc[0]) if "longitude" in df else None
    when = _infer_time(df)
    depth_range, temp_range, summary, n_measurements = _summarize_profile(df)

    with engine.begin() as conn:
        res = conn.execute(
            text(
                """
                INSERT INTO public.argo_profiles
                (file_source, profile_idx, latitude, longitude, time, n_measurements, depth_range, temp_range, summary)
                VALUES (:file_source, :profile_idx, :latitude, :longitude, :time, :n_measurements, :depth_range, :temp_range, :summary)
                RETURNING id
                """
            ),
            {
                "file_source": file_source,
                "profile_idx": int(profile_idx),
                "latitude": latitude,
                "longitude": longitude,
                "time": when,
                "n_measurements": n_measurements,
                "depth_range": depth_range or None,
                "temp_range": temp_range or None,
                "summary": summary,
            },
        )
        profile_id = int(res.scalar_one())

        # Insert measurements (pressure, temperature, salinity)
        pressures = df["pressure"].tolist() if "pressure" in df else []
        temps = df["temperature"].tolist() if "temperature" in df else []
        salts = df["salinity"].tolist() if "salinity" in df else []
        rows = []
        for i in range(len(df)):
            p = _safe_float(pressures[i] if i < len(pressures) else None)
            t = _safe_float(temps[i] if i < len(temps) else None)
            s = _safe_float(salts[i] if i < len(salts) else None)
            rows.append({"profile_id": profile_id, "pressure": p, "temperature": t, "salinity": s})

        if rows:
            conn.execute(
                text(
                    """
                    INSERT INTO public.argo_measurements (profile_id, pressure, temperature, salinity)
                    VALUES (:profile_id, :pressure, :temperature, :salinity)
                    """
                ),
                rows,
            )

    return profile_id


def _profile_exists(engine: Engine, file_source: str, profile_idx: int) -> bool:
    with engine.begin() as conn:
        res = conn.execute(
            text(
                "SELECT 1 FROM public.argo_profiles WHERE file_source=:fs AND profile_idx=:pi LIMIT 1"
            ),
            {"fs": file_source, "pi": int(profile_idx)},
        )
        return res.first() is not None


def _resolve_profile_idx(df: pd.DataFrame, fallback: int) -> int:
    try:
        if "profile_idx" in df.columns:
            val = df["profile_idx"].iloc[0]
            if pd.notna(val):
                return int(val)
    except Exception:
        pass
    return int(fallback)


def run_ingest(demo_count: Optional[int], ingest_all: bool, verbose: bool) -> None:
    cfg = Config()

    # Prepare engine and schema
    engine = _get_engine(cfg)
    _ensure_schema(engine)

    # Build profiles using existing ingestor
    ingestor = ARGODataIngestor(cfg)
    profiles = ingestor.process_all_files()
    if not profiles:
        raise RuntimeError(
            f"No ARGO profiles found in directory: {getattr(cfg, 'INDIAN_OCEAN_PATH', 'unknown')}"
        )

    total = len(profiles)
    if verbose:
        print(f"Discovered {total} profile(s).", flush=True)

    num_to_load = total if ingest_all else max(0, int(demo_count or 1))

    inserted_ids: list[int] = []
    duplicates_skipped = 0
    for idx, df in enumerate(profiles):
        if df is None or df.empty:
            if verbose:
                print(f"Skipping empty profile at index {idx}")
            continue
        try:
            file_source = str(df["file_source"].iloc[0]) if "file_source" in df else None
            resolved_idx = _resolve_profile_idx(df, idx)

            if file_source is None:
                if verbose:
                    print(f"Profile {idx} missing file_source; inserting anyway with idx={resolved_idx}")
            else:
                if _profile_exists(engine, file_source, resolved_idx):
                    duplicates_skipped += 1
                    if verbose:
                        print(f"Duplicate exists, skipping: (file_source='{file_source}', profile_idx={resolved_idx})")
                    # For demo mode: keep searching until we insert the requested number of uniques
                    continue

            pid = _insert_profile(engine, resolved_idx, df)
            inserted_ids.append(pid)
            if verbose:
                src = df["file_source"].iloc[0] if "file_source" in df else f"idx_{idx}"
                print(f"Inserted profile {pid} from {src} (profile_idx={resolved_idx})")

            # Stop early in demo mode when we have enough uniques
            if not ingest_all and len(inserted_ids) >= num_to_load:
                break
        except Exception as e:
            print(f"Failed to insert profile idx={idx}: {e}", file=sys.stderr)

    if not ingest_all:
        print(f"Demo complete. Inserted {len(inserted_ids)} unique profile(s). IDs: {inserted_ids}. Duplicates skipped: {duplicates_skipped}")
    else:
        print(f"Ingestion complete. Inserted {len(inserted_ids)} new unique profile(s). Duplicates skipped: {duplicates_skipped}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest ARGO profiles into PostgreSQL")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--demo-count", type=int, default=1, help="Number of profiles to ingest for demo (default: 1)")
    mode.add_argument("--all", action="store_true", help="Ingest all profiles")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    run_ingest(None if args.all else args.demo_count, args.all, args.verbose)


if __name__ == "__main__":
    main()


