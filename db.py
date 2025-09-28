# db.py (improved persistence helpers)
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional, Any

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from config import Config
from utils import setup_logging

logger = setup_logging(__name__)

Base = declarative_base()


class Profile(Base):
    __tablename__ = "argo_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_source = Column(String(255), index=True)
    profile_idx = Column(Integer, default=0)
    latitude = Column(Float)
    longitude = Column(Float)
    time = Column(DateTime)
    n_measurements = Column(Integer)
    depth_range = Column(String(64))
    temp_range = Column(String(64))
    summary = Column(String(512))
    created_at = Column(DateTime, server_default=text("NOW()"))

    measurements = relationship("Measurement", back_populates="profile", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("file_source", "profile_idx", name="uq_profile_file_idx"),
    )


class Measurement(Base):
    __tablename__ = "argo_measurements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey("argo_profiles.id", ondelete="CASCADE"), index=True)
    pressure = Column(Float)
    temperature = Column(Float)
    salinity = Column(Float)

    profile = relationship("Profile", back_populates="measurements")


def _build_db_url(cfg: Config = Config) -> str:
    if getattr(cfg, "POSTGRES_URL", None):
        return cfg.POSTGRES_URL
    return (
        f"postgresql+psycopg2://{cfg.POSTGRES_USER}:{cfg.POSTGRES_PASSWORD}"
        f"@{cfg.POSTGRES_HOST}:{cfg.POSTGRES_PORT}/{cfg.POSTGRES_DB}?sslmode={cfg.POSTGRES_SSLMODE}"
    )


def get_engine(cfg: Config = Config):
    db_url = _build_db_url(cfg)
    logger.info(f"Connecting to PostgreSQL at {cfg.POSTGRES_HOST}:{cfg.POSTGRES_PORT}/{cfg.POSTGRES_DB}")
    engine = create_engine(db_url, pool_pre_ping=True, future=True)
    return engine


def init_db(cfg: Config = Config) -> None:
    engine = get_engine(cfg)
    Base.metadata.create_all(engine)
    logger.info("PostgreSQL tables ensured (argo_profiles, argo_measurements)")


def save_profiles_to_postgres(
    profiles: List[pd.DataFrame] | List[Any],
    summaries: Optional[List[str]] = None,
    metadata_list: Optional[List[dict]] = None,
    cfg: Config = Config,
) -> int:
    """
    Persist parsed ARGO profiles and level measurements into PostgreSQL.
    Returns number of profiles saved/updated.
    """
    if not getattr(cfg, "POSTGRES_ENABLED", False):
        logger.warning("POSTGRES_ENABLED is False; skipping save.")
        return 0

    if not profiles:
        logger.warning("No profiles provided to save_profiles_to_postgres.")
        return 0

    # Ensure DB/schema
    init_db(cfg)
    engine = get_engine(cfg)
    SessionLocal = sessionmaker(bind=engine, future=True)

    saved = 0
    summaries = summaries or []
    metadata_list = metadata_list or []

    logger.info(f"Attempting to save {len(profiles)} profiles to Postgres")

    with SessionLocal() as session:
        try:
            for i, df in enumerate(profiles):
                try:
                    # Accept either DataFrame or something convertible (dict)
                    if df is None:
                        logger.debug(f"Skipping empty profile index {i} (None).")
                        continue

                    if isinstance(df, dict):
                        df = pd.DataFrame(df)

                    if not isinstance(df, pd.DataFrame):
                        logger.warning(f"Profile at index {i} not a DataFrame (type={type(df)}). Skipping.")
                        continue

                    if df.empty:
                        logger.debug(f"Skipping empty DataFrame at index {i}.")
                        continue

                    meta = metadata_list[i] if i < len(metadata_list) else {}
                    summary = summaries[i] if i < len(summaries) else None

                    # Resolve file_source and profile_idx
                    file_source = str(meta.get("file_source") or (df.get("file_source", pd.Series([None])).iloc[0]))
                    profile_idx = int(meta.get("profile_idx", i))

                    existing: Optional[Profile] = (
                        session.query(Profile)
                        .filter(Profile.file_source == file_source, Profile.profile_idx == profile_idx)
                        .one_or_none()
                    )

                    # Construct time safely
                    time_val = meta.get("time")
                    if time_val is None:
                        # try from df
                        if "time" in df.columns:
                            time_val = df["time"].iloc[0]
                    try:
                        time_val = pd.to_datetime(time_val)
                    except Exception:
                        time_val = datetime.utcnow()

                    # create or update profile
                    if existing is None:
                        profile = Profile(
                            file_source=file_source,
                            profile_idx=profile_idx,
                            latitude=float(meta.get("latitude", df.get("latitude", pd.Series([0])).iloc[0] or 0)),
                            longitude=float(meta.get("longitude", df.get("longitude", pd.Series([0])).iloc[0] or 0)),
                            time=time_val,
                            n_measurements=int(meta.get("n_measurements", len(df))),
                            depth_range=str(meta.get("depth_range", "")),
                            temp_range=str(meta.get("temp_range", "")),
                            summary=summary or "",
                        )
                        session.add(profile)
                        session.flush()  # ensure profile.id available
                    else:
                        profile = existing
                        profile.latitude = float(meta.get("latitude", profile.latitude or 0))
                        profile.longitude = float(meta.get("longitude", profile.longitude or 0))
                        profile.time = time_val
                        profile.n_measurements = int(meta.get("n_measurements", len(df)))
                        profile.depth_range = str(meta.get("depth_range", profile.depth_range or ""))
                        profile.temp_range = str(meta.get("temp_range", profile.temp_range or ""))
                        profile.summary = summary or profile.summary or ""
                        # remove existing measurements
                        profile.measurements.clear()
                        session.add(profile)
                        session.flush()

                    # Add measurements
                    records = []
                    for _, row in df.iterrows():
                        p = row.get("pressure", None)
                        t = row.get("temperature", None)
                        s = row.get("salinity", None)

                        # convert possible NaN to None
                        p_val = None if pd.isna(p) else float(p)
                        t_val = None if pd.isna(t) else float(t)
                        s_val = None if pd.isna(s) else float(s)

                        records.append(Measurement(pressure=p_val, temperature=t_val, salinity=s_val))

                    profile.measurements.extend(records)
                    session.add(profile)
                    saved += 1

                except Exception as perr:
                    logger.exception(f"Failed to save profile index {i}: {perr}")
                    # continue to next profile

            session.commit()
        except Exception as e:
            logger.exception(f"Fatal error saving profiles: {e}")
            session.rollback()
            raise

    logger.info(f"Saved {saved} profiles to PostgreSQL")
    return saved
