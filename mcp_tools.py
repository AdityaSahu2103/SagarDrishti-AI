"""
MCP Tools for FloatChat oceanographic data operations.
Each tool function is designed to be used by an MCP server or directly imported.

Tools implemented:
- argo_sql_query: Generate SQL for ARGO-like database queries (and an equivalent pandas filter plan)
- get_profiles_by_region: Extract profiles by geographic bounds
- compare_profiles: Compare multiple ARGO profiles (temperature/salinity/depth)
- get_temporal_data: Time series data extraction
- calculate_statistics: Ocean parameter statistics
- export_data: Export results to NetCDF/CSV formats

All tools operate on NetCDF files found under a given data root (e.g., Config.INDIAN_OCEAN_PATH)
or on preloaded pandas DataFrames supplied by the caller.
"""
from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np
import xarray as xr


@dataclass
class RegionBounds:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    min_depth: Optional[float] = None
    max_depth: Optional[float] = None


def _load_all_profiles_from_dir(data_dir: str) -> List[pd.DataFrame]:
    """Load NetCDF profiles from a directory into a list of pandas DataFrames.
    Expected variables: temperature, salinity, pressure (depth proxy), latitude, longitude, time.
    """
    profiles: List[pd.DataFrame] = []
    if not os.path.isdir(data_dir):
        return profiles

    for path in glob.glob(os.path.join(data_dir, "*.nc")):
        try:
            ds = xr.open_dataset(path)
            df = pd.DataFrame({
                "temperature": ds["temperature"].to_numpy().ravel() if "temperature" in ds else np.array([]),
                "salinity": ds["salinity"].to_numpy().ravel() if "salinity" in ds else np.array([]),
                "pressure": ds["pressure"].to_numpy().ravel() if "pressure" in ds else np.array([]),
            })
            # Broadcast scalar coords if necessary
            lat = float(ds["latitude"].values) if "latitude" in ds else np.nan
            lon = float(ds["longitude"].values) if "longitude" in ds else np.nan
            time = pd.to_datetime(ds["time"].values) if "time" in ds else pd.NaT
            df["latitude"] = lat
            df["longitude"] = lon
            df["time"] = time
            df["file_source"] = os.path.basename(path)
            profiles.append(df.dropna(subset=["temperature", "pressure"], how="all"))
        except Exception:
            # Skip unreadable files
            continue
    return profiles


def argo_sql_query(nl_query: str, table: str = "argo_profiles") -> Dict[str, Any]:
    """Generate a SQL-like query and structured filter plan from a natural language query.

    Supports:
    - Parameter filters: temperature/salinity comparisons (>, <, between)
    - Depth filters: pressure/depth thresholds (e.g., "> 500 m", "deepest")
    - Time filters: "March 2023", "2023-03", or year-only
    - Region filters: named basins (Arabian Sea, Bay of Bengal, etc.) mapped to lat/lon bounds
    - Proximity: "near 10N 75E within 200km" (returned as post_filters for client-side haversine)

    Returns: {
      sql, filters, order_by, limit, notes, post_filters
    }
    """
    import re
    q_raw = nl_query
    q = nl_query.lower()

    filters: List[Dict[str, Any]] = []
    where_clauses: List[str] = []
    order_by: Optional[Tuple[str, str]] = None
    limit: Optional[int] = None
    notes: List[str] = []
    post_filters: Dict[str, Any] = {}

    # --- Region dictionary (approximate bounding boxes) ---
    region_boxes = {
        "arabian sea": {"min_lat": 8, "max_lat": 25, "min_lon": 60, "max_lon": 78},
        "bay of bengal": {"min_lat": 8, "max_lat": 22, "min_lon": 80, "max_lon": 100},
        "equatorial indian ocean": {"min_lat": -10, "max_lat": 8, "min_lon": 40, "max_lon": 100},
        "southern indian ocean": {"min_lat": -60, "max_lat": -10, "min_lon": 20, "max_lon": 120},
        "central indian ocean": {"min_lat": -20, "max_lat": 20, "min_lon": 60, "max_lon": 100},
        "indian ocean": {"min_lat": -60, "max_lat": 30, "min_lon": 20, "max_lon": 120},
    }

    # --- Month/year parsing ---
    month_map = dict(jan=1, feb=2, mar=3, apr=4, may=5, jun=6, jul=7, aug=8, sep=9, oct=10, nov=11, dec=12)
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})", q)
    if m:
        mon = month_map.get(m.group(1)[:3], None)
        yr = int(m.group(2))
        if mon:
            filters.append({"field": "month", "op": "=", "value": mon})
        filters.append({"field": "year", "op": "=", "value": yr})
    m2 = re.search(r"(\d{4})-(\d{2})", q)
    if m2:
        filters.append({"field": "year", "op": "=", "value": int(m2.group(1))})
        filters.append({"field": "month", "op": "=", "value": int(m2.group(2))})
    m3 = re.search(r"\b(\d{4})\b", q)
    if m3 and not any(f["field"] == "year" for f in filters):
        filters.append({"field": "year", "op": "=", "value": int(m3.group(1))})

    # --- Parameter mentions ---
    if "temperature" in q:
        filters.append({"field": "has_temperature", "op": "=", "value": True})
    if "salinity" in q or "psu" in q:
        filters.append({"field": "has_salinity", "op": "=", "value": True})

    # --- Comparisons for temperature/salinity (>, <, between) ---
    def add_numeric_filter(field: str, pattern: str):
        for m in re.finditer(pattern, q):
            op = m.group(1)
            val = float(m.group(2))
            filters.append({"field": field, "op": op, "value": val})

    # e.g., "salinity > 35", "temperature < 20"
    add_numeric_filter("salinity", r"salinity\s*(>|<|>=|<=|=)\s*(\d+\.?\d*)")
    add_numeric_filter("temperature", r"temperature\s*(>|<|>=|<=|=)\s*(\d+\.?\d*)")

    # between ranges: "temperature 20-25", "salinity 34 to 36"
    m_between = re.search(r"(temperature|salinity)\s*(\d+\.?\d*)\s*(?:-|to)\s*(\d+\.?\d*)", q)
    if m_between:
        fld = m_between.group(1)
        lo = float(m_between.group(2)); hi = float(m_between.group(3))
        filters.append({"field": fld, "op": ">=", "value": lo})
        filters.append({"field": fld, "op": "<=", "value": hi})

    # --- Depth/pressure: "> 500 m", "deeper than 1000 m", "deepest" ---
    m_depth_cmp = re.search(r"(>|>=|<|<=)\s*(\d{2,4})\s*m", q)
    if m_depth_cmp:
        filters.append({"field": "pressure", "op": m_depth_cmp.group(1), "value": int(m_depth_cmp.group(2))})
        order_by = order_by or ("pressure", "DESC")
    if "deeper" in q or "deepest" in q:
        order_by = ("pressure", "DESC")

    # --- Named regions to lat/lon bounding boxes ---
    for name, box in region_boxes.items():
        if name in q:
            filters.append({"field": "latitude", "op": ">=", "value": box["min_lat"]})
            filters.append({"field": "latitude", "op": "<=", "value": box["max_lat"]})
            filters.append({"field": "longitude", "op": ">=", "value": box["min_lon"]})
            filters.append({"field": "longitude", "op": "<=", "value": box["max_lon"]})
            notes.append(f"Applied region box for {name}")

    # --- Proximity: "near 10N 75E within 200km" -> post_filters for client ---
    coord = re.search(r"near\s*(\d+\.?\d*)\s*([ns])\s*(\d+\.?\d*)\s*([we])", q)
    radius = re.search(r"within\s*(\d+\.?\d*)\s*km", q)
    if coord:
        lat = float(coord.group(1)) * (1 if coord.group(2) == 'n' else -1)
        lon = float(coord.group(3)) * (1 if coord.group(4) == 'e' else -1)
        r_km = float(radius.group(1)) if radius else 200.0
        post_filters["proximity"] = {"lat": lat, "lon": lon, "radius_km": r_km}
        notes.append(f"Proximity filter: {lat}, {lon} within {r_km} km (client-side)")

    # --- Compose SQL ---
    def to_sql_clause(f: Dict[str, Any]) -> Optional[str]:
        fld, op, val = f["field"], f["op"], f["value"]
        if val is None:
            return None
        if isinstance(val, str):
            return f"{fld} {op} '{val}'"
        if isinstance(val, bool):
            return f"{fld} IS {str(val).upper()}"
        return f"{fld} {op} {val}"

    for f in filters:
        clause = to_sql_clause(f)
        if clause:
            where_clauses.append(clause)

    sql = f"SELECT * FROM {table}"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    if order_by:
        sql += f" ORDER BY {order_by[0]} {order_by[1]}"
    if limit:
        sql += f" LIMIT {limit}"

    return {
        "sql": sql,
        "filters": filters,
        "order_by": order_by,
        "limit": limit,
        "notes": notes,
        "post_filters": post_filters,
        "query": q_raw,
    }


def get_profiles_by_region(data_dir: str, bounds: RegionBounds) -> List[pd.DataFrame]:
    """Return profile DataFrames whose coordinates fall within the given bounds and optional depth range."""
    profiles = _load_all_profiles_from_dir(data_dir)
    selected: List[pd.DataFrame] = []
    for df in profiles:
        lat = float(df["latitude"].iloc[0]) if not df.empty else np.nan
        lon = float(df["longitude"].iloc[0]) if not df.empty else np.nan
        in_box = (
            bounds.min_lat <= lat <= bounds.max_lat and
            bounds.min_lon <= lon <= bounds.max_lon
        )
        if not in_box:
            continue
        if bounds.min_depth is not None:
            df = df[df["pressure"] >= bounds.min_depth]
        if bounds.max_depth is not None:
            df = df[df["pressure"] <= bounds.max_depth]
        if not df.empty:
            selected.append(df)
    return selected


def compare_profiles(dfs: List[pd.DataFrame]) -> Dict[str, Any]:
    """Compare multiple profiles by summarizing key metrics."""
    out: Dict[str, Any] = {"profiles": []}
    for df in dfs:
        if df.empty:
            continue
        out["profiles"].append({
            "file_source": df["file_source"].iloc[0],
            "mean_temperature": float(np.nanmean(df["temperature"])) if "temperature" in df else np.nan,
            "mean_salinity": float(np.nanmean(df["salinity"])) if "salinity" in df else np.nan,
            "max_depth": float(np.nanmax(df["pressure"])) if "pressure" in df else np.nan,
            "location": (float(df["latitude"].iloc[0]), float(df["longitude"].iloc[0])),
        })
    return out


def get_temporal_data(data_dir: str, param: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None,
                      bounds: Optional[RegionBounds] = None) -> pd.DataFrame:
    """Extract a simple time series for the given parameter across profiles (mean per-profile)."""
    assert param in {"temperature", "salinity", "pressure"}, "Unsupported parameter"
    profiles = _load_all_profiles_from_dir(data_dir)
    rows: List[Dict[str, Any]] = []
    for df in profiles:
        if df.empty:
            continue
        lat = float(df["latitude"].iloc[0])
        lon = float(df["longitude"].iloc[0])
        if bounds:
            if not (bounds.min_lat <= lat <= bounds.max_lat and bounds.min_lon <= lon <= bounds.max_lon):
                continue
        t = df["time"].iloc[0]
        if isinstance(t, pd.Timestamp):
            if start and t < start:
                continue
            if end and t > end:
                continue
        rows.append({
            "time": t,
            param: float(np.nanmean(df[param])) if param in df else np.nan,
            "file_source": df["file_source"].iloc[0],
            "latitude": lat,
            "longitude": lon,
        })
    return pd.DataFrame(rows).sort_values("time")


def calculate_statistics(dfs: List[pd.DataFrame], param: str) -> Dict[str, Any]:
    """Compute basic statistics over a list of profile dataframes for given parameter."""
    assert param in {"temperature", "salinity", "pressure"}, "Unsupported parameter"
    series = pd.concat([df[param] for df in dfs if param in df and not df[param].empty], ignore_index=True)
    if series.empty:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def export_data(df_or_dfs: Any, path: str, fmt: str = "csv") -> str:
    """Export DataFrame(s) to CSV or NetCDF. Returns output path.
    - If df_or_dfs is a single DataFrame, export directly.
    - If it's a list, export concatenated data.
    """
    fmt = fmt.lower()
    if isinstance(df_or_dfs, list):
        df = pd.concat(df_or_dfs, ignore_index=True) if df_or_dfs else pd.DataFrame()
    else:
        df = df_or_dfs

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if fmt == "csv":
        df.to_csv(path, index=False)
        return path
    if fmt in {"nc", "netcdf", "cdf"}:
        ds = xr.Dataset.from_dataframe(df)
        out = path if path.endswith(".nc") else path + ".nc"
        ds.to_netcdf(out)
        return out
    raise ValueError("Unsupported export format: " + fmt)
