"""
MCP Server for FloatChat oceanographic tools.

Primary implementation uses fastmcp if available. If not installed, the script
falls back to a local debug message without starting a server, so it never
breaks the app environment.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from mcp_tools import (
    argo_sql_query,
    get_profiles_by_region,
    compare_profiles,
    get_temporal_data,
    calculate_statistics,
    export_data,
    RegionBounds,
)

try:
    # fastmcp provides a simple Pythonic MCP server runtime
    from fastmcp import MCP
    FASTMCP_AVAILABLE = True
except Exception:
    FASTMCP_AVAILABLE = False


def run_local_debug() -> None:
    print("MCP server runtime not available. Running in local debug mode.")
    print("Example: from mcp_tools import argo_sql_query; argo_sql_query('Show temperature profiles in Arabian Sea for March 2023')")


def run_fastmcp_server() -> None:
    mcp = MCP("floatchat-mcp")

    @mcp.tool()
    def argo_sql_query_tool(nl_query: str, table: str = "argo_profiles") -> Dict[str, Any]:
        return argo_sql_query(nl_query, table)

    @mcp.tool()
    def get_profiles_by_region_tool(data_dir: str, bounds: Dict[str, Any]) -> Dict[str, Any]:
        rb = RegionBounds(**bounds)
        dfs = get_profiles_by_region(data_dir, rb)
        info = [
            {
                "file_source": (df["file_source"].iloc[0] if not df.empty else ""),
                "latitude": float(df["latitude"].iloc[0]) if not df.empty else None,
                "longitude": float(df["longitude"].iloc[0]) if not df.empty else None,
                "max_depth": float(df["pressure"].max()) if not df.empty else None,
            }
            for df in dfs
        ]
        return {"count": len(dfs), "profiles": info}

    @mcp.tool()
    def get_temporal_data_tool(data_dir: str, param: str, start: Any | None = None, end: Any | None = None, bounds: Dict[str, Any] | None = None):
        rb = RegionBounds(**bounds) if bounds else None
        df = get_temporal_data(data_dir, param, start, end, rb)
        return json.loads(df.to_json(orient="records", date_format="iso"))

    @mcp.tool()
    def calculate_statistics_tool(param: str) -> Dict[str, Any]:
        # In this simple server, we don't receive DataFrames over the wire.
        # Keep this as a placeholder indicating in-process usage.
        return {"error": "calculate_statistics is intended for in-process DataFrames."}

    @mcp.tool()
    def export_data_tool(fmt: str = "csv") -> Dict[str, Any]:
        return {"message": "export_data should be called in-process to write files."}

    # Start the server (fastmcp chooses a transport, typically stdio or websockets)
    mcp.run()


def main() -> None:
    if FASTMCP_AVAILABLE:
        run_fastmcp_server()
    else:
        run_local_debug()


if __name__ == "__main__":
    main()
