"""
MCP Client for FloatChat.

This client provides a simple facade around the oceanographic tools. It will:
- Prefer calling a local MCP server via the Python MCP SDK if available (future-ready).
- Fall back to direct in-process calls to functions in mcp_tools.py.

This keeps the dashboard and RAG engine decoupled from the transport details.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Transport capability (optional). If unavailable, we'll use direct calls.
try:
    from mcp.client import Client  # type: ignore
    MCP_TRANSPORT_AVAILABLE = True
except Exception:
    MCP_TRANSPORT_AVAILABLE = False

try:
    from .mcp_tools import (
        argo_sql_query,
        get_profiles_by_region,
        compare_profiles,
        get_temporal_data,
        calculate_statistics,
        export_data,
        RegionBounds,
    )
except Exception:
    # Relative import fallback when used as a script
    from mcp_tools import (  # type: ignore
        argo_sql_query,
        get_profiles_by_region,
        compare_profiles,
        get_temporal_data,
        calculate_statistics,
        export_data,
        RegionBounds,
    )


class MCPClient:
    def __init__(self, server_url: Optional[str] = None) -> None:
        self.server_url = server_url
        self.remote: Optional[Any] = None
        # Placeholder: if you have a running MCP server and SDK, initialize the connection here.
        if MCP_TRANSPORT_AVAILABLE and server_url:
            try:
                self.remote = Client(server_url)  # Not used in this minimal integration
            except Exception:
                self.remote = None

    # --- Tool Facade Methods ---
    def tool_argo_sql_query(self, nl_query: str, table: str = "argo_profiles") -> Dict[str, Any]:
        if self.remote is not None:
            # Example call if MCP transport is wired
            try:
                return self.remote.call_tool("argo_sql_query", {"nl_query": nl_query, "table": table})
            except Exception:
                pass
        return argo_sql_query(nl_query, table)

    def tool_get_profiles_by_region(self, data_dir: str, bounds: RegionBounds):
        if self.remote is not None:
            try:
                return self.remote.call_tool("get_profiles_by_region", {"data_dir": data_dir, "bounds": bounds.__dict__})
            except Exception:
                pass
        return get_profiles_by_region(data_dir, bounds)

    def tool_compare_profiles(self, dfs: List):
        return compare_profiles(dfs)

    def tool_get_temporal_data(self, data_dir: str, param: str, start=None, end=None, bounds: Optional[RegionBounds] = None):
        return get_temporal_data(data_dir, param, start, end, bounds)

    def tool_calculate_statistics(self, dfs: List, param: str):
        return calculate_statistics(dfs, param)

    def tool_export_data(self, df_or_dfs, path: str, fmt: str = "csv"):
        return export_data(df_or_dfs, path, fmt)
