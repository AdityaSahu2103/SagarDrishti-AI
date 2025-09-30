import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def get_region(latitude, longitude):
    """Placeholder function to simulate ocean basin lookup."""
    if float(latitude) > 0 and float(longitude) < 0:
        return "North Atlantic"
    if float(latitude) <= 0 and float(longitude) >= 20 and float(longitude) <= 146:
        return "Indian Ocean"
    return "Unknown"


# --- Home Page Globe (Requires df_profiles DataFrame) ---
def display_home_page_globe(df_profiles):
    """
    Renders the attractive, spinning, satellite-like globe using Plotly scatter_geo.
    """
    if df_profiles.empty:
        st.info("No data profiles available to display on the globe map.")
        return

    try:
        # Compute regions (requires the external get_region function)
        df_profiles["Region"] = df_profiles.apply(
            lambda r: get_region(r["latitude"], r["longitude"]), axis=1
        )
    except Exception:
        df_profiles["Region"] = "Unknown"

    # Rename columns for clear Plotly mapping
    df_map = df_profiles.rename(
        columns={"latitude": "Latitude", "longitude": "Longitude"}
    )

    # Orthographic globe setup
    fig = px.scatter_geo(
        df_map,
        lat="Latitude",
        lon="Longitude",
        color="n_measurements",
        hover_data=["file_source", "time", "Region"],
        projection="orthographic",
        color_continuous_scale=[
            "#00eaff",
            "#00ffa3",
            "#39ff14",
            "#f7ff00",
            "#ff00ff",
        ],
        title="<span style='color:white;'>Profiles</span>",
    )

    # Add hidden traces for region legends
    for region in sorted(df_map["Region"].dropna().unique().tolist()):
        sample = df_map[df_map["Region"] == region].head(1)
        if not sample.empty:
            fig.add_trace(
                go.Scattergeo(
                    lat=[float(sample.iloc[0]["Latitude"])],
                    lon=[float(sample.iloc[0]["Longitude"])],
                    mode="markers",
                    marker=dict(size=1, opacity=0),
                    name=str(region),
                    hoverinfo="skip",
                    showlegend=True,
                    legendgroup="Region",
                )
            )

    # Satellite-like Styling
    fig.update_traces(marker=dict(size=5, opacity=0.9))
    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        showocean=True,
        # Realistic Earth colors for satellite look
        landcolor="rgb(120, 160, 100)",  # Earthy green/tan
        oceancolor="rgb(20, 100, 180)",  # Ocean blue
        showcountries=False,
        showcoastlines=True,
        coastlinecolor="rgb(120, 120, 120)",
        showsubunits=True,
        subunitcolor="rgb(120, 120, 120)",
        bgcolor="rgba(0,0,0,0)",
        lataxis_showgrid=False,
        lonaxis_showgrid=False,
    )

    # Layout, Size, and Smooth Spin Animation
    frames = [
        go.Frame(
            layout=go.Layout(geo_projection_rotation=dict(lon=k * 5, lat=10, roll=0))
        )
        for k in range(0, 72)
    ]
    fig.frames = frames

    fig.update_layout(
        height=700,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark",
        title_font_color="white",
        hovermode="closest",
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Spin",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 1000, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 800, "easing": "linear"}, # Smoother spin settings
                            },
                        ],
                    )
                ],
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Database Explorer Globe (Requires df_profiles DataFrame) ---
def display_database_explorer_globe(df_profiles):
    """
    Renders the attractive, spinning, satellite-like globe for the Database tab.
    (Functionally identical to the home page globe for consistency)
    """
    if df_profiles.empty:
        st.info("No data profiles retrieved from the database to display on the globe map.")
        return

    try:
        df_profiles["Region"] = df_profiles.apply(
            lambda r: get_region(r["latitude"], r["longitude"]), axis=1
        )
    except Exception:
        df_profiles["Region"] = "Unknown"

    df_map = df_profiles.rename(
        columns={"latitude": "Latitude", "longitude": "Longitude"}
    )

    fig = px.scatter_geo(
        df_map,
        lat="Latitude",
        lon="Longitude",
        color="n_measurements",
        hover_data=["file_source", "profile_idx", "time", "Region"],
        projection="orthographic",
        title="<span style='color:white;'>Profiles (from PostgreSQL)</span>",
    )

    for region in sorted(df_map["Region"].dropna().unique().tolist()):
        sample = df_map[df_map["Region"] == region].head(1)
        if not sample.empty:
            fig.add_trace(
                go.Scattergeo(
                    lat=[float(sample.iloc[0]["Latitude"])],
                    lon=[float(sample.iloc[0]["Longitude"])],
                    mode="markers",
                    marker=dict(size=1, opacity=0),
                    name=str(region),
                    hoverinfo="skip",
                    legendgroup="Region",
                )
            )

    # Satellite-like Styling
    fig.update_traces(marker=dict(size=5, opacity=0.9))
    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        showocean=True,
        landcolor="rgb(120, 160, 100)",
        oceancolor="rgb(20, 100, 180)",
        showcountries=False,
        showcoastlines=True,
        coastlinecolor="rgb(120, 120, 120)",
        showsubunits=True,
        subunitcolor="rgb(120, 120, 120)",
        bgcolor="rgba(0,0,0,0)",
        lataxis_showgrid=False,
        lonaxis_showgrid=False,
    )

    # Layout, Size, and Smooth Spin Animation
    frames = [
        go.Frame(
            layout=go.Layout(geo_projection_rotation=dict(lon=k * 5, lat=10, roll=0))
        )
        for k in range(0, 72)
    ]
    fig.frames = frames

    fig.update_layout(
        height=700,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        template="plotly_dark",
        title_font_color="white",
        hovermode="closest",
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Spin",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 1000, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 800, "easing": "linear"}, # Smoother spin settings
                            },
                        ],
                    )
                ],
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True)

# You can integrate this code into your existing Streamlit app by:
# 1. Importing the functions (e.g., from globe_functions import display_home_page_globe)
# 2. Replacing the old map visualization code with a call to the appropriate function, passing your DataFrame.