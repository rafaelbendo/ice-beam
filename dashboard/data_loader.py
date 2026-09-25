"""Cached parquet loaders shared by all dashboard pages.

Every function here loads exactly one file from dashboard/data/ (the
fig1_aux.ipynb exports) and is wrapped in st.cache_data so each page only
pays for the files it actually uses. Read-only.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "data"

# DSAS tables of the pipeline stages compared on the Erosion Results page
PIPELINE_STAGES_DIR = DATA_DIR / "pipeline_stages"

# Optional, not in the repository: historical (pre-ICESat-2) DSAS
# transect-intersection points. Place the shapefile here to show that layer.
HISTORICAL_DSAS_SHP = DATA_DIR / "historical_dsas" / "DSAS_intersection_bluff_MERGED.shp"

SEA_BOUNDARY_LON = -156.5  # Point Barrow -- same cutoff fig13.ipynb uses for its Chukchi/Beaufort split


@st.cache_data
def load_valid_inventory() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "valid_inventory.parquet")


@st.cache_data
def load_track_dates() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "track_dates_cell16.parquet")


@st.cache_data
def load_track_lon() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "track_lon_df_cell12.parquet")


@st.cache_data
def load_season_counts() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "season_counts_cell16.parquet")


@st.cache_data
def load_coasttype_stats() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "coasttype_stats.parquet")


@st.cache_data
def load_cycles_per_gt() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "cycles_per_gt.parquet")


@st.cache_data
def load_angle_stats() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "angle_stats.parquet")


@st.cache_data
def build_sea_map() -> dict:
    """track_id -> 'Chukchi' or 'Beaufort', by each track's mean longitude.

    Mirrors the sea split fig13.ipynb uses (SEA_BOUNDARY_LON = -156.5, Point
    Barrow), applied here to each track's mean longitude from
    track_lon_df_cell12.parquet -- fig13's own gdf_selected/clusters objects
    aren't part of this dashboard's data source (fig1_aux_outputs/).
    """
    lon = load_track_lon()
    sea = np.where(lon["mean_lon"] < SEA_BOUNDARY_LON, "Chukchi", "Beaufort")
    return dict(zip(lon["track_id"], sea))


@st.cache_data
def load_qc_stats() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "qc_stats_cell24.parquet")


@st.cache_data
def load_good_elev_per_track() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "good_elev_per_track.parquet")


@st.cache_data
def load_good_density_per_track() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "good_density_per_track.parquet")


@st.cache_data
def load_shoreline_utm() -> gpd.GeoDataFrame:
    return gpd.read_parquet(DATA_DIR / "shoreline_utm.parquet")


@st.cache_data
def load_tracks_utm() -> gpd.GeoDataFrame:
    return gpd.read_parquet(DATA_DIR / "tracks_utm.parquet")


@st.cache_data
def load_plot_df_all_files() -> gpd.GeoDataFrame:
    return gpd.read_parquet(DATA_DIR / "plot_df_all_files.parquet")


@st.cache_data
def load_tracks_distance() -> pd.DataFrame:
    """Per-track, per-beam across-track offset (m) -- a plain CSV, not parquet."""
    df = pd.read_csv(DATA_DIR / "tracks_distance.csv")
    df["track_id"] = df["track_id"].astype(str).str.zfill(4)
    return df


@st.cache_data
def build_gt_family_availability() -> gpd.GeoDataFrame:
    """One row per (track_id, gt_family in {gt1,gt2,gt3}) with a real crossing point
    and a completeness_pct/bin -- the per-GT-family analog of plot_df_availability.parquet
    (which only has one nominal-GT2 point per track).

    Position: centroid of that gt_family's actual resolved beam crossings
    (plot_df_all_files.parquet -- built from real ATL06 geometry, not the nominal
    GT2 reference line). Completeness: actual valid_inventory acquisitions for
    that gt_family / (n_possible_cycles * 2 beams) * 100 -- same definition as
    the "What I should have vs. what I have" metrics on the Overview page.
    Rows with no resolved crossing at all (no position to plot) are dropped.
    """
    all_files = load_plot_df_all_files()
    all_files = all_files.set_geometry("cross_pt")
    all_files["gt_family"] = all_files["gt"].str[:3]

    positions = (
        all_files.groupby(["track_id", "gt_family"])["cross_pt"]
        .apply(lambda g: g.union_all().centroid)
        .rename("cross_pt")
        .reset_index()
    )

    valid_inventory = load_valid_inventory()
    actual = valid_inventory.groupby(["track_id", "gt_family"]).size().rename("n_files")
    max_possible = n_possible_cycles() * 2  # L + R beams
    completeness_pct = (actual / max_possible * 100).rename("completeness_pct")

    out = pd.DataFrame({"n_files": actual, "completeness_pct": completeness_pct}).reset_index()
    out = out.merge(positions, on=["track_id", "gt_family"], how="inner")  # inner: needs a position to plot

    bins = [0, 25, 50, 75, 100.0001]
    labels = ["<25%", "25-50%", "50-75%", "75-100%"]
    out["completeness_bin"] = pd.cut(out["completeness_pct"], bins=bins, labels=labels, right=False)

    return gpd.GeoDataFrame(out, geometry="cross_pt", crs=all_files.crs)


# Repeat-cycle date boundaries (same table as fig1_aux.ipynb's "Season coverage"
# cell) and the observation window used there to count possible repeat cycles.
# Not part of fig1_aux_outputs/ -- a small literal reference table, like
# SEA_BOUNDARY_LON and TOWNS elsewhere in this dashboard.
_CYCLE_STARTS = pd.to_datetime([
    "2018-10-13", "2018-12-28", "2019-03-29", "2019-07-09", "2019-09-26", "2019-12-26",
    "2020-03-26", "2020-06-25", "2020-09-24", "2020-12-24", "2021-03-24", "2021-06-23",
    "2021-09-22", "2021-12-22", "2022-03-23", "2022-06-21", "2022-09-20", "2022-12-20",
    "2023-03-21", "2023-06-20", "2023-09-19", "2023-12-18", "2024-03-18", "2024-06-17",
    "2024-09-16", "2024-12-16", "2025-03-16", "2025-06-15", "2025-09-14", "2025-12-14",
    "2026-03-15", "2026-06-14",
])
_WINDOW_START = pd.Timestamp("2019-01-01")
_WINDOW_END = pd.Timestamp("2025-12-31")


def n_possible_cycles() -> int:
    """Number of repeat cycles whose start falls within the 2019-01-01 - 2025-12-31 window."""
    return int(((_CYCLE_STARTS >= _WINDOW_START) & (_CYCLE_STARTS <= _WINDOW_END)).sum())


def file_mtime(filename: str):
    """Last-modified timestamp of a file in DATA_DIR, as a datetime."""
    from datetime import datetime

    return datetime.fromtimestamp((DATA_DIR / filename).stat().st_mtime)


def data_dir_exists() -> bool:
    return DATA_DIR.exists()
