"""ICE-BEAM Dashboard -- Page 3: Erosion Results.

Clusters of every pipeline stage (direct / preprocessed / preproc+GIE /
cluster-only / ICE-BEAM) on one map, plus a per-stage cluster listing. Reads the
four DSAS outcome-table CSVs in dashboard/data/pipeline_stages/.
"""

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium

from data_loader import HISTORICAL_DSAS_SHP, PIPELINE_STAGES_DIR, load_shoreline_utm

st.set_page_config(page_title="ICE-BEAM — Erosion Results", layout="wide")

st.title("Erosion Results")

GEOMORPHIC_FEATURE_NAMES = {1: "Bluff", 2: "Delta", 3: "Rock cliff", 4: "Beach", 5: "Anthropogenic"}
GEOMORPHIC_FEATURE_COLORS = {1: "#a50026", 2: "#1a9850", 3: "#762a83", 4: "#f1a340", 5: "#4575b4"}

# Individual-cluster CSVs (not aggregated to one point per track/gt_family) --
# each already carries its own center_lat/center_lon, no join needed.
STAGE_ORDER = ["Direct", "Preprocessing", "Preproc+GIE", "cluster-only", "ICE-BEAM"]
STAGE_CSVS = {
    "Direct": (str(PIPELINE_STAGES_DIR / "DSAS_Raw_Step0.csv"), "EPR", "U_EPR_myr"),
    "Preprocessing": (str(PIPELINE_STAGES_DIR / "DSAS_Raw_Step1.csv"), "EPR", "U_EPR_myr"),
    "Preproc+GIE": (str(PIPELINE_STAGES_DIR / "DSAS_Raw_GIE_Step2.csv"), "EPR", "U_EPR_myr"),
}
STEP4_CSV = str(PIPELINE_STAGES_DIR / "DSAS_metrics_flaggedStep4.csv")
CLUSTER_COLS = ["track_id", "gt_family", "cluster_id", "center_lat", "center_lon"]

# Historical (pre-ICESat-2) DSAS transect-intersection points -- the merged
# output fig13.ipynb builds from shp/1historical/DSAS/merge.zip. Optional: the
# layer is shown only when HISTORICAL_DSAS_SHP (data_loader.py) exists.
HAS_HISTORICAL_DSAS = HISTORICAL_DSAS_SHP.exists()
HISTORICAL_DSAS_COLOR = "#6a3d9a"

# EPR symbology: an up/down arrow colored by accretion vs. erosion, unless the
# EPR magnitude is smaller than its own uncertainty (not distinguishable from
# zero) or missing -- those get a plain light-gray dot instead.
EPR_ACCRETION_COLOR = "#1a9850"
EPR_EROSION_COLOR = "#d73027"
EPR_UNCERTAIN_COLOR = "#b0b0b0"
ARROW_UP = "▲"
ARROW_DOWN = "▼"


def _epr_symbol(epr, uncertainty):
    """Classify one cluster's EPR into ('within_uncertainty' | 'unknown' | 'accretion' | 'erosion')."""
    if pd.isna(epr):
        return "unknown"
    if pd.notna(uncertainty) and abs(epr) < abs(uncertainty):
        return "within_uncertainty"
    return "accretion" if epr > 0 else "erosion"


def _epr_icon(symbol: str, color: str) -> folium.DivIcon:
    html = (
        f'<div style="font-size:16px; line-height:16px; color:{color}; '
        'text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;">'
        f'{symbol}</div>'
    )
    return folium.DivIcon(html=html, icon_size=(16, 16), icon_anchor=(8, 8))


# Selecting a row in a "Cluster listings" table highlights that cluster on the
# "Clusters by pipeline stage" map above it. One shared session_state slot
# holds the most recently selected (stage, row_idx) -- row_idx is a positional
# index into that stage's cluster_layers/cluster_listings DataFrame (both are
# built from the same _read_csv() call in the same row order, so they align).
HIGHLIGHT_KEY = "erosion_highlighted_cluster"


def _make_selection_callback(stage: str, table_key: str):
    def _on_select():
        rows = st.session_state[table_key]["selection"]["rows"]
        if rows:
            st.session_state[HIGHLIGHT_KEY] = {"stage": stage, "row_idx": rows[0]}
        else:
            st.session_state.pop(HIGHLIGHT_KEY, None)

    return _on_select


@st.cache_data
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _line_to_folium_locations(geom):
    """shapely LineString/MultiLineString -> folium PolyLine locations ([lat, lon], ...)."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return [(c[1], c[0]) for c in geom.coords]
    if geom.geom_type == "MultiLineString":
        return [[(c[1], c[0]) for c in part.coords] for part in geom.geoms]
    return None


@st.cache_data
def load_historical_dsas_points() -> list:
    """[lat, lon, tooltip] rows for every historical DSAS transect-intersection
    point -- 82k+ points across 21,876 transects (1947-2017), so this returns
    plain rows for FastMarkerCluster rather than individual folium.Marker
    objects (which are far too slow/heavy to build and render at this scale)."""
    gdf = gpd.read_file(HISTORICAL_DSAS_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3338")
    gdf_4326 = gdf.to_crs("EPSG:4326")

    tooltip = (
        "Historical: " + gdf_4326["sector"].astype(str) + " " + gdf_4326["uid"].astype(str)
        + " — " + gdf_4326["decimalye0"].round(1).astype(str)
        + ": " + gdf_4326["distance"].round(1).astype(str) + " m"
    )
    return list(zip(gdf_4326.geometry.y, gdf_4326.geometry.x, tooltip))


@st.cache_data
def load_cluster_layers() -> dict:
    """One raw, un-aggregated DataFrame per stage -- every individual cluster's
    own center_lat/center_lon, plus its EPR and EPR uncertainty for symbology."""
    layers = {}
    for stage, (path, epr_col, u_col) in STAGE_CSVS.items():
        df = _read_csv(path)
        layers[stage] = df[CLUSTER_COLS + [epr_col, u_col]].rename(columns={epr_col: "EPR", u_col: "U_EPR"})

    df4 = _read_csv(STEP4_CSV)
    layers["cluster-only"] = df4[CLUSTER_COLS + ["EPR_measured", "U_EPR_measured_myr"]].rename(
        columns={"EPR_measured": "EPR", "U_EPR_measured_myr": "U_EPR"}
    )
    layers["ICE-BEAM"] = df4[CLUSTER_COLS + ["EPR_flagged", "U_EPR_corrected_myr"]].rename(
        columns={"EPR_flagged": "EPR", "U_EPR_corrected_myr": "U_EPR"}
    )

    return layers


# Per-stage source columns for the cluster listing table. Direct/Preprocessing/
# Preproc+GIE share one set of column names; cluster-only and ICE-BEAM (both
# from DSAS_metrics_flaggedStep4.csv) use the "measured" vs. "flagged"/
# "corrected" variants -- same measured/flagged-vs-corrected pairing already
# used for the cluster map above.
STAGE_LISTING_COLUMNS = {
    "Direct": {"NSM": "NSM", "EPR": "EPR", "U_EPR": "U_EPR_myr", "LRR": "LRR"},
    "Preprocessing": {"NSM": "NSM", "EPR": "EPR", "U_EPR": "U_EPR_myr", "LRR": "LRR"},
    "Preproc+GIE": {"NSM": "NSM", "EPR": "EPR", "U_EPR": "U_EPR_myr", "LRR": "LRR"},
    "cluster-only": {"NSM": "NSM_measured", "EPR": "EPR_measured", "U_EPR": "U_EPR_measured_myr", "LRR": "LRR_measured"},
    "ICE-BEAM": {"NSM": "NSM_flagged", "EPR": "EPR_flagged", "U_EPR": "U_EPR_corrected_myr", "LRR": "LRR_flagged"},
}
LISTING_COLUMNS = [
    "track_id", "gt", "cluster_id", "NSM", "EPR", "U_EPR", "LRR",
    "used_cycles_cluster", "elev_avg", "ClusterTemporalSpanYears",
]


@st.cache_data
def load_cluster_listings() -> dict:
    """One row per cluster, per stage, with only LISTING_COLUMNS.

    used_cycles_cluster and elev_avg only exist in DSAS_metrics_flaggedStep4.csv
    (cluster-only/ICE-BEAM) -- Direct/Preprocessing/Preproc+GIE don't have a
    per-stage equivalent, so those two columns come back NaN for them.
    """
    raw = {stage: _read_csv(path) for stage, (path, _e, _u) in STAGE_CSVS.items()}
    df4 = _read_csv(STEP4_CSV)
    raw["cluster-only"] = df4
    raw["ICE-BEAM"] = df4

    listings = {}
    for stage in STAGE_ORDER:
        df = raw[stage]
        cols = STAGE_LISTING_COLUMNS[stage]
        listings[stage] = pd.DataFrame({
            "track_id": df["track_id"],
            "gt": df["gt_family"],
            "cluster_id": df["cluster_id"],
            "NSM": df[cols["NSM"]],
            "EPR": df[cols["EPR"]],
            "U_EPR": df[cols["U_EPR"]],
            "LRR": df[cols["LRR"]],
            "used_cycles_cluster": df["used_cycles_cluster"] if "used_cycles_cluster" in df.columns else np.nan,
            "elev_avg": df["elev_avg"] if "elev_avg" in df.columns else np.nan,
            "ClusterTemporalSpanYears": df["ClusterTemporalSpanYears"],
        })

    return listings


shoreline_utm = load_shoreline_utm()
shoreline_4326 = shoreline_utm.to_crs("EPSG:4326")

# ------------------------------------------------------------------
# Clusters by pipeline stage -- every individual cluster's own position,
# one color per stage, all on one map so coverage/spatial shift across
# stages can be compared directly.
# ------------------------------------------------------------------
st.header("Clusters by pipeline stage")
st.caption(
    "Every individual cluster's center position (center_lat/center_lon) for each of the five pipeline "
    "outputs -- NOT aggregated to one point per (track_id, gt_family) like the completeness map further "
    "down. cluster-only and ICE-BEAM share the same cluster positions (both from DSAS_metrics_flaggedStep4"
    ".csv) but represent the uncorrected vs. GIE-corrected EPR for the same clusters. Use the layer "
    "control (top right) to toggle stages on/off."
)
st.caption(
    f"Symbol = EPR sign: {ARROW_UP} accretion, {ARROW_DOWN} erosion. A cluster whose |EPR| is smaller than "
    "its own uncertainty (U_EPR) -- not statistically distinguishable from zero -- is drawn as a plain "
    "light-gray dot instead of an arrow; a cluster with no EPR value at all gets the same gray dot."
)
if HAS_HISTORICAL_DSAS:
    st.caption(
        "Also available: **Historical DSAS** (purple, off by default) -- pre-ICESat-2 shoreline-transect "
        "intersection points from DSAS_intersection_bluff_MERGED.shp (the merged output of fig13.ipynb's "
        "merge.zip step), 1947-2017. 82k+ points across 21,876 transects, so it's rendered as a client-side "
        "marker cluster rather than individual markers -- toggle it on in the layer control, it may take a "
        "moment to load the first time."
    )
else:
    st.caption(
        "The optional **Historical DSAS** layer (pre-ICESat-2 transects, 1947-2017) is not shown: place "
        "DSAS_intersection_bluff_MERGED.shp in dashboard/data/historical_dsas/ to enable it."
    )

cluster_layers = load_cluster_layers()
n_clusters_by_stage = {stage: len(cluster_layers[stage]) for stage in STAGE_ORDER}

cols = st.columns(len(STAGE_ORDER))
for col, stage in zip(cols, STAGE_ORDER):
    col.metric(stage, f"{n_clusters_by_stage[stage]:,}")

# Resolve the current highlight (set by a table row selection further down the
# page -- session_state already holds it by the time this rerun reaches here).
highlight = st.session_state.get(HIGHLIGHT_KEY)
highlight_point = None
if highlight is not None:
    h_df = cluster_layers.get(highlight["stage"])
    h_idx = highlight["row_idx"]
    if h_df is not None and 0 <= h_idx < len(h_df):
        h_row = h_df.iloc[h_idx]
        highlight_point = {
            "lat": h_row["center_lat"], "lon": h_row["center_lon"], "stage": highlight["stage"],
            "track_id": h_row["track_id"], "gt_family": h_row["gt_family"], "cluster_id": h_row["cluster_id"],
        }
    else:
        st.session_state.pop(HIGHLIGHT_KEY, None)

if highlight_point is not None:
    st.info(
        f"Highlighted: Track {highlight_point['track_id']} ({highlight_point['gt_family']}) "
        f"cluster {highlight_point['cluster_id']} — {highlight_point['stage']}"
    )
    if st.button("Clear highlight"):
        st.session_state.pop(HIGHLIGHT_KEY, None)
        st.rerun()

all_lat = pd.concat([cluster_layers[s]["center_lat"] for s in STAGE_ORDER])
all_lon = pd.concat([cluster_layers[s]["center_lon"] for s in STAGE_ORDER])
if len(all_lat):
    c_minx, c_maxx, c_miny, c_maxy = all_lon.min(), all_lon.max(), all_lat.min(), all_lat.max()
else:
    c_minx, c_miny, c_maxx, c_maxy = shoreline_4326.total_bounds

if highlight_point is not None:
    m_clusters = folium.Map(
        location=[highlight_point["lat"], highlight_point["lon"]], zoom_start=13, tiles="OpenStreetMap",
    )
else:
    m_clusters = folium.Map(
        location=[(c_miny + c_maxy) / 2, (c_minx + c_maxx) / 2], zoom_start=6, tiles="OpenStreetMap",
    )
    m_clusters.fit_bounds([[c_miny, c_minx], [c_maxy, c_maxx]])

for ct, name in GEOMORPHIC_FEATURE_NAMES.items():
    sub = shoreline_4326[shoreline_4326["CoastType"] == ct]
    for _, row in sub.iterrows():
        locs = _line_to_folium_locations(row.geometry)
        if locs:
            folium.PolyLine(locs, color=GEOMORPHIC_FEATURE_COLORS[ct], weight=3, opacity=0.9).add_to(m_clusters)

for stage in STAGE_ORDER:
    df = cluster_layers[stage]
    fg = folium.FeatureGroup(name=f"{stage} (n={len(df)})", show=True)
    for row in df.itertuples(index=False):
        epr_txt = f"{row.EPR:.2f} m/yr" if pd.notna(row.EPR) else "n/a"
        u_txt = f"{row.U_EPR:.2f} m/yr" if pd.notna(row.U_EPR) else "n/a"
        tooltip = f"Track {row.track_id} ({row.gt_family}) cluster {row.cluster_id} — {stage}: EPR={epr_txt} (± {u_txt})"

        kind = _epr_symbol(row.EPR, row.U_EPR)
        if kind == "within_uncertainty" or kind == "unknown":
            folium.CircleMarker(
                [row.center_lat, row.center_lon], radius=4, color="white", weight=0.5,
                fill=True, fill_color=EPR_UNCERTAIN_COLOR, fill_opacity=0.85, tooltip=tooltip,
            ).add_to(fg)
        else:
            symbol = ARROW_UP if kind == "accretion" else ARROW_DOWN
            color = EPR_ACCRETION_COLOR if kind == "accretion" else EPR_EROSION_COLOR
            folium.Marker(
                [row.center_lat, row.center_lon], icon=_epr_icon(symbol, color), tooltip=tooltip,
            ).add_to(fg)
    fg.add_to(m_clusters)

if HAS_HISTORICAL_DSAS:
    historical_dsas_points = load_historical_dsas_points()
    historical_callback = f"""
    function (row) {{
        var circle = L.circleMarker(new L.LatLng(row[0], row[1]), {{
            radius: 3, color: "{HISTORICAL_DSAS_COLOR}", weight: 1, fill: true, fillOpacity: 0.7
        }});
        circle.bindTooltip(row[2]);
        return circle;
    }}
    """
    FastMarkerCluster(
        historical_dsas_points, callback=historical_callback,
        name=f"Historical DSAS (n={len(historical_dsas_points):,})", show=False,
    ).add_to(m_clusters)

if highlight_point is not None:
    # Always-on ring (not a toggleable FeatureGroup) drawn last so it sits on
    # top of every stage layer regardless of which ones are toggled off.
    folium.CircleMarker(
        [highlight_point["lat"], highlight_point["lon"]], radius=14, color="#000000", weight=3, fill=False,
        tooltip=(
            f"Selected: Track {highlight_point['track_id']} ({highlight_point['gt_family']}) "
            f"cluster {highlight_point['cluster_id']} — {highlight_point['stage']}"
        ),
    ).add_to(m_clusters)

folium.LayerControl(collapsed=False).add_to(m_clusters)

st_folium(m_clusters, height=550, use_container_width=True, returned_objects=[], key="cluster_stage_map")

# ------------------------------------------------------------------
# Cluster listings -- every cluster, per stage, as a plain table.
# ------------------------------------------------------------------
st.header("Cluster listings")
st.caption(
    "All clusters for each stage, with that stage's own NSM/EPR/U_EPR/LRR. cluster-only uses the "
    "*_measured columns, ICE-BEAM uses EPR_flagged/NSM_flagged/LRR_flagged with U_EPR_corrected_myr "
    "(same measured-vs-flagged/corrected pairing as the map above). used_cycles_cluster "
    "and elev_avg only exist in DSAS_metrics_flaggedStep4.csv -- Direct/Preprocessing/Preproc+GIE show "
    "them as empty."
)
st.caption("Select a row to highlight that cluster on the “Clusters by pipeline stage” map above.")

cluster_listings = load_cluster_listings()
listing_tabs = st.tabs(STAGE_ORDER)
for tab, stage in zip(listing_tabs, STAGE_ORDER):
    with tab:
        listing = cluster_listings[stage]
        st.caption(f"{len(listing):,} clusters")
        table_key = f"cluster_listing_{stage}"
        st.dataframe(
            listing[LISTING_COLUMNS], hide_index=True, width="stretch",
            on_select=_make_selection_callback(stage, table_key),
            selection_mode="single-row",
            key=table_key,
        )
