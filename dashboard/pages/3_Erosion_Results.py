"""ICE-BEAM Dashboard -- Page 3: Erosion Results.

Map only for now -- the four pipeline-stage stat sections (direct / preprocessed /
preproc+GIE / cluster+ICE-BEAM) come in a later pass. This page reads the four
DSAS outcome-table CSVs in dashboard/data/pipeline_stages/, plus
gt_family_crossings.parquet for point positions.
"""

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium

from data_loader import DATA_DIR, HISTORICAL_DSAS_SHP, PIPELINE_STAGES_DIR, load_shoreline_utm

st.set_page_config(page_title="ICE-BEAM — Erosion Results", layout="wide")

st.title("Erosion Results")

GT_FAMILY_CROSSINGS_FP = DATA_DIR / "gt_family_crossings.parquet"

GEOMORPHIC_FEATURE_NAMES = {1: "Bluff", 2: "Delta", 3: "Rock cliff", 4: "Beach", 5: "Anthropogenic"}
GEOMORPHIC_FEATURE_COLORS = {1: "#a50026", 2: "#1a9850", 3: "#762a83", 4: "#f1a340", 5: "#4575b4"}

CATEGORY_ORDER = ["usable", "nan_at_stage", "missing_upstream", "not_in_icebeam"]
CATEGORY_COLORS = {
    "usable": "#2ECC71",
    "nan_at_stage": "#f1a340",
    "missing_upstream": "#4575b4",
    "not_in_icebeam": "#a50026",
}
KEYS = ["track_id", "gt_family"]

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


# ------------------------------------------------------------------
# Load + aggregate the four pipeline-stage CSVs (exact code/logic as given)
# ------------------------------------------------------------------
@st.cache_data
def load_pipeline_stage_aggregates():
    df0 = _read_csv(STAGE_CSVS["Direct"][0])
    df1 = _read_csv(STAGE_CSVS["Preprocessing"][0])
    df2 = _read_csv(STAGE_CSVS["Preproc+GIE"][0])
    df4 = _read_csv(STEP4_CSV)

    def aggregate_stage(df, stage_name):
        grouped = df.groupby(['track_id', 'gt_family'])
        out = grouped.agg(**{
            f'EPR_{stage_name}_median': ('EPR', 'median'),
            f'U_{stage_name}_median': ('U_EPR_myr', 'median'),
            f'n_{stage_name}': ('EPR', 'count'),
        }).reset_index()
        return out

    agg_direct = aggregate_stage(df0, 'direct')
    agg_preprocessed = aggregate_stage(df1, 'preprocessed')
    agg_preproc_gie = aggregate_stage(df2, 'preproc_gie')

    def aggregate_df4(df):
        grouped = df.groupby(['track_id', 'gt_family'])
        out = grouped.agg(
            EPR_cluster_only_median=('EPR_measured', 'median'),
            U_cluster_only_median=('U_EPR_measured_myr', 'median'),
            EPR_icebeam_median=('EPR_flagged', 'median'),
            U_icebeam_median=('U_EPR_corrected_myr', 'median'),
            n_icebeam=('EPR_flagged', 'count'),
            angle_used_median_deg_agg=('angle_used_median_deg', 'median'),
            coast_slope_sign_agg=('coast_slope_sign', 'median'),
            max_abs_gie_agg=('max_abs_gie', 'median'),
        ).reset_index()
        pct_flagged_df = df.groupby(['track_id', 'gt_family'])['flag_status'] \
            .apply(lambda s: (s == 'flagged').mean()).reset_index(name='pct_flagged')
        out = out.merge(pct_flagged_df, on=['track_id', 'gt_family'], how='left')
        return out

    agg_cluster_icebeam = aggregate_df4(df4)

    return agg_direct, agg_preprocessed, agg_preproc_gie, agg_cluster_icebeam


agg_direct, agg_preprocessed, agg_preproc_gie, agg_cluster_icebeam = load_pipeline_stage_aggregates()


@st.cache_data
def load_gt_family_crossings() -> gpd.GeoDataFrame:
    return gpd.read_parquet(GT_FAMILY_CROSSINGS_FP)


gt_family_crossings = load_gt_family_crossings()


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
# used for the cluster map and for agg_cluster_icebeam above.
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
    "(same measured-vs-flagged/corrected pairing as the map and metrics above). used_cycles_cluster "
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

# ------------------------------------------------------------------
# Key normalization -- track_id/gt_family dtypes are NOT assumed to already
# match between the DSAS tables (int64 track_id) and gt_family_crossings.parquet
# (zero-padded string track_id).
# ------------------------------------------------------------------
st.header("Key normalization check")

col_dsas, col_crossings = st.columns(2)
with col_dsas:
    st.markdown("**agg_direct** (representative of the DSAS pipeline tables)")
    st.code(
        f"track_id  dtype={agg_direct['track_id'].dtype}  sample={agg_direct['track_id'].head(5).tolist()}\n"
        f"gt_family dtype={agg_direct['gt_family'].dtype}  sample={agg_direct['gt_family'].head(5).tolist()}"
    )
with col_crossings:
    st.markdown("**gt_family_crossings.parquet**")
    st.code(
        f"track_id  dtype={gt_family_crossings['track_id'].dtype}  "
        f"sample={gt_family_crossings['track_id'].head(5).tolist()}\n"
        f"gt_family dtype={gt_family_crossings['gt_family'].dtype}  "
        f"sample={gt_family_crossings['gt_family'].head(5).tolist()}"
    )


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """track_id -> 4-digit zero-padded string, gt_family -> lowercase string."""
    df = df.copy()
    df["track_id"] = pd.to_numeric(df["track_id"]).astype(int).astype(str).str.zfill(4)
    df["gt_family"] = df["gt_family"].astype(str).str.lower()
    return df


agg_direct_n = normalize_keys(agg_direct)
agg_preprocessed_n = normalize_keys(agg_preprocessed)
agg_preproc_gie_n = normalize_keys(agg_preproc_gie)
agg_cluster_icebeam_n = normalize_keys(agg_cluster_icebeam)
gt_family_crossings_n = normalize_keys(gt_family_crossings)

direct_keys = set(zip(agg_direct_n["track_id"], agg_direct_n["gt_family"]))
crossings_keys = set(zip(gt_family_crossings_n["track_id"], gt_family_crossings_n["gt_family"]))
unmatched_direct = direct_keys - crossings_keys
unmatched_crossings = crossings_keys - direct_keys

col_a, col_b = st.columns(2)
col_a.metric(
    "agg_direct rows with no match in crossings",
    f"{len(unmatched_direct)} / {len(direct_keys)}",
)
col_b.metric(
    "gt_family_crossings rows with no match in agg_direct",
    f"{len(unmatched_crossings)} / {len(crossings_keys)}",
)
st.caption(
    "Checked after normalization (track_id -> 4-digit zero-padded string, gt_family -> lowercase). "
    "gt_family_crossings.parquet legitimately covers more (track_id, gt_family) pairs than any single "
    "DSAS stage -- it's built from the full IS2_tracks.shp x coastline intersection, not filtered to "
    "units that made it into a given pipeline stage."
)

# ------------------------------------------------------------------
# Completeness classification per (track_id, gt_family) unit
# ------------------------------------------------------------------
st.header("Completeness classification")


def _presence(df: pd.DataFrame, flag_name: str) -> pd.DataFrame:
    out = df[KEYS].drop_duplicates().copy()
    out[flag_name] = True
    return out


presence = (
    _presence(agg_direct_n, "present_direct")
    .merge(_presence(agg_preprocessed_n, "present_preprocessed"), on=KEYS, how="outer")
    .merge(_presence(agg_preproc_gie_n, "present_preproc_gie"), on=KEYS, how="outer")
    .merge(_presence(agg_cluster_icebeam_n, "present_icebeam"), on=KEYS, how="outer")
)
for flag in ["present_direct", "present_preprocessed", "present_preproc_gie", "present_icebeam"]:
    presence[flag] = presence[flag].fillna(False)

classified = (
    presence
    .merge(agg_direct_n[KEYS + ["EPR_direct_median"]], on=KEYS, how="left")
    .merge(agg_preprocessed_n[KEYS + ["EPR_preprocessed_median"]], on=KEYS, how="left")
    .merge(agg_preproc_gie_n[KEYS + ["EPR_preproc_gie_median"]], on=KEYS, how="left")
    .merge(agg_cluster_icebeam_n[KEYS + ["EPR_icebeam_median"]], on=KEYS, how="left")
)

PRE_ICEBEAM_EPR_COLS = ["EPR_direct_median", "EPR_preprocessed_median", "EPR_preproc_gie_median"]

present_all_four = (
    classified["present_direct"] & classified["present_preprocessed"]
    & classified["present_preproc_gie"] & classified["present_icebeam"]
)
null_pre_icebeam = classified[PRE_ICEBEAM_EPR_COLS].isna().any(axis=1)
null_icebeam = classified["EPR_icebeam_median"].isna()

usable = present_all_four & ~null_pre_icebeam & ~null_icebeam
# nan_at_stage per spec is "null in a pre-ICE-BEAM stage"; a unit present in all
# four stages but null ONLY in the ICE-BEAM-stage EPR (no pre-stage null) is rare
# in practice -- folded into nan_at_stage too rather than left unclassified.
nan_at_stage = present_all_four & ~usable

missing_upstream = classified["present_icebeam"] & ~(
    classified["present_direct"] & classified["present_preprocessed"] & classified["present_preproc_gie"]
)
not_in_icebeam = (
    (classified["present_direct"] | classified["present_preprocessed"] | classified["present_preproc_gie"])
    & ~classified["present_icebeam"]
)

classified["category"] = np.select(
    [usable, nan_at_stage, missing_upstream, not_in_icebeam],
    ["usable", "nan_at_stage", "missing_upstream", "not_in_icebeam"],
    default="unclassified",
)

n_icebeam_only_null = int((present_all_four & ~null_pre_icebeam & null_icebeam).sum())
if n_icebeam_only_null:
    st.caption(
        f"Note: {n_icebeam_only_null} unit(s) present in all four stages with non-null pre-ICE-BEAM EPR "
        "but a null ICE-BEAM-stage EPR -- counted under nan_at_stage above (not a separate category)."
    )

category_counts = classified["category"].value_counts().reindex(CATEGORY_ORDER + ["unclassified"]).fillna(0).astype(int)

# "Total ICE-BEAM ground-truth units" = units ICE-BEAM actually produced output
# for (present_icebeam == True) -- usable + nan_at_stage + missing_upstream.
# NOT the same as len(classified), which also includes not_in_icebeam units
# that never reached ICE-BEAM at all (present in some upstream stage only).
n_all_units = len(classified)
n_icebeam_units = int(category_counts[["usable", "nan_at_stage", "missing_upstream"]].sum())

REFERENCE = {"total": 104, "usable": 98, "missing_upstream": 5, "nan_at_stage": 1}
comparison = pd.DataFrame({
    "category": ["total (ICE-BEAM units)"] + CATEGORY_ORDER,
    "observed": [n_icebeam_units] + [int(category_counts.get(c, 0)) for c in CATEGORY_ORDER],
    "expected (~)": [REFERENCE.get(k, None) for k in ["total"] + CATEGORY_ORDER],
})
st.dataframe(comparison, hide_index=True, width="stretch")
st.caption(
    f"not_in_icebeam: {int(category_counts.get('not_in_icebeam', 0))} unit(s) present in an upstream "
    f"stage but never reached ICE-BEAM -- outside the ~104 ICE-BEAM ground-truth scope above. "
    f"Across all four stages combined, {n_all_units} distinct (track_id, gt_family) pairs appear in total."
)
if category_counts.get("unclassified", 0):
    st.warning(f"{category_counts['unclassified']} unit(s) did not fall into any of the four categories.")

# ------------------------------------------------------------------
# Join to gt_family_crossings for plotting
# ------------------------------------------------------------------
mapped = classified.merge(
    gt_family_crossings_n[KEYS + ["geometry", "cross_pt_method"]], on=KEYS, how="left",
)
mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=gt_family_crossings_n.crs)

n_no_geometry = mapped["geometry"].isna().sum()
if n_no_geometry:
    st.caption(f"{n_no_geometry} classified unit(s) have no resolved crossing point and are not shown on the map.")

plottable = mapped[mapped["geometry"].notna()].copy()
plottable_4326 = plottable.to_crs("EPSG:4326")

# ------------------------------------------------------------------
# Metrics row (live-computed)
# ------------------------------------------------------------------
n_usable = int(category_counts.get("usable", 0))
usable_pct = (n_usable / n_icebeam_units * 100) if n_icebeam_units else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("ICE-BEAM ground-truth units", f"{n_icebeam_units:,}")
col2.metric("Usable", f"{n_usable:,}")
col3.metric("Usable %", f"{usable_pct:.1f}%")

# ------------------------------------------------------------------
# Map -- same folium style as app.py: coastline always-on, colored by
# CoastType, one toggleable FeatureGroup per category, LayerControl legend.
# ------------------------------------------------------------------
st.header("Map")
st.caption(
    "One dot per (track_id, gt_family) ground-truth unit, at its gt_family's nominal-track x coastline "
    "crossing point (gt_family_crossings.parquet), colored by completeness category. Same coastline "
    "background as the Overview page. Use the layer control (top right) to toggle categories on/off."
)

if not plottable_4326.empty:
    minx, miny, maxx, maxy = plottable_4326.total_bounds
else:
    minx, miny, maxx, maxy = shoreline_4326.total_bounds

m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=6, tiles="OpenStreetMap")
m.fit_bounds([[miny, minx], [maxy, maxx]])

# Coastline drawn directly on the map (always on, not a FeatureGroup).
for ct, name in GEOMORPHIC_FEATURE_NAMES.items():
    sub = shoreline_4326[shoreline_4326["CoastType"] == ct]
    for _, row in sub.iterrows():
        locs = _line_to_folium_locations(row.geometry)
        if locs:
            folium.PolyLine(locs, color=GEOMORPHIC_FEATURE_COLORS[ct], weight=3, opacity=0.9).add_to(m)

# Drawn most-common-category first so rarer categories end up on top.
cat_counts_plottable = plottable_4326["category"].value_counts()
draw_order = sorted(CATEGORY_ORDER, key=lambda c: cat_counts_plottable.get(c, 0), reverse=True)

for category in draw_order:
    sub = plottable_4326[plottable_4326["category"] == category]
    if sub.empty:
        continue
    fg = folium.FeatureGroup(name=f"{category} (n={len(sub)})", show=True)
    for _, row in sub.iterrows():
        pt = row["geometry"]
        folium.CircleMarker(
            [pt.y, pt.x], radius=6, color="white", weight=1,
            fill=True, fill_color=CATEGORY_COLORS[category], fill_opacity=0.95,
            tooltip=f"Track {row['track_id']} ({row['gt_family']}): {category}",
        ).add_to(fg)
    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, height=550, use_container_width=True, returned_objects=[], key="erosion_results_map")
