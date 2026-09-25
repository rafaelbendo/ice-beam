"""ICE-BEAM Dashboard -- Page 0: Overview.

Run from the repository root:  streamlit run dashboard/app.py

Reads only from dashboard/data/ (see data_loader.py).
"""

import folium
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from data_loader import (
    DATA_DIR,
    build_gt_family_availability,
    data_dir_exists,
    load_coasttype_stats,
    load_shoreline_utm,
    load_track_dates,
    load_tracks_utm,
    load_valid_inventory,
    n_possible_cycles,
)

TRACK_COLOR = "#2ECC71"
BLUFF_TRACK_COLOR = "#a50026"
TRACK_CLIP_BUFFER_M = 100_000  # keep only the part of each track within 100km of the coastline
GEOMORPHIC_FEATURE_NAMES = {1: "Bluff", 2: "Delta", 3: "Rock cliff", 4: "Beach", 5: "Anthropogenic"}
GEOMORPHIC_FEATURE_COLORS = {1: "#a50026", 2: "#1a9850", 3: "#762a83", 4: "#f1a340", 5: "#4575b4"}
AVAILABILITY_COLORS = {"<25%": "#440154", "25-50%": "#31688e", "50-75%": "#35b779", "75-100%": "#fde725"}
AVAILABILITY_ORDER = ["<25%", "25-50%", "50-75%", "75-100%"]


def _line_to_folium_locations(geom):
    """shapely LineString/MultiLineString -> folium PolyLine locations ([lat, lon], ...).

    Some track lines carry a Z coordinate (has_z), so coords come as (x, y, z)
    tuples -- index instead of unpacking to (lon, lat) so both 2D and 3D
    geometries work.
    """
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return [(c[1], c[0]) for c in geom.coords]
    if geom.geom_type == "MultiLineString":
        return [[(c[1], c[0]) for c in part.coords] for part in geom.geoms]
    return None

st.set_page_config(page_title="ICE-BEAM Dashboard", page_icon="\U0001F9CA", layout="wide")

st.title("ICE-BEAM — Overview")
st.caption(
    "Arctic coastal bluff/shoreline erosion pipeline: ICESat-2 ATL06 data characterization and QC. "
    "Interactive companion to the paper figures."
)

if not data_dir_exists():
    st.error(f"Data folder not found: {DATA_DIR}")
    st.stop()

valid_inventory = load_valid_inventory()
track_dates = load_track_dates()
coasttype_stats = load_coasttype_stats()
coasttype_stats["gt_family"] = coasttype_stats["gt"].str[:3]

n_tracks_total_coast = coasttype_stats["track_id"].nunique()
tracks_hit_by_family = (
    coasttype_stats.dropna(subset=["dominant_type"]).groupby("gt_family")["track_id"].nunique()
)
actual_by_family = valid_inventory.groupby("gt_family").size()

N_CYCLES = n_possible_cycles()
GT_FAMILIES = ["gt1", "gt2", "gt3"]
possible_beams = {family: int(tracks_hit_by_family.get(family, 0)) * 2 * N_CYCLES for family in GT_FAMILIES}
actual_beams = {family: int(actual_by_family.get(family, 0)) for family in GT_FAMILIES}
possible_total = sum(possible_beams.values())
actual_total = sum(actual_beams.values())

# ------------------------------------------------------------------
# What we should have vs. what we have
# ------------------------------------------------------------------
st.header("📉 What I should have vs. what I have")
st.caption(
    f"Possible = tracks hitting coast × 2 beams (L/R) × {N_CYCLES} possible repeat cycles "
    "(2019-01-01 to 2025-12-31 window, same as fig1_aux.ipynb's completeness cell). "
    "Actual = every beam-pass file actually in valid_inventory.parquet, regardless of "
    "whether that track hits the coast."
)

fig_funnel = go.Figure(go.Funnel(
    y=["What I should have", "What I have"],
    x=[possible_total, actual_total],
    textinfo="value+percent initial",
))
fig_funnel.update_layout(title="Overall (GT1 + GT2 + GT3)", height=350)
st.plotly_chart(fig_funnel, width="stretch")

cols = st.columns(4)
for col, family in zip(cols[:3], GT_FAMILIES):
    pct = actual_beams[family] / possible_beams[family] * 100 if possible_beams[family] else 0
    col.metric(f"{family.upper()}", f"{actual_beams[family]:,} / {possible_beams[family]:,}", f"{pct:.0f}%")
overall_pct = actual_total / possible_total * 100 if possible_total else 0
cols[3].metric("Overall", f"{actual_total:,} / {possible_total:,}", f"{overall_pct:.0f}%")

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Total tracks", f"{valid_inventory['track_id'].nunique():,}")
col2.metric("Total beams (rows)", f"{len(valid_inventory):,}")
col3.metric("Date range", f"{track_dates['date'].min().date()} → {track_dates['date'].max().date()}")

st.divider()

st.header("GT coastline crossings")
st.caption(
    "tracks_utm.parquet only carries the nominal GT2 reference line per track (199 rows, "
    "all gt == 'GT2') -- not real gt1/gt2/gt3 beam geometry. Using coasttype_stats.parquet "
    "instead, which has actual per-beam crossings (gt1l, gt1r, gt2l, gt2r, gt3l, gt3r) from "
    "real ATL06 beam geometry. A track counts for a GT family if either its L or R beam had "
    "at least one file with a classified coastal-type crossing."
)

col_gt1, col_gt2, col_gt3 = st.columns(3)
for col, family in zip([col_gt1, col_gt2, col_gt3], GT_FAMILIES):
    n = int(tracks_hit_by_family.get(family, 0))
    col.metric(f"{family.upper()} hits coast", f"{n:,} / {n_tracks_total_coast:,}")

st.header("Map")
st.caption(
    "Coastline colored by geomorphic type (always shown, not in the layer control). "
    "ICESat-2 tracks (bluff-crossing tracks highlighted) clipped to within "
    f"{TRACK_CLIP_BUFFER_M // 1000} km of the coastline -- raw tracks_utm lines run the full "
    "satellite ground-track length (up to ~3,700 km), way beyond the North Slope. Use the "
    "layer control (top right of the map) to toggle the two track layers on/off."
)

shoreline_utm = load_shoreline_utm()
coastline_union = shoreline_utm.geometry.union_all()
coast_buffer = coastline_union.buffer(TRACK_CLIP_BUFFER_M)

tracks_utm = load_tracks_utm()
tracks_utm = tracks_utm.set_geometry(tracks_utm.geometry.intersection(coast_buffer))
tracks_utm = tracks_utm[~tracks_utm.geometry.is_empty]
tracks_4326 = tracks_utm.to_crs("EPSG:4326")
shoreline_4326 = shoreline_utm.to_crs("EPSG:4326")

minx, miny, maxx, maxy = tracks_4326.total_bounds
m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=6, tiles="OpenStreetMap")
m.fit_bounds([[miny, minx], [maxy, maxx]])

# Coastline drawn directly on the map (always on) -- not a FeatureGroup, so it
# never shows up as a checkbox in the layer control. Only the two track
# groups below are toggleable.
for ct, name in GEOMORPHIC_FEATURE_NAMES.items():
    sub = shoreline_4326[shoreline_4326["CoastType"] == ct]
    for _, row in sub.iterrows():
        locs = _line_to_folium_locations(row.geometry)
        if locs:
            folium.PolyLine(locs, color=GEOMORPHIC_FEATURE_COLORS[ct], weight=3, opacity=0.9).add_to(m)

bluff_tracks = tracks_4326[tracks_4326["crosses_bluff"]]
other_tracks = tracks_4326[~tracks_4326["crosses_bluff"]]

fg_bluff = folium.FeatureGroup(name="IS2 tracks (bluff-crossing)", show=True)
for _, row in bluff_tracks.iterrows():
    locs = _line_to_folium_locations(row.geometry)
    if locs:
        folium.PolyLine(locs, color=BLUFF_TRACK_COLOR, weight=1.5, tooltip=f"Track {row['track_id']}").add_to(fg_bluff)
fg_bluff.add_to(m)

fg_other = folium.FeatureGroup(name="IS2 tracks (other)", show=True)
for _, row in other_tracks.iterrows():
    locs = _line_to_folium_locations(row.geometry)
    if locs:
        folium.PolyLine(locs, color=TRACK_COLOR, weight=1, tooltip=f"Track {row['track_id']}").add_to(fg_other)
fg_other.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, height=550, use_container_width=True, returned_objects=[])

st.divider()

# ------------------------------------------------------------------
# Data availability map (separate from the tracks map above)
# ------------------------------------------------------------------
st.header("Data availability map")
st.caption(
    "One dot per (track, GT family) -- gt1/gt2/gt3, each combining its L+R beam -- at that family's "
    "real coastline crossing (centroid of its resolved beam crossings in plot_df_all_files.parquet), "
    "colored by completeness bin. Same coastline background as the map above. Use the layer control "
    "to toggle each completeness bin on/off."
)

availability = build_gt_family_availability().to_crs("EPSG:4326")

m2 = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=6, tiles="OpenStreetMap")
m2.fit_bounds([[miny, minx], [maxy, maxx]])

for ct, name in GEOMORPHIC_FEATURE_NAMES.items():
    sub = shoreline_4326[shoreline_4326["CoastType"] == ct]
    for _, row in sub.iterrows():
        locs = _line_to_folium_locations(row.geometry)
        if locs:
            folium.PolyLine(locs, color=GEOMORPHIC_FEATURE_COLORS[ct], weight=3, opacity=0.9).add_to(m2)

# Drawn most-common-bin first so the rarest bin ends up on top (same
# draw-order logic as the fig1a.ipynb reference).
bin_counts = availability["completeness_bin"].value_counts()
draw_order = sorted(AVAILABILITY_ORDER, key=lambda b: bin_counts.get(b, 0), reverse=True)

for label in draw_order:
    sub = availability[availability["completeness_bin"] == label]
    if sub.empty:
        continue
    fg = folium.FeatureGroup(name=f"Data availability: {label}", show=True)
    for _, row in sub.iterrows():
        pt = row["cross_pt"]
        folium.CircleMarker(
            [pt.y, pt.x], radius=6, color="white", weight=1,
            fill=True, fill_color=AVAILABILITY_COLORS[label], fill_opacity=0.95,
            tooltip=(
                f"Track {row['track_id']} ({row['gt_family']}): "
                f"{row['completeness_pct']:.1f}% ({row['n_files']} files)"
            ),
        ).add_to(fg)
    fg.add_to(m2)

folium.LayerControl(collapsed=False).add_to(m2)

st_folium(m2, height=550, use_container_width=True, returned_objects=[], key="availability_map")

st.subheader("Data availability distribution")
median_completeness = availability["completeness_pct"].median()
fig_avail_hist = go.Figure(go.Histogram(
    x=availability["completeness_pct"], nbinsx=15,
    marker_color="#3987e5", marker_line_color="white", marker_line_width=0.5,
))
fig_avail_hist.add_vline(
    x=median_completeness, line_dash="dash", line_color="#0d366b",
    annotation_text=f"median {median_completeness:.1f}%", annotation_position="top",
)
fig_avail_hist.update_layout(
    xaxis_title="Data availability (%)", yaxis_title="(Track, GT family) pairs", height=350,
    title=f"Completeness across {len(availability):,} (track, GT family) pairs with a resolved coastline crossing",
)
st.plotly_chart(fig_avail_hist, width="stretch")

st.divider()
st.caption(f"Data source: `{DATA_DIR}`")
