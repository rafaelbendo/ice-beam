"""ICE-BEAM Dashboard -- Page 1: Data Characterization."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    SEA_BOUNDARY_LON,
    build_sea_map,
    load_angle_stats,
    load_coasttype_stats,
    load_cycles_per_gt,
    load_season_counts,
    load_track_dates,
    load_tracks_distance,
    load_valid_inventory,
)

st.set_page_config(page_title="ICE-BEAM — Data Characterization", layout="wide")

st.title("Data Characterization")

valid_inventory = load_valid_inventory()

# ------------------------------------------------------------------
# Track/beam counts by region (sea)
# ------------------------------------------------------------------
st.header("Track/beam counts by region")
st.caption(
    f"Split by sea (Chukchi vs Beaufort) at longitude {SEA_BOUNDARY_LON}° "
    "(Point Barrow) -- the same SEA_BOUNDARY_LON cutoff fig13.ipynb uses, "
    "applied here to each track's mean longitude (track_lon_df_cell12.parquet) "
    "rather than fig13's per-cluster geometry, which isn't part of this "
    "dashboard's data source."
)

sea_map = build_sea_map()
vi_sea = valid_inventory.copy()
vi_sea["sea"] = vi_sea["track_id"].map(sea_map)

sea_summary = (
    vi_sea.groupby("sea")
    .agg(tracks=("track_id", "nunique"), beams=("track_id", "size"))
    .reindex(["Chukchi", "Beaufort"])
    .reset_index()
)

col_track, col_beam = st.columns(2)
with col_track:
    fig_tracks = px.bar(
        sea_summary, x="sea", y="tracks", color="sea",
        labels={"sea": "Sea", "tracks": "Tracks"},
        title="Tracks per sea",
    )
    st.plotly_chart(fig_tracks, width="stretch")
with col_beam:
    fig_beams = px.bar(
        sea_summary, x="sea", y="beams", color="sea",
        labels={"sea": "Sea", "beams": "Beams"},
        title="Beams per sea",
    )
    st.plotly_chart(fig_beams, width="stretch")

# ------------------------------------------------------------------
# Beam counts by season
# ------------------------------------------------------------------
st.header("Beam counts by season")

SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
SEASON_COLORS = {"Winter": "#4C72B0", "Spring": "#55A868", "Summer": "#C44E52", "Fall": "#DD8452"}

# --- Panel 1: temporal coverage per track, split by sea ---
track_dates = load_track_dates().copy()
track_dates["sea"] = track_dates["track_id"].map(sea_map)

sea_order = ["Chukchi", "Beaufort"]  # west -> east
sea_bounds = {sea: (g["y"].min(), g["y"].max()) for sea, g in track_dates.groupby("sea")}
sea_centers = {sea: (lo + hi) / 2 for sea, (lo, hi) in sea_bounds.items()}

fig_timeline = go.Figure()
for sea in sea_order:
    sub = track_dates[track_dates["sea"] == sea]
    fig_timeline.add_trace(go.Scatter(
        x=sub["date"], y=sub["y"], mode="markers",
        marker=dict(size=4, opacity=0.5), name=sea,
    ))

if set(sea_order) <= sea_bounds.keys():
    boundary_y = (sea_bounds["Chukchi"][1] + sea_bounds["Beaufort"][0]) / 2
    fig_timeline.add_hline(y=boundary_y, line_color="gray", line_width=1)

fig_timeline.update_yaxes(
    tickmode="array",
    tickvals=[sea_centers[s] for s in sea_order],
    ticktext=sea_order,
    title="Sea (west → east)",
)
fig_timeline.update_xaxes(title="Acquisition date")
fig_timeline.update_layout(title="Temporal coverage per track, split by sea", showlegend=False, height=450)
st.plotly_chart(fig_timeline, width="stretch")

# --- Panel 2: acquisitions by season and year (2026 dropped -- partial year) ---
season_counts = load_season_counts()
season_counts_plot = season_counts[season_counts["season_year"] != 2026]

season_long = season_counts_plot.melt(
    id_vars="season_year", value_vars=SEASON_ORDER, var_name="season", value_name="count",
)
fig_season = px.bar(
    season_long, x="season_year", y="count", color="season", barmode="group",
    category_orders={"season": SEASON_ORDER}, color_discrete_map=SEASON_COLORS,
    labels={"season_year": "Year", "count": "Acquisitions"},
    title="Acquisitions by season and year",
)
st.plotly_chart(fig_season, width="stretch")

# --- Panel 3: share of total acquisitions per season, pooled across all (full) years ---
season_totals = (
    season_counts_plot[SEASON_ORDER].sum().reindex(SEASON_ORDER).reset_index()
)
season_totals.columns = ["season", "count"]
season_totals["pct"] = season_totals["count"] / season_totals["count"].sum() * 100

fig_season_total = px.bar(
    season_totals, x="season", y="pct", color="season",
    category_orders={"season": SEASON_ORDER}, color_discrete_map=SEASON_COLORS,
    labels={"season": "Season", "pct": "% of acquisitions"},
    title="Share of total acquisitions per season (2019–2025 pooled)",
    text=season_totals["pct"].map(lambda p: f"{p:.1f}%"),
)
st.plotly_chart(fig_season_total, width="stretch")

# ------------------------------------------------------------------
# Beam counts by coastal type
# ------------------------------------------------------------------
st.header("Beam counts by coastal type")
st.caption("dominant_type = the geomorphic feature with the most in-buffer points for that file.")

coasttype_stats = load_coasttype_stats()
n_total_files = len(coasttype_stats)
n_no_crossing = int(coasttype_stats["dominant_type"].isna().sum())

st.caption(
    f"{n_no_crossing:,} of {n_total_files:,} files ({n_no_crossing / n_total_files * 100:.1f}%) had no "
    "in-buffer points of any coastal type (beam line missed the 300m buffer, or its points happened to "
    "fall just outside it) and are excluded below -- shares are recomputed over the remaining "
    f"{n_total_files - n_no_crossing:,} files that hit a classified coast."
)

dominant_counts = (
    coasttype_stats["dominant_type"]
    .dropna()
    .value_counts()
    .rename_axis("coastal_type")
    .reset_index(name="count")
)
dominant_counts["pct"] = dominant_counts["count"] / dominant_counts["count"].sum() * 100

fig_coast = px.bar(
    dominant_counts, x="coastal_type", y="pct", color="coastal_type",
    labels={"coastal_type": "Dominant coastal type", "pct": "% of files (coast-hitting only)"},
    title="Files by dominant coastal type (no-crossing files excluded)",
    custom_data=["count"],
)
fig_coast.update_traces(
    hovertemplate="%{x}<br>%{y:.1f}% (%{customdata[0]:,} files)<extra></extra>"
)
st.plotly_chart(fig_coast, width="stretch")

# ------------------------------------------------------------------
# Cycles per track and per GT
# ------------------------------------------------------------------
st.header("Cycles per track and per GT")

cycles_per_gt = load_cycles_per_gt()

col_a, col_b = st.columns(2)

with col_a:
    fig_cycles_gt = px.box(
        cycles_per_gt, x="gt", y="n_cycles",
        labels={"gt": "GT beam", "n_cycles": "Cycles observed"},
        title="Cycles per (track, gt) — distribution by beam",
    )
    st.plotly_chart(fig_cycles_gt, width="stretch")

with col_b:
    # No standalone cycles_per_track.parquet was exported -- derive it here
    # from cycles_per_gt.parquet as the union of cycles seen across all of a
    # track's gt beams.
    exploded = cycles_per_gt[["track_id", "cycles_present"]].explode("cycles_present")
    cycles_per_track = exploded.groupby("track_id")["cycles_present"].nunique().reset_index(name="n_cycles")
    fig_cycles_track = px.histogram(
        cycles_per_track, x="n_cycles", nbins=20,
        labels={"n_cycles": "Distinct cycles observed"},
        title="Cycles per track (derived: union across a track's GTs)",
    )
    st.plotly_chart(fig_cycles_track, width="stretch")

GT_COLS = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
beams_per_track_gt = (
    valid_inventory.groupby(["track_id", "gt"])
    .size()
    .reset_index(name="n_beams")
    .pivot(index="track_id", columns="gt", values="n_beams")
    .reindex(columns=GT_COLS)
    .astype("Int64")  # keeps missing gt1l/gt1r for a couple of tracks as <NA> instead of 18.0-style floats
    .reset_index()
    .sort_values("track_id")
)
with st.expander(f"Full list: beams per (track, gt) — {len(beams_per_track_gt):,} tracks"):
    st.dataframe(beams_per_track_gt, width="stretch", hide_index=True)

# ------------------------------------------------------------------
# Beam-coastline angle distribution
# ------------------------------------------------------------------
st.header("Beam-coastline crossing angle")

st.warning(
    "This angle is computed independently in this QA notebook and has not "
    "yet been cross-checked against the production pipeline's "
    "angle_used_mean_deg convention. Treat as provisional until verified."
)

angle_stats = load_angle_stats()
ANGLE_MIN_DEG = 45

col1, col2, col3 = st.columns(3)
col1.metric("Mean angle", f"{angle_stats['angle_deg'].mean():.1f}°")
col2.metric("Median angle", f"{angle_stats['angle_deg'].median():.1f}°")
pct_below = (angle_stats["angle_deg"] < ANGLE_MIN_DEG).mean() * 100
col3.metric(
    f"% below {ANGLE_MIN_DEG}°",
    f"{pct_below:.1f}%",
    help=f"Would fail production angle_used_mean_deg >= {ANGLE_MIN_DEG} filter, not applied here.",
)

angle_stats["sea"] = angle_stats["track_id"].map(sea_map)

bin_edges = list(range(0, 100, 10))
bin_labels = [f"[{lo},{lo + 10})" for lo in bin_edges[:-1]]
angle_stats["angle_bin"] = pd.cut(angle_stats["angle_deg"], bins=bin_edges, right=False, labels=bin_labels)
angle_hist = angle_stats.groupby(["sea", "angle_bin"], observed=True).size().reset_index(name="count")

fig_angle = px.bar(
    angle_hist, x="angle_bin", y="count", color="sea", barmode="group",
    category_orders={"angle_bin": bin_labels, "sea": ["Chukchi", "Beaufort"]},
    labels={"angle_bin": "angle_deg (10° bins)", "count": "Files"},
    title="Beam-coastline crossing angle distribution by sea",
)
fig_angle.add_vline(x=(ANGLE_MIN_DEG / 10) - 0.5, line_dash="dash", line_color="red")
st.plotly_chart(fig_angle, width="stretch")

st.subheader("Angle by sea")
angle_by_sea = (
    angle_stats.groupby("sea")["angle_deg"]
    .agg(mean="mean", median="median", std="std", n="count")
    .reindex(["Chukchi", "Beaufort"])
    .round(2)
)
st.dataframe(angle_by_sea, width="stretch")

# ------------------------------------------------------------------
# Beam distance (across-track offset)
# ------------------------------------------------------------------
st.header("Beam distance (across-track offset)")
st.caption(
    "Lateral offset between repeat passes of the same beam, from tracks_distance.csv. "
    "Capped at 10,000 m and the x-axis zoomed to 4,500 m to match the reference figure -- "
    "a handful of far-outlier tracks extend well beyond that."
)

DISTANCE_CAP_M = 10_000
tracks_distance = load_tracks_distance()
dist_values = tracks_distance[GT_COLS].melt(value_name="distance_m")["distance_m"].dropna()
dist_values = dist_values[dist_values <= DISTANCE_CAP_M]

dist_median = dist_values.median()
dist_p90 = dist_values.quantile(0.9)

col1, col2, col3 = st.columns(3)
col1.metric("Median offset", f"{dist_median:.1f} m")
col2.metric("90th percentile", f"{dist_p90:.1f} m")
col3.metric("Observations (≤ cap)", f"{len(dist_values):,}")

bin_width = 200
fig_dist = go.Figure(go.Histogram(
    x=dist_values, xbins=dict(start=0, end=float(dist_values.max()) + bin_width, size=bin_width),
    histnorm="percent", marker_color="#7f7f7f", marker_line_color="white", marker_line_width=0.5,
))
fig_dist.add_vline(
    x=dist_median, line_dash="dash", line_color="crimson",
    annotation_text=f"Median = {dist_median:.1f} m", annotation_position="top",
)
fig_dist.add_vline(
    x=dist_p90, line_dash="dash", line_color="steelblue",
    annotation_text=f"90th percentile = {dist_p90:.1f} m", annotation_position="top",
)
fig_dist.update_layout(
    xaxis_title="Across-track offset (m)", yaxis_title="Percentage of observations (%)",
    xaxis_range=[0, 4500], title="Beam distance (across-track offset) distribution",
)
st.plotly_chart(fig_dist, width="stretch")
