"""ICE-BEAM Dashboard -- Page 2: QC / Filtering."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    file_mtime,
    load_coasttype_stats,
    load_qc_stats,
    n_possible_cycles,
)

st.set_page_config(page_title="ICE-BEAM — QC / Filtering", layout="wide")

st.title("QC / Filtering")

qc_stats_all = load_qc_stats()
mtime = file_mtime("qc_stats_cell24.parquet")
st.caption(f"Source: `qc_stats_cell24.parquet` — last modified **{mtime:%Y-%m-%d %H:%M:%S}**")

HAS_INSUFFICIENT_INTERSECTION = "insufficient_intersection" in qc_stats_all.columns

# ------------------------------------------------------------------
# Step 0: filter to files that cross Bluff -- this is the QC population for
# every section below, not just the funnels at the bottom.
# ------------------------------------------------------------------
st.header("Bluff filter")
st.caption(
    "First step: keep only files (one beam on one date) that cross Bluff coast, i.e. have any "
    "points on a Bluff segment (coasttype_stats.parquet, crosses_Bluff), even if another coastal "
    "type has more points. Every section below uses this filtered population as its starting "
    "point, not the full file set."
)

coasttype_stats = load_coasttype_stats()
qc_stats = qc_stats_all.merge(
    coasttype_stats[["track_id", "gt", "date", "crosses_Bluff", "dominant_type"]],
    on=["track_id", "gt", "date"], how="left",
)
n_all_files = len(qc_stats_all)
qc_stats = qc_stats[qc_stats["crosses_Bluff"].eq(True)]
n_bluff_files = len(qc_stats)
n_not_bluff = n_all_files - n_bluff_files

col1, col2, col3 = st.columns(3)
col1.metric("Total files", f"{n_all_files:,}")
col2.metric("Cross Bluff", f"{n_bluff_files:,}", f"{n_bluff_files / n_all_files * 100:.1f}%")
col3.metric("Don't cross Bluff", f"{n_not_bluff:,}", f"{n_not_bluff / n_all_files * 100:.1f}%", delta_color="inverse")

fig_bluff = px.bar(
    x=["Cross Bluff", "Don't cross Bluff"], y=[n_bluff_files, n_not_bluff],
    labels={"x": "File", "y": "Files"},
    title="Files crossing Bluff vs. not",
)
st.plotly_chart(fig_bluff, width="stretch")

qc_stats["gt_family"] = qc_stats["gt"].str[:3]
tracks_cross_bluff_by_family = qc_stats.groupby("gt_family")["track_id"].nunique()
n_tracks_cross_bluff = qc_stats["track_id"].nunique()

col_gt1, col_gt2, col_gt3, col_total = st.columns(4)
for col, family in zip([col_gt1, col_gt2, col_gt3], ["gt1", "gt2", "gt3"]):
    col.metric(f"{family.upper()} crosses Bluff", f"{int(tracks_cross_bluff_by_family.get(family, 0)):,} tracks")
col_total.metric("Total tracks crossing Bluff", f"{n_tracks_cross_bluff:,}")

# ------------------------------------------------------------------
# Elevation anomaly (elev_trash)
# ------------------------------------------------------------------
st.header("Elevation anomaly (elev_trash)")

n_total = len(qc_stats)
n_elev_trash = int(qc_stats["elev_trash"].sum())
qc_after_elev = qc_stats[~qc_stats["elev_trash"]]
n_after_elev = len(qc_after_elev)
col1, col2 = st.columns(2)
col1.metric("Files flagged elev_trash", f"{n_elev_trash:,}", f"{n_elev_trash / n_total * 100:.1f}% of {n_total:,}")
col2.metric("Files clean", f"{n_after_elev:,}", f"{n_after_elev / n_total * 100:.1f}%")

good_elev_per_track = (
    qc_stats.groupby("track_id")["elev_trash"]
    .apply(lambda s: (~s).mean() * 100)
    .rename("good_elev_pct")
    .reset_index()
    .sort_values("good_elev_pct")
)
fig_elev = px.bar(
    good_elev_per_track, x="track_id", y="good_elev_pct",
    labels={"track_id": "Track", "good_elev_pct": "% Bluff files without elev_trash"},
    title="Per-track elevation QC pass rate, Bluff files only (sorted ascending, worst first)",
)
fig_elev.update_xaxes(type="category")
st.plotly_chart(fig_elev, width="stretch")

per_gt_elev_trash = (
    qc_stats.groupby(["track_id", "gt"])["elev_trash"].sum().rename("n").reset_index()
)
elev_trash_list = (
    per_gt_elev_trash.groupby("track_id")
    .agg(
        n_elev_trash=("n", "sum"),
        max_elev_trash=("n", "max"),
        worst_gt=("n", lambda s: per_gt_elev_trash.loc[s.idxmax(), "gt"]),
    )
    .reset_index()
    .sort_values("n_elev_trash", ascending=False)
)
with st.expander(f"Per-track elev_trash detail — {len(elev_trash_list):,} tracks with a Bluff file"):
    st.caption(
        "max_elev_trash = the highest elev_trash count among that track's individual gt beams "
        "(worst_gt names which one); n_elev_trash = total across all of that track's beams."
    )
    st.dataframe(elev_trash_list, width="stretch", hide_index=True)

# ------------------------------------------------------------------
# Density pass/fail
# ------------------------------------------------------------------
st.header("Density pass/fail")
st.caption(
    f"Starts from the {n_after_elev:,} files that are NOT flagged elev_trash (the elevation "
    "step above runs first now)."
)

if not HAS_INSUFFICIENT_INTERSECTION:
    st.warning(
        "qc_stats_cell24.parquet has no `insufficient_intersection` column -- "
        "this export predates the fencepost fix (few_points < 10 expected_points "
        "excluded from the density check). The QC data below may be pre-fix. "
        "Re-run the notebook's export cell to refresh it."
    )

n_fail = int(qc_after_elev["few_points"].sum())
n_pass = n_after_elev - n_fail

col1, col2, col3 = st.columns(3)
col1.metric("Total files (elev-clean, Bluff)", f"{n_after_elev:,}")
col2.metric("Density pass", f"{n_pass:,}", f"{n_pass / n_after_elev * 100:.1f}%")
col3.metric("Density fail (few_points)", f"{n_fail:,}", f"{n_fail / n_after_elev * 100:.1f}%", delta_color="inverse")

fig_density = px.bar(
    x=["Pass", "Fail"], y=[n_pass, n_fail],
    labels={"x": "Density check", "y": "Files"},
    title="Density pass/fail counts",
)
st.plotly_chart(fig_density, width="stretch")

GT_COLS = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
density_removed_by_gt = (
    qc_after_elev[qc_after_elev["few_points"]]
    .groupby(["track_id", "gt"])
    .size()
    .reset_index(name="n_removed")
    .pivot(index="track_id", columns="gt", values="n_removed")
    .reindex(columns=GT_COLS)
    .fillna(0)
    .astype(int)
    .reset_index()
)
density_removed_by_gt["total_removed"] = density_removed_by_gt[GT_COLS].sum(axis=1)
density_removed_by_gt = density_removed_by_gt.sort_values("total_removed", ascending=False)

with st.expander(f"Files removed by density check, per (track, gt) — {len(density_removed_by_gt):,} tracks"):
    st.caption("Count of files failing the density check (few_points), out of the elev-clean Bluff population.")
    st.dataframe(density_removed_by_gt, width="stretch", hide_index=True)

# ------------------------------------------------------------------
# Per-track density pass rate
# ------------------------------------------------------------------
st.header("Per-track density pass rate")

good_density_per_track = (
    qc_after_elev.groupby("track_id")["few_points"]
    .apply(lambda s: (~s).mean() * 100)
    .rename("good_density_pct")
    .reset_index()
    .sort_values("good_density_pct")
)
fig_good_density = px.bar(
    good_density_per_track, x="track_id", y="good_density_pct",
    labels={"track_id": "Track", "good_density_pct": "% files passing density check"},
    title="Per-track density pass rate, elev-clean Bluff files (sorted ascending, worst first)",
)
fig_good_density.update_xaxes(type="category")
st.plotly_chart(fig_good_density, width="stretch")

with st.expander("Distribution (histogram)"):
    fig_hist = px.histogram(
        good_density_per_track, x="good_density_pct",
        labels={"good_density_pct": "% files passing density check"},
        title="Per-track density pass rate — distribution across tracks",
    )
    st.plotly_chart(fig_hist, width="stretch")

# ------------------------------------------------------------------
# Filter funnel
# ------------------------------------------------------------------
st.header("Filter funnel")

if not HAS_INSUFFICIENT_INTERSECTION:
    st.caption(
        "The insufficient_intersection exclusion step isn't shown -- that "
        "column isn't in this qc_stats_cell24.parquet export yet (see the "
        "warning above). Funnel below uses only the density (few_points) and "
        "elevation (elev_trash) filters that ARE present."
    )

after_density = qc_after_elev.loc[~qc_after_elev["few_points"]]

# "What I should have" -- same possible-beams definition as the Overview page's
# funnel: tracks hitting ANY coastal type (not just Bluff) x 2 beams x possible cycles.
coasttype_stats_all = load_coasttype_stats()
tracks_hit_by_family_all = (
    coasttype_stats_all.dropna(subset=["dominant_type"])
    .assign(gt_family=coasttype_stats_all["gt"].str[:3])
    .groupby("gt_family")["track_id"]
    .nunique()
)
n_possible_total = int(tracks_hit_by_family_all.sum() * 2 * n_possible_cycles())

funnel_stages = [
    "What I should have", "All files", "Bluff files (cross Bluff)", "After elevation filter", "After density filter (remaining)",
]
funnel_counts = [n_possible_total, n_all_files, n_bluff_files, n_after_elev, len(after_density)]

col_a, col_b = st.columns([1, 2])
with col_a:
    st.dataframe(
        {"Stage": funnel_stages, "Files remaining": funnel_counts},
        hide_index=True, width="stretch",
    )
with col_b:
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_stages, x=funnel_counts, textinfo="value+percent initial",
    ))
    fig_funnel.update_layout(title="QC filter funnel")
    st.plotly_chart(fig_funnel, width="stretch")

# ------------------------------------------------------------------
# Bluff filter funnel -- Bluff-only from the start
# ------------------------------------------------------------------
st.header("Bluff filter funnel")
st.caption(
    "Bluff-only from the start. **What I should have** = beams that pass a Bluff feature "
    "(each (track, beam) pair, gt1l … gt3r, that crosses Bluff in at least one file) × "
    f"{n_possible_cycles()} possible repeat cycles (2019-01-01 to 2025-12-31). **Files on Bluff** = "
    "the actual files that cross Bluff (same population as the Bluff filter above); the "
    "elevation and density filters then run on those files."
)

bluff_beams = (
    coasttype_stats.loc[coasttype_stats["crosses_Bluff"].eq(True), ["track_id", "gt"]]
    .drop_duplicates()
    .assign(gt_family=lambda d: d["gt"].str[:3])
)
n_bluff_should_have = len(bluff_beams) * n_possible_cycles()

bluff_funnel_stages = [
    "What I should have",
    "Files on Bluff",
    "After elevation filter",
    "After density filter (remaining)",
]
bluff_funnel_counts = [n_bluff_should_have, n_bluff_files, n_after_elev, len(after_density)]

col1, col2, col3 = st.columns(3)
col1.metric("Beams crossing Bluff", f"{len(bluff_beams):,}")
col2.metric("Tracks with a Bluff-crossing beam", f"{bluff_beams['track_id'].nunique():,}")
col3.metric(
    "Remaining vs. should have", f"{len(after_density):,} / {n_bluff_should_have:,}",
    f"{len(after_density) / n_bluff_should_have * 100:.0f}%" if n_bluff_should_have else None,
    delta_color="off",
)

col_a, col_b = st.columns([1, 2])
with col_a:
    st.dataframe(
        {"Stage": bluff_funnel_stages, "Files remaining": bluff_funnel_counts},
        hide_index=True, width="stretch",
    )
with col_b:
    fig_bluff_funnel = go.Figure(go.Funnel(
        y=bluff_funnel_stages, x=bluff_funnel_counts, textinfo="value+percent initial",
    ))
    fig_bluff_funnel.update_layout(title="Bluff filter funnel")
    st.plotly_chart(fig_bluff_funnel, width="stretch")

by_family = (
    bluff_beams.groupby("gt_family").size().mul(n_possible_cycles()).rename("What I should have").to_frame()
    .join(qc_stats.groupby("gt_family").size().rename("Files on Bluff"))
    .join(qc_after_elev.groupby("gt_family").size().rename("After elevation filter"))
    .join(after_density.groupby("gt_family").size().rename("After density filter"))
    .fillna(0)
    .astype(int)
)
by_family["Remaining %"] = (by_family["After density filter"] / by_family["What I should have"] * 100).round(1)
with st.expander("Bluff filter funnel by GT family"):
    st.dataframe(by_family.reset_index().rename(columns={"gt_family": "GT family"}), hide_index=True, width="stretch")
