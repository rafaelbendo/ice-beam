# Dashboard data

Read-only inputs for the Streamlit dashboard (`dashboard/app.py`).

| Files | Source |
|---|---|
| `*.parquet`, `tracks_distance.csv` | Exports of the data-characterization notebook (`fig1_aux.ipynb`): beam inventory, dates, coastal-type and angle statistics, QC statistics, shoreline and track geometry |
| `pipeline_stages/DSAS_Raw_Step0.csv` | DSAS metrics, raw ATL06 pipeline, no preprocessing ("Direct") |
| `pipeline_stages/DSAS_Raw_Step1.csv` | Raw ATL06 pipeline with preprocessing ("Preprocessing") |
| `pipeline_stages/DSAS_Raw_GIE_Step2.csv` | Raw ATL06 pipeline with preprocessing and GIE correction ("Preproc+GIE") |
| `pipeline_stages/DSAS_metrics_flaggedStep4.csv` | ICE-BEAM cluster results (180 m size limit) after post-processing flags ("cluster-only" and "ICE-BEAM") |

Optional, not included: `historical_dsas/DSAS_intersection_bluff_MERGED.shp` (historical
DSAS transect-intersection points, 1947–2017). When present, the Erosion Results page adds a
"Historical DSAS" map layer.
