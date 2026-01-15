# ICE-BEAM (ICESat-2 Bluff Erosion Assessment Method) 
A Python framework to quantify bluffs coastal change using ICESat-2 (ATL06) elevation profiles, clustering, bias filtering, and DSAS-style shoreline metrics.

IS2-BR

<p align="center">
	<img src="Book/Pictures/bluff.png" alt="Alt text">
</p>

# Purpose
The framework is a modular geospatial workflow that processes and analyzes ICESat-2 elevation data to quantify Arctic coastal retreat. It filters, clusters, and aligns elevation profiles near the shoreline to reduce spatial offset and measurement bias, enabling consistent detection of shoreline change across multiple observation years.

## What it does
- Builds oriented shoreline extraction boxes per ground-track family (gt1/gt2/gt3)
- Clips and aligns elevation profiles (offshore → inland distance)
- Clusters beams near the coast
- Applies vertical bias filtering across cycles
- Estimates shoreline/bluff positions and computes change metrics (NSM, EPR, LRR where available)

# Introduction

Importance of permafrost to climate changes…
The impact of the permafrost thaw and erosion…
How hard is to monitor it. The potential of RS …
IS-2 data as an option

#
•	Why do this?

ICESat-2 data often contain an offset that causes beams to drift by hundreds of meters, making it impossible to calculate shoreline retreat accurately without post-processing.

•	How do we address it?

Our approach aggregates nearby beams to reduce coastal feature mixing and maximize temporal coverage.

•	What do we aim to measure?

The main goal is to use reliable, post-processed ICESat-2 data to quantify coastal retreat in the Arctic region.

#

# Ideal world

Every cycle from a track pass throw out the same place on the coast that allow us to create profile where we can measure the erosion, something such as this picture below.

<p align="center">
	<img src="Book\Pictures\profile.png" alt="Alt text">
</p>

However, the reality is that the ICESat-2 track has a horizontal offset that make the use of the data almost impossible as it is (See figure below).

<p align="center">
	<img src="Book\Pictures\offsetTrack.png" alt="Alt text">
</p>

Because of that we decided to create this framework to group the beam in the smallest cluster possible. Avoiding to compare beam on different coastal feature or too far apart.

# Challenges

Download and prepare data to be used.
Create a coastline
Define AOI
Download data

<p align="center">
	<img src="Book\Pictures\tracks.png" alt="Alt text">
</p>

Link to notebooks to download data

Issues:
Bad or No data

# Workflow (Show every challenge and How I solve it)

1.	Load and Preprocess ALT06
2.	Build oriented shoreline–crossing boxes
3.	Build clusters
4.	Select optimal clusters
5.	Apply vertical bias filter
6.	Create the bluff detection
7.	Compute DSAS metrics
8.	DSAS Runner

# logos