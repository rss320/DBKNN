
# DB-KNN

The hybrid DB-KNN algorithm from: Stroud, Ryan S., et al. "Testing Outlier Detection Algorithms for Identifying Early Stage Solute Clusters in Atom Probe Tomography." Microscopy and Microanalysis 30.5 (2024): 853-865.

# Usage

## Install

pip install git+https://github.com/rss320/DBKNN.git

or if using uv (https://docs.astral.sh/uv/),

uv add git+https://github.com/rss320/DBKNN.git

or as a script,

```python
# /// script
# dependencies = [
#   "DBKNN @ git+https://github.com/rss320/DBKNN.git",
# ]
# ///
```

## XYZ Example

```python
from dbknn import DBKNN
from dbknn.io import load_apt_data, save_labeled_xyz
from dbknn.plotting import plot_results_3d, plot_scores

solute_pos, all_pos, ground_truth = load_apt_data(
    "dump.xyz",
    solute_species=["Ni", "Mn", "Cu"],  # or matrix_species=["Fe"]
)

model = DBKNN(eps=8.0, min_samples=15, k=10)
model.fit(solute_pos)

plot_scores(model.hybrid_scores_, model.threshold_, output_path="scores.png")
plot_results_3d(solute_pos, model.labels_, output_path="results_3d.png")
save_labeled_xyz(
    "dump.xyz", "labeled.xyz",
    solute_species=["Ni", "Mn", "Cu"],
    solute_labels=model.labels_,
    solute_scores=model.hybrid_scores_,
)
```

## CSV Example

APT CSV files with columns `[x, y, z, mc]` (positions in nm, mass-to-charge ratio) are also supported. Species are assigned via a dictionary of mass-to-charge ranges:

```python
from dbknn import DBKNN
from dbknn.io import load_apt_csv, save_labeled_csv, save_csv_as_xyz

mc_ranges = {
    "Fe": [(23.30, 24.50), (26.50, 28.50), (53.50, 56.90)],
    "Ni": [(28.50, 30.60), (57.50, 60.60)],
    "Mn": [(26.50, 28.50), (54.00, 55.50)],
    "Cu": [(31.00, 32.50), (63.00, 65.50)],
}

solute_pos, all_pos, _ = load_apt_csv(
    "apt_data.csv", mc_ranges, matrix_species=["Fe"],
)

model = DBKNN(eps=8.0, min_samples=15, k=10)
model.fit(solute_pos)

save_labeled_csv("apt_data.csv", "labeled.csv", mc_ranges, matrix_species=["Fe"],
                 solute_labels=model.labels_, solute_scores=model.hybrid_scores_)
save_csv_as_xyz("apt_data.csv", "labeled.xyz", mc_ranges, matrix_species=["Fe"],
                solute_labels=model.labels_, solute_scores=model.hybrid_scores_)
```

## Manual Threshold

By default KARCH auto-thresholding is used. To set a manual hybrid-score cutoff (atoms with score <= threshold are classified as cluster):

```python
from dbknn import DBKNN

model = DBKNN(eps=8.0, min_samples=15, k=10, threshold=5.0)
model.fit(solute_pos)
```

## Custom DBSCAN Multiplier Weights

The hybrid score is `kNN_distance * multiplier`, where the multiplier depends on DBSCAN cluster membership. The defaults are 0.5 for cluster atoms and 1.5 for noise:

```python
from dbknn import DBKNN

model = DBKNN(eps=8.0, min_samples=15, k=10, cluster_weight=0.3, noise_weight=2.0)
model.fit(solute_pos)
```

## Cluster Analysis

Per-cluster volume (convex hull) and composition statistics. Requires a species label array matching each solute atom:

```python
import numpy as np
from dbknn import DBKNN
from dbknn.analysis import compute_cluster_stats, save_cluster_stats_csv
from dbknn.io import load_apt_data
from dbknn.plotting import plot_cluster_stats
from ase.io import read as ase_read

filepath = "dump.xyz"
solute_species = ["Ni", "Mn", "Cu"]

solute_pos, _, _ = load_apt_data(filepath, solute_species=solute_species)

atoms = ase_read(filepath, format="extxyz")
all_symbols = np.array(atoms.get_chemical_symbols())
species = all_symbols[np.isin(all_symbols, solute_species)]

model = DBKNN(eps=8.0, min_samples=15, k=10)
model.fit(solute_pos)

stats = compute_cluster_stats(
    solute_pos, model.dbscan_labels_, model.labels_, species=species,
)
save_cluster_stats_csv(stats, output="cluster_stats.csv")
plot_cluster_stats(stats, output_path="cluster_stats.png")
```

The CSV contains per-cluster rows with atom count, convex hull volume, and composition (at.%), plus a summary with total volume fraction and mean composition. The plot shows a cluster volume histogram and mean composition bar chart.

## Evaluations

Also included is a series of tools to evaluate the algorithm against a dataset with ground truth labels as in the original paper. See `examples/examples.ipynb` for worked examples.

## Visualisation

Plotting functions have been included. However, the recommended way to visualise the clusters is to output as an xyz file and import into OVITO (https://www.ovito.org/)

# About

This repository provides an easy-to-use implementation of the DB-KNN algorithm described in the paper: Stroud, Ryan S., et al. "Testing Outlier Detection Algorithms for Identifying Early Stage Solute Clusters in Atom Probe Tomography." Microscopy and Microanalysis 30.5 (2024): 853-865.
