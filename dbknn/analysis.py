"""Cluster analysis: volume fractions and compositions via convex hulls."""

import csv
import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

log = logging.getLogger(__name__)


def _safe_hull_volume(points: NDArray[np.floating]) -> float:
    """Convex hull volume, returning NaN if fewer than 4 non-coplanar points."""
    if len(points) < 4:
        return float("nan")
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return float("nan")


def compute_cluster_stats(
    positions: NDArray[np.floating],
    dbscan_labels: NDArray[np.integer],
    cluster_mask: NDArray[np.integer],
    species: Optional[NDArray[np.str_]] = None,
) -> dict[str, object]:
    """Compute per-cluster volume (convex hull) and composition.

    Parameters
    ----------
    positions : (n_solutes, 3) array
        Solute atom positions (angstrom).
    dbscan_labels : (n_solutes,) array
        DBSCAN cluster IDs (-1 = noise, >=0 = cluster).
    cluster_mask : (n_solutes,) array
        Binary mask: 1 = cluster atom (passed threshold), 0 = matrix.
    species : (n_solutes,) array of str, optional
        Species label for each solute atom.  Required for composition.

    Returns
    -------
    dict with keys:
        clusters, total_volume, cluster_volume_sum, volume_fraction,
        mean_composition, species_list, n_noise_cluster_atoms.
    """
    mask = cluster_mask.astype(bool)

    total_volume = _safe_hull_volume(positions)

    cluster_ids = np.unique(dbscan_labels[mask])
    cluster_ids = cluster_ids[cluster_ids >= 0]

    species_list: list[str] = sorted(set(species.tolist())) if species is not None else []

    clusters: list[dict[str, object]] = []
    for cid in cluster_ids:
        in_cluster = mask & (dbscan_labels == cid)
        n_atoms = int(in_cluster.sum())
        if n_atoms == 0:
            continue

        vol = _safe_hull_volume(positions[in_cluster])

        comp: dict[str, float] = {}
        if species is not None:
            cluster_species = species[in_cluster]
            for sp in species_list:
                comp[sp] = float(np.sum(cluster_species == sp)) / n_atoms * 100.0

        clusters.append({
            "id": int(cid),
            "n_atoms": n_atoms,
            "volume": vol,
            "composition": comp,
        })

    cluster_volume_sum = sum(
        c["volume"] for c in clusters if not np.isnan(c["volume"])  # type: ignore[arg-type]
    )
    volume_fraction = cluster_volume_sum / total_volume if total_volume > 0 else 0.0

    mean_composition: dict[str, float] = {}
    if clusters and species_list:
        for sp in species_list:
            vals = [c["composition"].get(sp, 0.0) for c in clusters]  # type: ignore[union-attr]
            mean_composition[sp] = float(np.mean(vals))

    noise_in_cluster = mask & (dbscan_labels == -1)
    n_noise_cluster = int(noise_in_cluster.sum())

    log.info(
        "Cluster analysis: %d clusters, volume_fraction=%.4f%%",
        len(clusters), volume_fraction * 100.0,
    )
    if n_noise_cluster > 0:
        log.info("  %d noise atoms also classified as cluster (excluded from volume)", n_noise_cluster)
    for c in clusters:
        log.info("  Cluster %d: %d atoms, volume=%.1f angstrom^3", c["id"], c["n_atoms"], c["volume"])

    return {
        "clusters": clusters,
        "total_volume": total_volume,
        "cluster_volume_sum": cluster_volume_sum,
        "volume_fraction": volume_fraction,
        "mean_composition": mean_composition,
        "species_list": species_list,
        "n_noise_cluster_atoms": n_noise_cluster,
    }


def save_cluster_stats_csv(
    stats: dict[str, object],
    output: str = "cluster_stats.csv",
) -> None:
    """Save per-cluster statistics and summary to CSV."""
    clusters: list[dict[str, object]] = stats["clusters"]  # type: ignore[assignment]
    species_list: list[str] = stats["species_list"]  # type: ignore[assignment]

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["cluster_id", "n_atoms", "volume_angstrom3"]
        header.extend([f"{sp}_at_pct" for sp in species_list])
        writer.writerow(header)

        for c in clusters:
            row: list[object] = [c["id"], c["n_atoms"], f"{c['volume']:.2f}"]
            for sp in species_list:
                row.append(f"{c['composition'].get(sp, 0.0):.2f}")  # type: ignore[union-attr]
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(["# Summary"])
        writer.writerow(["total_volume_angstrom3", f"{stats['total_volume']:.2f}"])
        writer.writerow(["cluster_volume_sum_angstrom3", f"{stats['cluster_volume_sum']:.2f}"])
        writer.writerow(["volume_fraction_pct", f"{stats['volume_fraction'] * 100:.4f}"])  # type: ignore[arg-type]

        mean_composition: dict[str, float] = stats["mean_composition"]  # type: ignore[assignment]
        if mean_composition:
            writer.writerow([])
            writer.writerow(["# Mean cluster composition (at.%)"])
            for sp in species_list:
                writer.writerow([sp, f"{mean_composition.get(sp, 0.0):.2f}"])

    log.info("Saved cluster stats to %s", output)
