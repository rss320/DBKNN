"""Plotting functions (lazy matplotlib + scienceplots import)."""

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

COLORS = ["xkcd:blue", "xkcd:orange", "xkcd:black", "xkcd:red"]


def _setup_style(scatter: bool = False) -> "plt":
    """Import matplotlib, apply science style, and return plt."""
    try:
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install dbknn[plot] for plotting support") from exc

    styles = ["science", "ieee", "grid", "no-latex"]
    if scatter:
        styles.append("scatter")
    plt.style.use(styles)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLORS)
    return plt


def plot_scores(
    scores: NDArray[np.floating],
    threshold: float,
    output_path: str = "scores_histogram.png",
) -> None:
    """Plot hybrid score histogram with threshold line."""
    plt = _setup_style()

    fig, ax = plt.subplots()
    ax.hist(scores, bins=100, color=COLORS[0], alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(threshold, color=COLORS[3], linestyle="--", linewidth=1.2, label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Hybrid score (\u00c5)")
    ax.set_ylabel("Count")
    ax.legend(frameon=True, edgecolor="black")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved score histogram to %s", output_path)


def plot_results_3d(
    positions: NDArray[np.floating],
    cluster_mask: NDArray[np.integer],
    output_path: str = "results_3d.png",
) -> None:
    """3D scatter plot: cluster vs matrix solutes."""
    plt = _setup_style(scatter=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    mask = cluster_mask.astype(bool)
    matrix = positions[~mask]
    clusters = positions[mask]

    ax.scatter(matrix[:, 0], matrix[:, 1], matrix[:, 2], s=1, alpha=0.15, color=COLORS[0], label="Matrix")
    ax.scatter(clusters[:, 0], clusters[:, 1], clusters[:, 2], s=5, alpha=0.8, color=COLORS[3], label="Cluster")

    ax.set_xlabel("x (\u00c5)")
    ax.set_ylabel("y (\u00c5)")
    ax.set_zlabel("z (\u00c5)")
    ax.legend(frameon=True, edgecolor="black")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 3D results to %s", output_path)


def plot_cluster_stats(
    stats: dict[str, object],
    output_path: str = "cluster_stats.png",
) -> None:
    """Two-panel figure: (a) cluster volume distribution, (b) mean composition.

    Parameters
    ----------
    stats : dict
        Output of :func:`dbknn.analysis.compute_cluster_stats`.
    output_path : str
        Path to save the figure.
    """
    plt = _setup_style()

    clusters: list[dict[str, object]] = stats["clusters"]  # type: ignore[assignment]
    species_list: list[str] = stats["species_list"]  # type: ignore[assignment]
    mean_comp: dict[str, float] = stats["mean_composition"]  # type: ignore[assignment]
    volume_fraction: float = stats["volume_fraction"]  # type: ignore[assignment]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # (a) Cluster volume histogram
    volumes = [c["volume"] for c in clusters if not np.isnan(c["volume"])]  # type: ignore[arg-type]
    if volumes:
        ax1.hist(
            volumes, bins=min(20, max(5, len(volumes))),
            color=COLORS[0], alpha=0.7, edgecolor="black", linewidth=0.3,
        )
    ax1.set_xlabel(r"Cluster volume ($\mathrm{\AA}^3$)")
    ax1.set_ylabel("Count")
    ax1.text(
        0.95, 0.95, f"$V_f$ = {volume_fraction * 100:.2f}%",
        transform=ax1.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=0.5),
    )
    ax1.text(0.02, 0.95, "a)", transform=ax1.transAxes, ha="left", va="top", fontweight="bold")

    # (b) Mean composition bar chart
    if mean_comp:
        sp_labels = [sp for sp in species_list if mean_comp.get(sp, 0.0) > 0.5]
        values = [mean_comp[sp] for sp in sp_labels]
        bar_colors = [COLORS[i % len(COLORS)] for i in range(len(sp_labels))]
        ax2.bar(sp_labels, values, color=bar_colors, edgecolor="black", linewidth=0.3)
    ax2.set_xlabel("Species")
    ax2.set_ylabel("Mean composition (at.%)")
    ax2.text(0.02, 0.95, "b)", transform=ax2.transAxes, ha="left", va="top", fontweight="bold")

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cluster stats plot to %s", output_path)


def plot_structure(
    positions: NDArray[np.floating],
    symbols: list[str],
    cluster_labels: NDArray[np.integer],
    matrix_species: list[str] | None = None,
    output_path: str = "structure.png",
) -> None:
    """3D scatter plot of the APT structure.

    Parameters
    ----------
    matrix_species : list[str] | None
        Species to exclude from the plot (e.g. ``["Fe"]``).
        If *None*, all atoms are plotted.
    """
    plt = _setup_style(scatter=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if matrix_species is not None:
        solute_mask = np.array([s not in matrix_species for s in symbols])
    else:
        solute_mask = np.ones(len(symbols), dtype=bool)
    sol_pos = positions[solute_mask]
    sol_labels = cluster_labels[solute_mask]

    matrix_mask = sol_labels == 0
    cluster_mask = sol_labels > 0

    ax.scatter(
        sol_pos[matrix_mask, 0], sol_pos[matrix_mask, 1], sol_pos[matrix_mask, 2],
        s=1, alpha=0.2, color=COLORS[0], label="Matrix solutes",
    )
    ax.scatter(
        sol_pos[cluster_mask, 0], sol_pos[cluster_mask, 1], sol_pos[cluster_mask, 2],
        s=5, alpha=0.8, color=COLORS[3], label="Cluster solutes",
    )

    ax.set_xlabel("x (\u00c5)")
    ax.set_ylabel("y (\u00c5)")
    ax.set_zlabel("z (\u00c5)")
    ax.legend(frameon=True, edgecolor="black")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot to %s", output_path)
