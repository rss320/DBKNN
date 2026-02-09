"""Pure algorithmic functions for DBKNN."""

import logging

import numpy as np
from numpy.typing import NDArray
from pythresh.thresholds.karch import KARCH
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

log = logging.getLogger(__name__)


def compute_knn_distances(
    positions: NDArray[np.floating],
    k: int = 10,
) -> NDArray[np.floating]:
    """Compute distance to the k-th nearest neighbour for each atom.

    Matches PyOD KNN ``method='largest'`` used in the original paper.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(positions)
    distances, _ = nn.kneighbors(positions)
    kth_dist: NDArray[np.floating] = distances[:, -1]  # k-th (farthest) neighbour
    log.info("k-NN distances (k=%d): mean=%.2f, std=%.2f", k, kth_dist.mean(), kth_dist.std())
    return kth_dist


def run_dbscan(
    positions: NDArray[np.floating],
    eps: float = 8.0,
    min_samples: int = 15,
) -> NDArray[np.integer]:
    """Run DBSCAN and return labels (-1 = noise)."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels: NDArray[np.integer] = db.fit_predict(positions)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_cluster_atoms = int((labels >= 0).sum())
    log.info(
        "DBSCAN (eps=%.1f, min_samples=%d): %d clusters, %d cluster atoms, %d noise",
        eps, min_samples, n_clusters, n_cluster_atoms, int((labels == -1).sum()),
    )
    return labels


def compute_dbscan_multiplier(
    labels: NDArray[np.integer],
    cluster_weight: float = 0.5,
    noise_weight: float = 1.5,
) -> NDArray[np.floating]:
    """Map DBSCAN labels to multipliers: cluster (>=0) -> *cluster_weight*, noise (-1) -> *noise_weight*."""
    multiplier = np.where(labels >= 0, cluster_weight, noise_weight).astype(np.float64)
    return multiplier


def compute_hybrid_score(
    knn_dist: NDArray[np.floating],
    multiplier: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Hybrid score = kNN distance x DBSCAN multiplier."""
    score: NDArray[np.floating] = knn_dist * multiplier
    return score


def karcher_classify(
    scores: NDArray[np.floating],
    method: str = "simple",
    threshold: float | None = None,
) -> tuple[NDArray[np.integer], float]:
    """Classify atoms by thresholding hybrid scores.

    If *threshold* is given, it is applied directly (scores <= threshold → cluster).
    Otherwise the KARCH auto-thresholder is used.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        1 = cluster, 0 = matrix.
    threshold : float
        Threshold in the original score space.
    """
    if threshold is not None:
        labels: NDArray[np.integer] = (scores <= threshold).astype(int)
        log.info(
            "Manual threshold=%.4f: %d / %d atoms classified as cluster",
            threshold, int(labels.sum()), len(labels),
        )
        return labels, float(threshold)

    s_min, s_max = float(scores.min()), float(scores.max())
    thresholder = KARCH(method=method)
    outlier_labels = thresholder.eval(scores)
    # KARCH: 1 = above threshold (high score = noise), 0 = below (cluster)
    labels = (1 - outlier_labels).astype(int)
    # Denormalize threshold from [0, 1] back to original score space
    auto_threshold = float(thresholder.thresh_ * (s_max - s_min) + s_min)
    log.info(
        "KARCH (method=%s): threshold=%.4f, %d / %d atoms classified as cluster",
        method, auto_threshold, int(labels.sum()), len(labels),
    )
    return labels, auto_threshold
