"""DBKNN sklearn-compatible estimator."""

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted

from dbknn.algorithm import (
    compute_dbscan_multiplier,
    compute_hybrid_score,
    compute_knn_distances,
    karcher_classify,
    run_dbscan,
)

log = logging.getLogger(__name__)


class DBKNN(BaseEstimator, ClusterMixin):
    """DBSCAN-KNN hybrid estimator for solute cluster identification.

    Parameters
    ----------
    eps : float
        DBSCAN neighbourhood radius in angstroms.
    min_samples : int
        DBSCAN minimum samples per core point.
    k : int
        Number of nearest neighbours for kNN distance.
    method : str
        KARCH thresholding method ('simple' or 'complex').
    threshold : float | None
        Manual hybrid-score threshold.  Atoms with score <= threshold
        are labelled as cluster.  If *None* (default), KARCH auto-threshold.
    cluster_weight : float
        DBSCAN multiplier for atoms inside a cluster.
    noise_weight : float
        DBSCAN multiplier for noise atoms.
    """

    def __init__(
        self,
        eps: float = 8.0,
        min_samples: int = 15,
        k: int = 10,
        method: str = "simple",
        threshold: float | None = None,
        cluster_weight: float = 0.5,
        noise_weight: float = 1.5,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.k = k
        self.method = method
        self.threshold = threshold
        self.cluster_weight = cluster_weight
        self.noise_weight = noise_weight

    def fit(self, X: NDArray[np.floating], y: None = None) -> "DBKNN":
        """Run the DBKNN pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Atom positions.
        y : ignored

        Returns
        -------
        self
        """
        X = check_array(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        knn_dist = compute_knn_distances(X, k=self.k)
        self.dbscan_labels_ = run_dbscan(X, eps=self.eps, min_samples=self.min_samples)
        multiplier = compute_dbscan_multiplier(
            self.dbscan_labels_,
            cluster_weight=self.cluster_weight,
            noise_weight=self.noise_weight,
        )
        self.hybrid_scores_ = compute_hybrid_score(knn_dist, multiplier)
        self.labels_, self.threshold_ = karcher_classify(
            self.hybrid_scores_, method=self.method, threshold=self.threshold,
        )

        return self

    def score_samples(self) -> NDArray[np.floating]:
        """Return hybrid scores (lower = more likely cluster).

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        return self.hybrid_scores_
