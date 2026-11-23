from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels``.
        """
        # Hint: Start with computing centroid = X.mean(axis=0)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a nonempty two-dimensional array")

        n = len(X)
        nodes = np.arange(n)
        centroid_distances = np.linalg.norm(X - X.mean(axis=0), axis=1)
        squared_distances = np.empty((n, n))
        for i, point in enumerate(X):
            differences = X - point
            squared_distances[i] = np.einsum("ij,ij->i", differences, differences)

        sigma = self.sigma
        if sigma is None:
            if n > 1:
                pairwise_distances = np.sqrt(squared_distances[np.triu_indices(n, k=1)])
                sigma = 0.2 * np.median(pairwise_distances)
            else:
                sigma = 0.0

        effective_k = min(self.k, n - 1)
        np.fill_diagonal(squared_distances, np.inf)
        neighbors = np.argsort(squared_distances, axis=1, kind="stable")[:, :effective_k].copy()
        graph = np.zeros((n, n), dtype=bool)
        graph[nodes[:, None], neighbors] = True
        graph |= graph.T

        transport_weights = np.zeros((n, n))
        if sigma == 0:
            # In the zero-bandwidth limit, only coincident points have weight.
            transport_weights[graph] = squared_distances[graph] == 0
        else:
            scaled_distances = np.sqrt(squared_distances[graph]) / sigma
            transport_weights[graph] = np.exp(-(scaled_distances ** 2))

        w_ij = transport_weights[nodes[:, None], neighbors]

        # Rows are senders, and columns are receivers.
        downhill = centroid_distances[None, :] <= centroid_distances[:, None] + 0.1 * sigma
        downhill_weights = transport_weights * downhill
        flux = np.ones(n)
        delta = np.zeros_like(downhill_weights)
        outgoing = np.zeros(n)
        for iteration in range(self.T):
            weights = downhill_weights
            if iteration > 0 and self.beta != 0:
                if self.beta is None:
                    reinforcement = np.log1p(flux)
                else:
                    reinforcement = flux ** self.beta
                weights = downhill_weights * reinforcement[None, :]

            totals = weights.sum(axis=1)
            outgoing = np.where(totals > 0, self.alpha * flux, 0.0)
            # Normalize each sender's weights independently. Empty rows send zero.
            np.divide(weights, totals[:, None], out=delta, where=totals[:, None] > 0)
            delta[totals == 0] = 0
            delta *= outgoing[:, None]
            flux = (flux - outgoing) + delta.sum(axis=0)

        sinks = np.flatnonzero(outgoing < self.epsilon).tolist()
        if not sinks:
            sinks.append(int(centroid_distances.argmin()))

        steepest = delta.argmax(axis=1)
        steepest[outgoing == 0] = nodes[outgoing == 0]
        labels = np.full(n, -1, dtype=int)
        labels[sinks] = np.arange(len(sinks))
        for start in nodes:
            current = start
            path = []
            visited = set()
            while labels[current] == -1:
                visited.add(current)
                path.append(current)
                next_node = steepest[current]
                if next_node in visited:
                    labels[current] = len(sinks)
                    sinks.append(int(current))
                    break
                current = next_node
            labels[path] = labels[current]

        min_size = max(2, n // 20)
        counts = np.bincount(labels)
        large_clusters = np.flatnonzero(counts >= min_size)
        small_clusters = np.flatnonzero(counts < min_size)
        if large_clusters.size and small_clusters.size:
            centroids = np.zeros((len(counts), X.shape[1]))
            np.add.at(centroids, labels, X)
            centroids /= counts[:, None]
            merged_labels = np.arange(len(counts))
            for cluster in small_clusters:
                distances = np.linalg.norm(centroids[large_clusters] - centroids[cluster], axis=1)
                merged_labels[cluster] = large_clusters[distances.argmin()]
            labels = merged_labels[labels]

        sinks = np.asarray(sinks, dtype=int)

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"
