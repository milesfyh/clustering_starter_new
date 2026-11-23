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
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a nonempty 2D array.")

        n = X.shape[0]
        k_eff = min(self.k, n - 1)
        if k_eff < 0:
            raise ValueError("k must be nonnegative.")
        if k_eff == 0:
            self.w_ij = np.empty((n, 0), dtype=float)
            self.flux = np.ones(n, dtype=float)
            self.sinks = np.ones(n, dtype=bool)
            self.labels = np.arange(n)
            return self

        nodes = np.arange(n)

        # Step 1: Pre-compute centroid, centroid distances, sigma, and k-NN edge weights
        centroid = X.mean(axis=0)
        d = np.linalg.norm(X - centroid, axis=1)

        # Compute one row at a time to avoid an n-by-n-by-dimension array.
        sq_dists = np.empty((n, n))
        for i, point in enumerate(X):
            diff = X - point
            sq_dists[i] = np.einsum("ij,ij->i", diff, diff)

        if self.sigma is None:
            pairwise_distances = np.sqrt(sq_dists[np.triu_indices(n, k=1)])
            sigma = 0.2 * float(np.median(pairwise_distances))
        else:
            sigma = float(self.sigma)

        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1, kind="stable")[:, :k_eff].copy()
        knn_sq_dists = sq_dists[nodes[:, None], knn_indices]

        tau = 0.1 * sigma
        downhill_mask = d[knn_indices] <= (d[:, None] + tau)
        if sigma > 0.0:
            with np.errstate(over="ignore", under="ignore"):
                w_ij = np.exp(-(np.sqrt(knn_sq_dists) / sigma) ** 2)
                base_downhill_w = np.where(downhill_mask, w_ij, 0.0)

                # Rescale a row if all its eligible weights rounded to zero.
                # The same factor cancels when the outgoing weights are normalized.
                underflowed = downhill_mask.any(axis=1) & ~base_downhill_w.any(axis=1)
                eligible_distances = np.where(
                    downhill_mask[underflowed], knn_sq_dists[underflowed], np.inf
                )
                relative_distances = eligible_distances - eligible_distances.min(
                    axis=1, keepdims=True
                )
                base_downhill_w[underflowed] = np.exp(
                    -(np.sqrt(relative_distances) / sigma) ** 2
                )
        else:
            w_ij = (knn_sq_dists == 0.0).astype(float)
            base_downhill_w = np.where(downhill_mask, w_ij, 0.0)

        # Step 2: Initialize stored flux f_i(0) = 1
        flux = np.ones(n, dtype=float)

        # Step 3: Simulate flow for T iterations
        delta = np.zeros((n, k_eff), dtype=float)
        outgoing = np.zeros(n, dtype=float)

        for t in range(self.T):
            if t == 0 or self.beta == 0:
                w_tilde = base_downhill_w
            elif self.beta is None:
                phi = np.log1p(flux)
                w_tilde = base_downhill_w * phi[knn_indices]
            else:
                phi = np.power(flux, self.beta)
                w_tilde = base_downhill_w * phi[knn_indices]

            denom = w_tilde.sum(axis=1)
            has_outflow = denom > 0.0

            delta.fill(0.0)
            outgoing = np.where(has_outflow, self.alpha * flux, 0.0)
            np.divide(
                w_tilde,
                denom[:, None],
                out=delta,
                where=has_outflow[:, None],
            )
            delta *= outgoing[:, None]

            incoming = np.zeros(n, dtype=float)
            np.add.at(incoming, knn_indices, delta)
            flux = flux - outgoing + incoming

        # Step 4: Detect sinks (low outgoing flux, excluding depleted nodes)
        sinks = (outgoing < self.epsilon) & (flux >= self.epsilon)
        if not np.any(sinks):
            sinks[int(np.argmin(d))] = True

        # Step 5: Assign clusters via steepest downhill path with cycle detection
        masked_delta = np.where(downhill_mask, delta, -np.inf)
        has_downhill = downhill_mask.any(axis=1)
        best_pos = np.argmax(masked_delta, axis=1)
        steepest = np.where(has_downhill, knn_indices[nodes, best_pos], nodes)

        labels = np.full(n, -1, dtype=int)
        next_label = 0
        for sink_node in np.flatnonzero(sinks):
            labels[sink_node] = next_label
            next_label += 1

        for i in range(n):
            if labels[i] != -1:
                continue

            path = []
            visited = set()
            curr = i

            while labels[curr] == -1:
                visited.add(curr)
                path.append(curr)
                next_node = int(steepest[curr])
                if next_node in visited:
                    sinks[curr] = True
                    labels[curr] = next_label
                    next_label += 1
                    break
                curr = next_node

            terminal_label = labels[curr]
            for node in path:
                labels[node] = terminal_label

        # Step 6: Merge clusters of size s_min or less into a larger cluster.
        s_min = max(2, n // 20)
        counts = np.bincount(labels)
        large_ids = np.flatnonzero(counts > s_min)
        small_ids = np.flatnonzero(counts <= s_min)

        if len(large_ids) > 0 and len(small_ids) > 0:
            cluster_centroids = np.zeros((len(counts), X.shape[1]), dtype=float)
            np.add.at(cluster_centroids, labels, X)
            cluster_centroids /= counts[:, None]

            remap = np.arange(len(counts))
            for cid in small_ids:
                dists_to_large = np.linalg.norm(
                    cluster_centroids[large_ids] - cluster_centroids[cid], axis=1
                )
                remap[cid] = large_ids[int(np.argmin(dists_to_large))]
            labels = remap[labels]

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
