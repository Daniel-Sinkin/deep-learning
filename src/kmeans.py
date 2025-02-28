"""Contains the KMeans class."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class KMeans:
    """
    Implementation of the KMeans clustering algorithm, does some numpy tricks to speed up
    computations but performance was not a real priority.

    Use the `KMeans.from_blobs()` function to quickly get an example.
    """

    def __init__(self, data, K: int = 3, seed=None):
        self.data = data
        self.K = K

        _rng = np.random.default_rng(seed=seed)
        centroid_idxs = _rng.choice(range(len(data)), K, replace=False)
        self.centroids = data[centroid_idxs]

    def get_assignments(self) -> np.ndarray:
        """Gets a numpy array with the current cluster assignments."""
        dists = np.array(
            [((self.data - centroid) ** 2).sum(axis=1) for centroid in self.centroids]
        )
        return dists.argmin(axis=0)

    def plot(self, show_fig: bool = True, filepath: Optional[str] = None) -> None:
        """Plots the cluster. Add a filepath if you want to save the plot."""
        plt.tight_layout()

        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, len(self.centroids)))

        cluster_assignments = self.get_assignments()

        for i, color in enumerate(colors):
            data_clustered = self.data[np.where(cluster_assignments == i)]
            plt.scatter(
                data_clustered[:, 0],
                data_clustered[:, 1],
                s=50,
                color=color,
                label=f"Cluster {i} ({len(data_clustered)})",
                alpha=0.6,
            )

            plt.scatter(
                self.centroids[i][0],
                self.centroids[i][1],
                s=100,
                alpha=1.0,
                c="black",
                zorder=10,
                marker="x",
            )

        title_str = list(map(lambda x: f"({x[0]:.2f}, {x[1]:.2f})", self.centroids))
        plt.title(f"Clustering Visualization\nCentroids = {title_str}")
        plt.legend()

        if filepath:
            plt.savefig(filepath)
        if show_fig:
            plt.show()
        else:
            plt.clf()

    def update(self) -> bool:
        """Modifies centroids inplace, returns True if the centroids changed."""
        cluster_assignments_before = self.get_assignments()
        for centroid_idx in range(self.K):
            new_centroid = self.data[
                np.where(cluster_assignments_before == centroid_idx)
            ].mean(axis=0)
            if not np.allclose(self.centroids[centroid_idx], new_centroid):
                self.centroids[centroid_idx] = new_centroid
        cluster_assignments_after = self.get_assignments()

        return (cluster_assignments_before != cluster_assignments_after).any()

    @staticmethod
    def from_blobs(K: int = 4, n_samples: int = 1000, cluster_std=0.6) -> "KMeans":
        """Returns a KMeans object"""
        samples, _ = make_blobs(
            n_samples=n_samples, centers=K, cluster_std=cluster_std, random_state=0
        )
        return KMeans(data=samples, K=K)


if __name__ == "__main__":
    iteration = 0
    kmeans = KMeans.from_blobs()
    kmeans.plot(show_fig=False, filepath=f"./screenshots/kmeans/{iteration}.png")
    iteration += 1

    while kmeans.update():
        kmeans.plot(show_fig=False, filepath=f"./screenshots/kmeans/{iteration}.png")
        iteration += 1

    kmeans.plot(show_fig=False, filepath=f"./screenshots/kmeans/{iteration}.png")
    iteration += 1
