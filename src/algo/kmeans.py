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

    def __init__(
        self, data, K: int = 3, centroids: Optional[np.ndarray] = None, seed=None
    ):
        self.data = data
        self.K = K

        if centroids is None:
            _rng = np.random.default_rng(seed=seed)
            centroid_idxs = _rng.choice(range(len(data)), K, replace=False)
            self.centroids = data[centroid_idxs]
        else:
            self.centroids = centroids

    def __call__(self) -> bool:
        return self.update()

    def __len__(self) -> int:
        return len(self.data)

    def get_dists(self) -> np.ndarray:
        """Gets the distances of the data to the centroids."""
        return np.array(
            [((self.data - centroid) ** 2).sum(axis=1) for centroid in self.centroids]
        )

    def get_cost(self) -> np.ndarray:
        """
        Computes the total cost / objective function value of the clustering, i.e., returns
        J = sum_{n = 1}^N sum_{k = 1}^K r_{nk} ||x_n - mu_k||^2
        where
        r_nk = 1 if k = arg min_j ||x_n - mu_j||^2 and 0 otherwise.
        """
        return self.get_dists().sum()

    def get_assignments(self) -> np.ndarray:
        """Gets a numpy array with the current cluster assignments."""
        return self.get_dists().argmin(axis=0)

    def plot(self, show_fig: bool = True, filepath: Optional[str] = None) -> None:
        """Plots the cluster. Add a filepath if you want to save the plot."""
        plt.figure(figsize=(12, 9))

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
                c="red",
                zorder=10,
                marker="x",
            )

        centroids_str = list(map(lambda x: f"({x[0]:.2f}, {x[1]:.2f})", self.centroids))
        cost = self.get_cost()
        cost_rel = cost / len(self)
        titles = [
            "Clustering Visualization",
            f"n_samples = {len(self)}, K = {self.K}, cost = {cost:.2f} ({cost_rel:.2f} per sample)",
            f"Centroids = {centroids_str}",
        ]
        plt.title("\n".join(titles))
        plt.legend()

        if filepath is not None:
            plt.savefig(filepath, dpi=300)

        if show_fig:
            plt.show()
        else:
            plt.close()

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
    def from_blobs(K: int = 4, n_samples: int = 1000, cluster_std=0.6) -> KMeans:
        """Returns a KMeans object"""
        samples, _ = make_blobs(
            n_samples=n_samples, centers=K, cluster_std=cluster_std, random_state=0
        )
        return KMeans(data=samples, K=K)

    @staticmethod
    def get_lowest_cost_centroid_after_n_step(
        data: np.ndarray, K: int, n_steps: int = 10
    ) -> np.ndarray:
        """Runs n_steps of clustering, returning the centroids with the lowest cost of those."""
        kmeans = KMeans(data=data, K=K)

        min_cost = kmeans.get_cost()
        min_centroids = kmeans.centroids
        for _ in range(n_steps):
            if not kmeans.update():
                # Steps if we already found a (local) minimum with KMeans
                break
            new_cost = kmeans.get_cost()
            if new_cost < min_cost:
                min_cost = new_cost
                min_centroids = kmeans.centroids

        return min_centroids


def example() -> None:
    """Runs Kmeans until there is no more improvement, saves every iteration as screenshots."""
    iteration = 0
    kmeans = KMeans.from_blobs()
    kmeans.plot(show_fig=False, filepath=f"./screenshots/kmeans/{iteration}.png")
    iteration += 1

    while kmeans():
        kmeans.plot(show_fig=False, filepath=f"./screenshots/kmeans/{iteration}.png")
        iteration += 1

    # If we immediately stopped then we don't need to make a final screenshot
    if iteration > 1:
        kmeans.plot(show_fig=False, filepath=f"./screenshots/kmeans/{iteration}.png")
        iteration += 1


if __name__ == "__main__":
    example()
