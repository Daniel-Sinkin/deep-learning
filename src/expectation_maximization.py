"""Implementation of the EM Algorithm."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

from .kmeans import KMeans


class ExpectationMaximization:
    def __init__(
        self, data: np.ndarray, centroids: np.ndarray, cluster_assignments: np.ndarray
    ):
        """
        Parameters:
        - data: np.ndarray of shape (N, D), the dataset.
        - cluster_assignemnts: np.ndarray of shape (N, ) with values 0 <= v < K
        - centroids: np.ndarray of shape (K, D), initial centroids.
        """
        self.data = data
        self.mus = centroids.copy()
        self.K, self.D = self.mus.shape
        self.N = len(data)

        self.cluster_assignments = cluster_assignments

        self.sigmas = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            cluster_points = data[self.cluster_assignments == k]
            if cluster_points.shape[0] > 1:
                self.sigmas[k] = np.cov(cluster_points, rowvar=False)
            else:
                # For a single point, default to identity matrix
                self.sigmas[k] = np.eye(self.D)

        self.pis = np.ones(self.K) / self.K

        self.gamma = None

    def compute_gamma(
        self, data: np.ndarray, pis: np.ndarray, mus: np.ndarray, sigmas: np.ndarray
    ) -> np.ndarray:
        """
        Compute the responsibility matrix gamma.

        Parameters:
        - data: (N, D) dataset
        - pis: (K,) mixing coefficients
        - mus: (K, D) means of Gaussian components
        - sigmas: (K, D, D) covariance matrices

        Returns:
        - gamma: (N, K) responsibility matrix
        """
        N, _ = data.shape
        K = pis.shape[0]
        gamma = np.zeros((N, K))
        for j in range(K):
            pdf_values = multivariate_normal.pdf(data, mean=mus[j], cov=sigmas[j])
            gamma[:, j] = pis[j] * pdf_values
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def update(self) -> None:
        """
        Perform one iteration of the EM algorithm (E-step and M-step).
        """
        # E-step: compute responsibilities
        self.gamma = self.compute_gamma(self.data, self.pis, self.mus, self.sigmas)
        Nhat = self.gamma.sum(axis=0)

        # Update mixing coefficients (pis)
        self.pis = Nhat / self.N

        # M-step: update means (mus)
        for j in range(self.K):
            self.mus[j] = np.dot(self.gamma[:, j], self.data) / Nhat[j]

        # M-step: update covariance matrices (sigmas)
        new_sigmas = np.zeros_like(self.sigmas)
        for j in range(self.K):
            accumulator = np.zeros((self.D, self.D))
            for i in range(self.N):
                v = self.data[i] - self.mus[j]
                accumulator += self.gamma[i, j] * np.outer(v, v)
            new_sigmas[j] = accumulator / Nhat[j]
        self.sigmas = new_sigmas

    @staticmethod
    def draw_ellipse(
        mean: np.ndarray, cov: np.ndarray, ax=None, n_std=2.0, **kwargs
    ) -> None:
        """
        Draw an ellipse representing the Gaussian component.

        Parameters:
        - mean: (2,) mean of the Gaussian.
        - cov: (2, 2) covariance matrix of the Gaussian.
        - ax: Matplotlib axis (optional).
        - n_std: Number of standard deviations for ellipse scaling.
        """
        if ax is None:
            ax = plt.gca()

        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        eigenvalues = np.maximum(eigenvalues, 1e-6)  # Ensure non-negative eigenvalues
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        width = min(width, 100)  # Clamp extreme values
        height = min(height, 100)
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor="black",
            facecolor="none",
            linewidth=2,
            **kwargs,
        )
        ax.add_patch(ellipse)

    def plot(self, show_fig: bool = True, filepath=None) -> None:
        """
        Visualize the current state of the EM algorithm. Plots the data points colored by their most
        likely cluster and overlays the Gaussian ellipses.
        """
        if self.gamma is None:
            self.gamma = self.compute_gamma(self.data, self.pis, self.mus, self.sigmas)

        plt.figure(figsize=(12, 9))
        cluster_assignments = np.argmax(self.gamma, axis=1)

        for j in range(self.K):
            cluster_data = self.data[cluster_assignments == j]
            plt.scatter(
                cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {j}", alpha=0.6
            )
            self.draw_ellipse(self.mus[j], self.sigmas[j])

        plt.scatter(
            self.mus[:, 0],
            self.mus[:, 1],
            marker="x",
            color="red",
            s=100,
            label="Cluster Means",
        )
        plt.legend()
        plt.title("EM Algorithm State: Gaussian Mixtures")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        if filepath is not None:
            plt.savefig(filepath, dpi=300)
        if show_fig:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def from_kmeans(kmeans: KMeans) -> ExpectationMaximization:
        """Creates a EM object from a kmeans object."""
        return ExpectationMaximization(
            data=kmeans.data,
            centroids=kmeans.centroids,
            cluster_assignments=kmeans.get_assignments(),
        )


def example() -> None:
    K = 4
    kmeans = KMeans.from_blobs(K=K, cluster_std=1.5)
    for _ in range(5):
        kmeans.update()

    kmeans.plot(show_fig=False, filepath="./screenshots/em/kmeans.png")

    em = ExpectationMaximization.from_kmeans(kmeans=kmeans)
    for i in range(5):
        em.update()
        em.plot(show_fig=False, filepath=f"./screenshots/em/{i}.png")


if __name__ == "__main__":
    example()
