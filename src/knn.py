import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class KNN:
    def __init__(self, samples, target):
        self.samples = samples
        self.target = target
        self.colors = ["red", "green", "blue", "orange"]

    def classify_sample(self, sample, K):
        """Returns index of nearest sample, the distance to it and the corresponding label."""
        dx = (self.samples[:, 0] - sample[0]) ** 2
        dy = (self.samples[:, 1] - sample[1]) ** 2
        nearest_idxs = np.argpartition(dx + dy, K)[:K]
        nn_slice = self.samples[nearest_idxs]
        dist = np.sqrt(
            np.max(
                (nn_slice[:, 0] - sample[0]) ** 2 + (nn_slice[:, 1] - sample[1]) ** 2
            )
        )
        neighbor_labels = self.target[nearest_idxs]
        dominant_label = np.bincount(neighbor_labels).argmax()
        return nearest_idxs, dist, int(dominant_label)

    def plot(self, sample, K):
        """Plots."""
        nearest_idxs, dist, dominant_label = self.classify_sample(sample, K)
        colors = self.colors
        x_min = self.samples[:, 0].min()
        x_max = self.samples[:, 0].max()
        y_min = self.samples[:, 1].min()
        y_max = self.samples[:, 1].max()
        plt.figure(figsize=(12, 9))
        for color, val in zip(colors, np.unique(self.target)):
            cluster = self.samples[self.target == val]
            plt.scatter(cluster[:, 0], cluster[:, 1], color=color, alpha=0.6)
        dominant_color = colors[dominant_label]
        plt.scatter(
            sample[0], sample[1], color=dominant_color, zorder=9, alpha=0.5, s=130
        )
        plt.scatter(
            sample[0],
            sample[1],
            color="black",
            zorder=10,
            marker="+",
            s=100,
            label="Query Point",
        )
        circle = patches.Circle(
            (sample[0], sample[1]),
            radius=dist,
            color="black",
            fill=False,
            linestyle="dashed",
            linewidth=2,
        )
        plt.gca().add_patch(circle)
        for idx in nearest_idxs:
            neighbor = self.samples[idx]
            line_color = colors[self.target[idx]]
            plt.plot(
                [neighbor[0], sample[0]], [neighbor[1], sample[1]], color=line_color
            )
        plt.scatter(
            self.samples[nearest_idxs, 0],
            self.samples[nearest_idxs, 1],
            marker="+",
            zorder=9,
            color="black",
            s=75,
            alpha=0.6,
        )
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect("equal", adjustable="datalim")
        plt.title(f"Full Dataset with K-NN Circle\nK = {K}")
        plt.legend()
        plt.show()

    def plot_zoomed(self, sample, K: int) -> None:
        """Plots with a zoom around the KNN sphere."""
        nearest_idxs, dist, dominant_label = self.classify_sample(sample, K)
        colors = self.colors
        zoom_margin = 0.2 * dist
        x_low, x_high = sample[0] - dist - zoom_margin, sample[0] + dist + zoom_margin
        y_low, y_high = sample[1] - dist - zoom_margin, sample[1] + dist + zoom_margin
        zoomed_mask = (
            (self.samples[:, 0] >= x_low)
            & (self.samples[:, 0] <= x_high)
            & (self.samples[:, 1] >= y_low)
            & (self.samples[:, 1] <= y_high)
        )
        zoomed_samples = self.samples[zoomed_mask]
        zoomed_labels = self.target[zoomed_mask]
        plt.figure(figsize=(12, 9))
        for color, val in zip(colors, np.unique(self.target)):
            cluster = zoomed_samples[zoomed_labels == val]
            plt.scatter(cluster[:, 0], cluster[:, 1], color=color, alpha=0.6)
        plt.scatter(
            self.samples[nearest_idxs, 0],
            self.samples[nearest_idxs, 1],
            marker="+",
            zorder=9,
            color="black",
            s=75,
            alpha=0.6,
            label="Nearest Neighbors",
        )
        dominant_color = colors[dominant_label]
        plt.scatter(
            sample[0], sample[1], color=dominant_color, zorder=10, alpha=0.5, s=130
        )
        plt.scatter(
            sample[0],
            sample[1],
            color="black",
            zorder=10,
            marker="+",
            s=100,
            label="Query Point",
        )
        circle = patches.Circle(
            (sample[0], sample[1]),
            radius=dist,
            color="black",
            fill=False,
            linestyle="dashed",
            linewidth=2,
        )
        plt.gca().add_patch(circle)
        for idx in nearest_idxs:
            neighbor = self.samples[idx]
            line_color = colors[self.target[idx]]
            plt.plot(
                [neighbor[0], sample[0]], [neighbor[1], sample[1]], color=line_color
            )
        plt.xlim(x_low, x_high)
        plt.ylim(y_low, y_high)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"Zoomed-in View of K-NN Circle with Relevant Samples\nK = {K}")
        plt.legend()
        plt.show()


def example() -> None:
    """Example showing the different functionalities."""
    samples, t = make_blobs(n_samples=1000, centers=4, cluster_std=0.8, random_state=0)
    x_min = samples[:, 0].min()
    x_max = samples[:, 0].max()
    y_min = samples[:, 1].min()
    y_max = samples[:, 1].max()
    rng = np.random.default_rng()
    xs = rng.uniform(x_min, x_max)
    ys = rng.uniform(y_min, y_max)
    xs = 0.0
    ys = 3.0
    K = 13
    knn = KNN(samples, t)
    _ = knn.classify_sample((xs, ys), K)
    knn.plot((xs, ys), K)
    knn.plot_zoomed((xs, ys), K)


if __name__ == "__main__":
    example()
