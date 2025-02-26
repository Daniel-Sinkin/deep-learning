import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from src.mlp_model import MLPModel, ReLUActivation, SigmoidActivation


def main():
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X = X.T
    y = y.reshape(1, -1)

    # Standard scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X.T).T

    layers = [2, 20, 40, 20, 1]
    mlp = MLPModel(
        layers,
        hidden_activation="relu",
        output_activation="sigmoid",
        seed=42,
    )

    losses = []
    for epoch in range(3000):
        loss = mlp.backprop(X, y, lr=0.001)  # smaller LR
        losses.append(loss)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss = {loss:.4f}")

    plt.figure()
    plt.plot(losses)
    plt.title("Cross-Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Plot decision boundary
    x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
    y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    Z = mlp.forward(grid_points).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2, colors=["blue", "red"])
    plt.contour(xx, yy, Z, levels=[0.5], colors="black")
    plt.scatter(
        X[0, y[0] == 0], X[1, y[0] == 0], c="blue", edgecolor="k", label="Class 0"
    )
    plt.scatter(
        X[0, y[0] == 1], X[1, y[0] == 1], c="red", edgecolor="k", label="Class 1"
    )
    plt.title("Moon Dataset Decision Boundary")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
