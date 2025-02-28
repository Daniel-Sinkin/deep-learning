import numpy as np


class LearningRateScheduling:
    """Adapting Learning Rate."""

    @staticmethod
    def step_decay(t: int, lr_min: float, lr_max: float, t_threshold: int) -> float:
        """Step Decay."""
        return lr_max if t < t_threshold else lr_min

    @staticmethod
    def exp_decay(t: int, lr_0: float, lr_min: float, decay_factor: float):
        """Exponential Decay."""
        return (lr_0 - lr_min) * np.exp(-decay_factor * t) + lr_min

    @staticmethod
    def cosine_decay(t: int, lr_min: float, lr_max: float, t_max: float):
        """Cosine Decay."""
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(t / t_max * np.pi))


def main() -> None:
    """Plots the LearningRateScheduling functions in one plot and saves the result to a png."""
    lr_max = 1.0
    lr_min = 0.1
    t_threshold = 0.8

    t_min = 0.0
    t_max = 1.0

    num_t = 50

    ts = np.linspace(t_min, t_max, num_t)

    lrs_step = map(
        lambda t: LearningRateScheduling.step_decay(
            t, lr_min=lr_min, lr_max=lr_max, t_threshold=t_threshold
        ),
        ts,
    )
    lrs_exp = map(
        lambda t: LearningRateScheduling.exp_decay(
            t, lr_0=lr_max, lr_min=lr_min, decay_factor=4.0
        ),
        ts,
    )
    lrs_cos = map(
        lambda t: LearningRateScheduling.cosine_decay(
            t, lr_min=lr_min, lr_max=lr_max, t_max=t_max
        ),
        ts,
    )

    import matplotlib.pyplot as plt  # pylint: disable=C0415

    plt.figure(figsize=(12, 9))

    plt.plot(
        [t_min, t_threshold],
        [lr_max, lr_max],
        color="green",
        marker="*",
        label="Step Decay",
    )
    plt.plot([t_threshold, t_max], [lr_min, lr_min], color="green", marker="*")
    plt.plot(ts, list(lrs_exp), marker="*", label="Exponential Decay")
    plt.plot(ts, list(lrs_cos), marker="*", label="Cosine Decay")

    plt.title("Learning Rate Scheduling")
    plt.legend()
    plt.savefig("./screenshots/util/lrs.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
