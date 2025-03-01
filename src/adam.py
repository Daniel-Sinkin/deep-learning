"""
https://arxiv.org/abs/1412.6980
"""

import numpy as np


class AdamOptimizer:
    """Adaptive Moment Estimation"""

    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        # alpha / stepsize
        self.lr = np.float32(lr)
        # Exponential decay rates for the moment estimates
        self.beta1 = np.float32(beta1)
        self.beta2 = np.float32(beta2)

        self.eps = np.float32(eps)

        self.model = model

        # 1. moment vector
        self.m = {param: np.zeros_like(param) for param in model.get_parameters()}
        # 2. moment vector
        self.v = {param: np.zeros_like(param) for param in model.get_parameters()}
        # timestep
        self.t = 0

    def step(self) -> None:
        """One optimization iteration."""
        self.t += 1
        for param in self.model.get_parameters():
            # Get gradients w.r.t. stochastci objective at timestep t
            g = param.grad
            # Update biased first moment estimate
            self.m[param] = self.beta1 * self.m[param] + (1.0 - self.beta1) * g
            # Update biased second raw moment estimate
            self.v[param] = self.beta2 * self.v[param] + (1.0 - self.beta2) * g**2

            # Compute bias-corrected first moment estimate
            m_hat = self.m[param] / (1.0 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[param] / (1.0 - self.beta2**self.t)
            # Update Parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
