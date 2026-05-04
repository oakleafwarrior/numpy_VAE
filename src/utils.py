"""
Computational tools:
    AdamOptimizer: implementation of adam stochastic optimizer
"""

import numpy as np

class AdamOptimizer:
    def __init__(self, weights, alpha = 0.001, beta1 = 0.9, beta2 = 0.99, eps = 1e-8):
        """
        A numpy implementation of ADAM stochastic optimizer following the original paper
            weights - weights of neurons
            alpha - step size in parameter space
            beta1 - exponential decay rate for first moment
            beta2 - exponential decay rate for second moment
            eps - 
        """
        
        # initializing weights
        self.m = {key: np.zeros_like(m) for key, m in weights.items()}
        self.v = {key: np.zeros_like(v) for key, v in weights.items()}

        # intializing time step
        self.t = 0

        # initializing weights
        self.weights = weights
        
        # intializing hyperparams
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    # one step of the activation
    def step(self, weights, grads):
        """
        One pass of adam optimizer
            weights - weights of neurons
            grads - gradients of activation functions
        """
        
        self.t += 1
        alpha_t = self.alpha * np.sqrt(1-self.beta2 ** self.t) / (1-self.beta1 ** self.t)

        for key in weights.keys():
            # updating moments
            self.m[key] = self.beta1 * self.m[key] + (1-self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1-self.beta2) * grads[key] ** 2

            # updating weights
            weights[key] -= alpha_t * self.m[key] / (np.sqrt(self.v[key]) + self.eps)

        

    