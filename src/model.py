"""
VAE implementation
"""

import numpy as np
from src.utils import AdamOptimizer

class VAE:
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        """
        Initializing the VAE
            input_dim   -   input dimension of flattened data
            hidden_dim  -   list where entry i is the size of layer i of the encoder/decoder neural network
            latent_dim  -   latent representation size
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # initalizing
        self.params = {}
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializing the weights and biases of each layer in the encoder and decoder
            Wei - encoder layer i weights
            bei - encoder layer i biases
            _mu, _lv - mu, log variance layer weights and biases
            Wdi - decoder layer i weights
            bdi - decoder layer i biases
            W_out, b_out - output layer of the decoder weights and biases
        Using ReLU activation for eveything but final layer of decoder using sigmoid activation
        """
        # initializing encoder...
        current_dim = self.input_dim
        # initializing hidden layers in encoder
        for i, h_dim in enumerate(self.hidden_dims):
            self.params[f'We{i}'] = np.random.randn(current_dim, h_dim) * np.sqrt(2/current_dim)
            self.params[f'be{i}'] = np.zeros((1, h_dim))
            current_dim = h_dim

        # initializing end layers of encoder N(mu, var) prior
        self.params['Wmu'] = np.random.randn(current_dim, self.latent_dim) * np.sqrt(2/current_dim)
        self.params['bmu'] = np.zeros((1, self.latent_dim))
        self.params['Wlv'] = np.random.randn(current_dim, self.latent_dim) *  np.sqrt(2/current_dim)
        self.params['bmu'] = np.zeros((1, self.latent_dim))

        # initializing decoder (reverse of encoder)...
        current_dim = self.latent_dim
        # hidden layers
        for i, h_dim in enumerate(self.hidden_dims[::-1]):
            self.params[f'Wd{i}'] = np.random.randn(current_dim, h_dim) * np.sqrt(2/current_dim)
            self.params[f'bd{i}'] = np.zeros((1, h_dim))
            current_dim = h_dim

        # final layer
        self.params['W_out'] = np.random.randn(current_dim, self.input_dim) * np.sqrt(1/current_dim)
        self.params['b_out'] = np.zeros((1, self.input_dim))