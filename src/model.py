"""
VAE implementation
"""

import numpy as np
#from src.utils import AdamOptimizer

class VAE:
    """
    Variational autoencoder with isotropic gaussian prior, p_theta(z), and p_theta(x|z) multivariate gaussian. 
    Encoder and decoder are multilayered MLP.
    """
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        """
        Initializing the VAE
        args:
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
        args:
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

        # initializing end layers of encoder: N(mu, var) prior
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

    def encode(self, X):
        """
        Pass through the encoder and caches the activations for backprop. 
        Input -> hidden layers -> latent
        args:
            X - data
        returns estimated params of prior
            mu, var 
        """
        # feeding through encoder input -> last hidden layer
        self.cache = {'Ae': X}
        A = X
        for i in range(1,len(self.hidden_dims)):
            W = self.params[f'We{i}']
            b = self.params[f'be{i}']
            Z = A @ W + b
            A = np.maximum(0, Z)
            self.cache[f'Ze{i}'] = Z
            self.cache[f'Ae{i}'] = A
        
        # encoding latent representation params
        Wmu = self.params['Wmu']
        bmu = self.params['Wmu']
        Wlv = self.params['bmu']
        blv = self.params['blv']

        mu = A @ Wmu + bmu
        lv = A @ Wlv + blv

        return mu, lv
    
    def reparam_trick(self, mu, lv):
        eps = np.random.multivariate_normal(np.zeros_like(mu), np.identity(len(mu)))
        v = np.exp(0.5 * lv)
        return mu + np.multiply(v, eps)

    def decode(self, X):
        """
        Pass through the decoder and caches the activations for backprop. 
        Latent -> hidden layers -> input
        args:
            X - data
        returns reconstruction
            Z
        """
        # hidden layers except the end
        self.cache = {'Ad': X}
        A = X
        for i in range(1,len(self.hidden_dims)-1):
            W = self.params[f'Wd{i}']
            b = self.params[f'bd{i}']
            Z = A @ W + b
            A = np.maximum(0, Z)
            self.cache[f'Ze{i}'] = Z
            self.cache[f'Ae{i}'] = A
        
        # last sigmoid hidden layer
        W = self.params[f'Wd{len(self.hidden_dims)-1}']
        b = self.params[f'bd{len(self.hidden_dims)-1}']
        Z = A @ W + b
        A = 1 / (1 + np.exp(-Z))
        self.cache[f'Ze{len(self.hidden_dims)-1}'] = Z
        self.cache[f'Ae{len(self.hidden_dims)-1}'] = A
        
        # output
        W_out = self.params[f'Wd{len(self.hidden_dims)}']
        b_out = self.params[f'bd{len(self.hidden_dims)}']
        Z_out = A @ W_out + b_out
        
        return Z_out