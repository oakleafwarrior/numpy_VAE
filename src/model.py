"""
VAE implementation
"""

import numpy as np

class VAE:
    """
    Variational autoencoder for classification. 
    Isotropic gaussian prior, p_theta(z), and p_theta(x|z) multivariate gaussian. 
    Encoder and decoder are multilayered MLP. ReLU activation in each layer except last decoder layer which is sigmoid.
    Cross entropy + KL divergence loss 
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
        self.params['blv'] = np.zeros((1, self.latent_dim))

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
        for i in range(len(self.hidden_dims)):
            W = self.params[f'We{i}']
            b = self.params[f'be{i}']
            Z = A @ W + b
            A = np.maximum(0, Z)
            self.cache[f'Ze{i}'] = Z
            self.cache[f'Ae{i}'] = A
        
        # encoding latent representation params
        Wmu = self.params['Wmu']
        bmu = self.params['bmu']
        Wlv = self.params['bmu']
        blv = self.params['blv']

        mu = A @ Wmu + bmu
        lv = A @ Wlv + blv

        return mu, lv
    
    def reparam_trick(self, mu, lv):
        eps = np.random.multivariate_normal(np.zeros_like(mu), np.identity(len(mu)))
        self.cache['eps'] = eps
        v = np.exp(0.5 * lv)
        return mu + np.multiply(v, eps)

    def decode(self, Z):
        """
        Pass through the decoder and caches the activations for backprop. 
        Latent -> hidden layers -> input
        args:
            Z - latent sample
        returns reconstruction
            Xhat - reconstructed data
        """
        # hidden layers except the end
        self.cache['Ad'] = Z
        A = Z
        for i in range(len(self.hidden_dims)-1):
            W = self.params[f'Wd{i}']
            b = self.params[f'bd{i}']
            Z = A @ W + b
            A = np.maximum(0, Z)
            self.cache[f'Zd{i}'] = Z
            self.cache[f'Ad{i}'] = A
        
        # last sigmoid hidden layer and output
        W_out = self.params[f'Wd{len(self.hidden_dims)-1}']
        b_out = self.params[f'bd{len(self.hidden_dims)-1}']
        Z_out = A @ W_out + b_out
        A_out = 1 / (1 + np.exp(-Z_out))
        self.cache[f'Z_out'] = Z_out
        self.cache[f'A_out'] = A_out
        
        # naming convention for my understanding
        X_hat = A_out

        return X_hat
    
    # forward pass of the model
    def forward_pass(self, X):
        """
        Whole forward pass.
        encode -> reparametrize -> decode
        args:
            - X data
        output:
            - reconstructed data
        """
        mu, lv = self.encode(X)
        z = self.reparam_trick(mu, lv)
        Xhat = self.decode(z)
        return Xhat
    
    # computing loss function after a forward pass
    def loss(self, X, Xhat, mu, lv):
        """
        Computing loss: KL + reconstruction loss
        """
        Xhat = np.clip(Xhat, 1e-10, 1-1e-10)
        kl = -0.5 * np.sum(1 + lv - np.square(mu) - np.exp(lv), axis = 1)
        ll = -np.sum(X * np.log(Xhat) + (1-X) * np.log(1 -Xhat), axis = 1)
        total_loss = np.mean(kl + ll)
        return total_loss
    
    def backprop(self, X):
        """
        Backprop gradients
        """
        grads = {}
        batch_size = X.shape[0]

        # cached variables
        A_out = self.cache['A_out']
        eps = self.cache['eps']
        mu = self.cache['mu']
        lv = self.cache['lv']

        # cross entropy loss + sigmoid activation derivative
        dZ = (A_out - X)/batch_size

        # output gradients
        A_prev = self.cache[f'Ad{len(self.hidden_dims)-1}']
        grads['Wd_out'] = A_prev.T @ dZ
        grads['bd_out'] = np.sum(dZ, axis = 0, keepdims = True)

        # backprop output weights
        W_out = self.params['W_out']
        dA_prev = dZ @ A_prev
        dZ = dA_prev @ (A_prev > 0).astype(float)

        # decoder backwards pass
        for i in range(len(self.hidden_dims)-1,-1, -1):
            # retrieving activated neuron
            if i == 0:
                A_prev = self.cache[f'Ad']
            else:
                A_prev = self.cache[f'Ad{i-1}']

            W = self.params[f'Wd{i}']

            # gradients for layer i weights/bias
            grads[f'Wd{i}'] = A_prev.T @ dZ
            grads[f'bd{i}'] = np.sum(dZ, axis = 0, keepdims = True)
            
            # backprop to dA
            dA_prev = dZ @ W.T

            # activation for next layer
            if i > 0:
                dZ = dA_prev * (A_prev > 0).astype(float)

        dA_latent = dA_prev

        # gradients of KL divergence
        dKL_mu = mu / batch_size
        dKL_lv = 0.5 * (np.exp(lv)-1) / batch_size

        # reconstruction gradients
        dRecon_mu = dA_latent
        dRecon_lv = 0.5 * dA_latent * np.exp(0.5 * lv) * eps

        # total mu, lv grads
        dZ_mu = dRecon_mu + dKL_mu
        dZ_lv = dRecon_lv + dKL_lv

        # encoder backward pass (same structure as decoder)

        # (mu, lv) head of encoder
        Ae_out = self.cache[f'Ae{len(self.hidden_dims)}']
        grads['We_mu'] = Ae_out.T @ dZ_mu
        grads['We_lv'] = Ae_out.T @ dZ_lv
        grads['be_mu'] = np.sum(dZ_mu, axis = 0, keepdims = True)
        grads['be_lv'] = np.sum(dZ_lv, axis = 0, keepdims = True)

        for i in range(len(self.hidden_dims)-1,-1,-1):
            if i == 0:
                A_prev = self.cache['Ae']
            else:
                A_prev = self.cache[f'Ae{i-1}']
            
            W = self.params[f'We{i}']

            grads[f'We{i}'] = A_prev.T @ dZ
            grads[f'be{i}'] = np.sum(dZ, axis = 0, keepdims = True)

            dA_prev = dZ @ W.T

            if i > 0:
                dZ = dA_prev * (A_prev > 0).astype(float)

        return grads

    def full_pass(self, X):
        """
        Full pass throught the VAE.
        args:
            - X: data
        returns:
            dict: dictionary containing grads and loss
        """
        # foward pass yielding reconstructed data
        Xhat = self.forward_pass(X)
        
        # latent vars
        mu = self.cache['mu']
        lv = self.cache['lv']

        # computing loss and grad
        loss = self.loss(X, Xhat, mu, lv)
        grads = self.backprop(X)

        return grads, loss