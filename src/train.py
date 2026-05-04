"""
Training of moodel using Adam optimizer
"""
import numpy as np
from src.model import VAE
from src.utils import AdamOptimizer

def train(data, hidden_dims: list, latent_dims: int, epochs : int, batch_size = 64):
    """
    Training loop. Wraps VAE and Optimizer.
    args: 
        - training data
        - hidden_dims, latent_dims: to be passed to vae
        - training epochs
    """
    # initializing VAE and adam optimizer
    vae = VAE(input_dim = data.shape[1], hidden_dims = hidden_dims, latent_dim = latent_dims)
    adam = AdamOptimizer(weights = vae.params)
    # loss dictionary
    epoch_loss = {}

    # helper for getting batches
    def get_batch(data):
        """
        Takes data and splits it into batches
        args: 
            - training dataset
        """
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        batches = []

        for i in range(0, data.shape[0], batch_size):
            batch = data[i: i+batch_size]
            batches.append(batch)
        return batches
    
    for e in range(epochs+1):
        # random training batch
        batches = get_batch(data)
        loss_list = []
        for b in batches:
            grads, loss = vae.full_pass(b)
            adam.step(vae.params, grads)
            loss_list.append(loss)

        # keeping track of mean epoch loss
        epoch_loss[e] = np.mean(loss_list)

    return vae, epoch_loss