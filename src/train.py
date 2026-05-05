"""
Training of moodel using Adam optimizer
"""
import numpy as np
from src.model import VAE
from src.utils import AdamOptimizer
import time

def train(vae, adam, data, epochs : int, batch_size = 64, patience = 5):
    """
    Training loop. Needs initialized VAE and adam. Keeps track of mean epoch loss and time to complete epoch.
    args: 
        - training data
        - hidden_dims, latent_dims: to be passed to vae
        - training epochs
    """
    # initializing VAE and adam optimizer
    #vae = VAE(input_dim = data.shape[1], hidden_dims = hidden_dims, latent_dim = latent_dims)
    #adam = AdamOptimizer(weights = vae.params)
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
    
    for e in range(epochs):

        # random training batch
        # starting timer
        start_time = time.perf_counter()
        print(f'Epoch {e+1}/{epochs} training...')

        # setting min loss, patience, and epochs with no improvement to keep track
        min_loss = np.inf
        epochs_no_improve = 0

        batches = get_batch(data)
        loss_list = []
        for b in batches:
            grads, loss = vae.full_pass(b)
            adam.step(vae.params, grads)
            loss_list.append(loss)

        # keeping track of mean epoch loss
        end_time = time.perf_counter()
        epoch_duration = end_time - start_time
        mean_loss = np.mean(loss_list)
        epoch_loss[e] = mean_loss

        # keeping track of min epoch loss to finish training early if possible
        if mean_loss < min_loss:
            min_loss = mean_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # reporting stats about epoch train
        print(f'Epoch {e+1}/{epochs} completed in {epoch_duration:.2f} seconds with mean loss {epoch_loss[e]:.3f}')

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {e}")

    return vae, epoch_loss