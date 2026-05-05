"""
Computational tools:
    AdamOptimizer: implementation of adam stochastic optimizer
Visualization tools:
    visualize_samples: function to visualize a few random handwritten digits
"""

import numpy as np
import matplotlib.pyplot as plt
import umap
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

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

        
def visualize_samples(X, num_samples = 7):
    """
    Randomly selects num_samples datapoints (images) to visualize
    args:
        - X: data
        - num_samples: number of data points to visualize
    """
    indices = np.random.choice(X.shape[0], num_samples, replace = False)

    plt.figure(figsize=(2 * num_samples, 3))
    
    for i, idx in enumerate(indices):

        # reshape to actual image dimensions
        img = X[idx].reshape(28, 28)
        
        # plot
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Sample {idx}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

def visualize_reconstruction(vae, X, num_samples = 7):
    """
    Plots original images and plots reconstructed image
    args:
        - vae: VAE instance to reconstruct data
        - X: data
        - num_samples: how many data points to reconstruct
    """
    indices = np.random.choice(X.shape[0], num_samples, replace=False)
    X_sample = X[indices]
    X_hat = vae.forward_pass(X_sample)

    # plotting
    plt.figure(figsize=(2 * num_samples, 4))
    
    for i in range(num_samples):
        # original images on top
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X_sample[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # reconstructed images below
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(X_hat[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstruction")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

def visualize_loss(epoch_loss):
    """
    Plots the training loss curve
    args:
        - epoch_loss: dictionary of mean loss per epoch
    """
    # extracting losses
    epochs = epoch_loss.keys()
    losses = [epoch_loss[e] for e in epochs]

    # initializing the plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, color='royalblue', linewidth=2, marker='o', markersize=4)

    # adding labels and styling
    plt.title("Model Training Loss Over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (ELBO)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()

def clustering(latent_rep, n_neighbors=30, resolution=0.5, min_dist=0.3, show = True):
    """
    Performs UMAP for visualization and Leiden clustering on a KNN graph
    args:
        - latent_rep: latent representation of data from vae (use vae.encode(data))
        - n_neighbors: number of neighbors for KNN and UMAP
        - resolution: controls cluster granularity (higher = more clusters)
        - min_dist: UMAP parameter for how tightly points are packed
        - show: show the plot or not
    returns:
        - embedding: 2d embedding of the latent space via UMAP
        - clusters: cluster assignments to each point in latent space
    """
    
    # UMAP for dimensionality reduction
    dim_reducer = umap.UMAP(n_neighbors=n_neighbors, 
                        min_dist=min_dist, 
                        n_components=2, 
                        random_state=42)
    embedding = np.asarray(dim_reducer.fit_transform(latent_rep))

    # knn graph for leiden
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(latent_rep)
    adj_matrix = knn.kneighbors_graph(latent_rep, mode='connectivity')

    # convert to igraph for the leiden clustering
    sources, targets = adj_matrix.nonzero() #type: ignore
    edges = list(zip(sources, targets))
    g = ig.Graph(n=latent_rep.shape[0], edges=edges, directed=False)

    # clustering
    partition = la.find_partition(g, la.RBConfigurationVertexPartition, 
                                 resolution_parameter=resolution)
    clusters = np.array(partition.membership)

    # visualize
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x = embedding[:, 0], 
        y = embedding[:, 1], 
        hue = clusters, 
        palette = 'viridis', 
        s = 30, 
        edgecolor = 'none',
        legend = 'full'
    )
    plt.title(f"VAE Latent Space: UMAP + Leiden (Clusters: {len(np.unique(clusters))})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.axis('off')
    if show:
        plt.show()

    return embedding, clusters

def cluster_reconstruction(vae, data):
    """
    Samples latent points from each cluster and plots original vs reconstructed data
    args:
        - vae: VAE instance
        - latent_rep: the latent representation of data
        - clusters: cluster assignments for each latent point
        - data: the original data input
    """
    # getting latent representation and clustering
    mu, _ = vae.encode(data)
    _, clusters = clustering(mu, show = False)

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    
    # setting up plot
    fig, axes = plt.subplots(n_clusters, 4, 
                             figsize=(8, n_clusters * 2))

    for i, cluster_id in enumerate(unique_clusters):
        # find indices belonging to the current cluster
        indices = np.where(clusters == cluster_id)[0]
        
        # randomly sample indices 
        n_to_sample = min(len(indices), 2)
        sampled_indices = np.random.choice(indices, n_to_sample, replace=False)
        
        for j, idx in enumerate(sampled_indices):
            # original, latent, reconstructed
            x = data[idx]
            z = mu[idx:idx+1]
            xhat = vae.decode(z)

            # plotting
            # original image
            ax_orig = axes[i, j * 2]
            ax_orig.imshow(x.reshape(28, 28), cmap='gray')
            ax_orig.set_title(f"C{cluster_id} Original")
            ax_orig.axis('off')
            
            # reconstructed image
            ax_recon = axes[i, j * 2 + 1]
            ax_recon.imshow(xhat.reshape(28,28), cmap='gray')
            ax_recon.set_title(f"C{cluster_id} Reconstruction")
            ax_recon.axis('off')

    plt.tight_layout()
    plt.show()