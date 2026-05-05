# Implementing a VAE in numpy

## Goal

The goal of this project is to build a VAE from scratch using only Numpy and no NN libraries.
This was inspired by using scVI and wanting to recreate it from the ground up. 
For now, I will be implementing a barebones VAE and develop it further from there.

No AI agents were used to write any code, but I did ask Gemini for help with some of the plotting functions. 

## Repository Outline
```
.
├── src/                # Source code
│   ├── model.py        # VAE model
│   ├── utils.py        # Adam optimizer and some plotting and clustering functions
│   └── train.py        # A wrapper to train the model
├── .gitignore          # Files to exclude from Git
├── README.md           # Voila
├── mnist_notebook.md   # Notebook testing the VAE and showing how latent clustering is helpful
├── intro_to_VAEs.pdf   # Expository paper about VAEs and how they work.
└── environment.yml     # Environment with all the necessary libraries
```

To experiment with the VAE and see a toy example of why latent space clustering can be helpful, check out mnist_notebook.md. 
I will not be writing an explanation of how VAEs work. See `intro_to_VAEs.pdf` (cited below) to learn more. 

Diederik, P. Kingma, and Welling Max. "An introduction to variational autoencoders." Foundations and Trends® in Machine Learning 12.4 (2019): 307-392.

## Future plans
After I finish implementing an scVI autoencoder, I may try to write a basic VAe in C to learn C.