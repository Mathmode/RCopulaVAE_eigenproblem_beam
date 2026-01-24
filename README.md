# Uncertainty-Aware Damage Identification via Copula-Based Variational Autoencoder

This repository contains the implementation of a **Physics-Informed Copula-Based Variational Autoencoder (VAE)** for Structural Health Monitoring (SHM). 

The model solves the inverse problem of identifying damage (stiffness reduction factors) in beam structures from modal properties (frequencies and mode shapes), explicitly accounting for uncertainty and variable dependencies.

## 📖 Context & Methodology
This project implements the methodology described in the paper: **"Uncertainty-aware damage identification in beams via Copula-Based Variational Autoencoder"**.

Unlike standard "black-box" VAEs, this architecture embeds the governing physics of the structure directly into the decoder, ensuring that all predictions satisfy the beam's equations of motion.

### Architecture Overview
The model consists of three main blocks:

1.  **Encoder (Inverse Model):** * Takes measured modal properties (Frequencies, Mode Shapes) as input.
    * Estimates the parameters of a **Gaussian Copula** (means, scales, and Cholesky factors of the correlation matrix) to model the posterior distribution of damage parameters.
2.  **Sampling Layer:**
    * Generates samples from the latent space using the learned Copula and Gaussian Marginals.
    * Handles the transformation between the unbounded latent space and the physical bounded domain of stiffness reduction factors $[0, 1]$.
3.  **Physics-Informed Decoder (Forward Model):**
    * **No Neural Network:** The decoder is a **differentiable Finite Element solver**.
    * It assembles the global stiffness matrix $K$ based on the sampled reduction factors.
    * It solves the generalized eigenvalue problem $K\Phi = \lambda M \Phi$ using a differentiable eigensolver to reconstruct the modal properties.


## 📂 Project Structure

```text
.
├── Data/                          # Directory for training datasets
├── Output/                        # Checkpoints and training logs
├── MODULES/
│   ├── COPULAS/
│   │   ├── GC_GMm_architectures.py  # Encoder & Copula layer definitions
│   │   ├── GC_GMm_eigen_functions.py # Differentiable Eigensolver & Matrix Assembly
│   │   ├── GC_GMm_functions.py      # Helper functions (Copula sampling, Marginals)
│   │   ├── GC_GMm_models.py         # Main VAE Model Class & Custom Losses
│   │   └── GC_postprocessing_copulas.py # Plotting and analysis tools
│   └── PREPROCESSING/             # Data loading utilities
├── main_GCopula_GMM.py            # Main training script
└── README.md
