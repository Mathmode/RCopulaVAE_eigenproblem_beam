# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 09:50:41 2026

@author: anafd
"""

import os 
from matplotlib import pyplot as plt

def plot_trainval_loss(history, folder_path):
    """
    Plots the training and validation evolution for the GC-VAE.
    Uses Matplotlib's internal mathtext for labels to avoid external LaTeX dependencies.
    """
    # Use a style suitable for academic publishing without requiring external LaTeX
    plt.rcParams.update({
        "text.usetex": False, # Set to False to resolve 'latex could not be found' error
        "font.family": "serif",
        "font.size": 12,
        "mathtext.fontset": "cm" # Use Computer Modern for a LaTeX-like look
    })
    
    trainvalloss_plot_fig = plt.figure(figsize=(8, 6))
    
    # Axis scales
    plt.yscale('log')
    plt.xscale('log')
    
    # Plotting data
    plt.plot(history.item()['loss'], 
             color='black', 
             linewidth=1.5, 
             linestyle='--', 
             label=r'Train loss ($\mathcal{L}_{\mathrm{ELBO}}^{\text{train}}$)')
    
    plt.plot(history.item()['val_loss'], 
             color='darkgray', 
             linewidth=1.5, 
             label=r'Validation loss ($\mathcal{L}_{\mathrm{ELBO}}^{\text{val}}$)')
    
    # Labels and Legend
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(r'$\mathcal{L}_{\mathrm{ELBO}}$ (log scale)', fontsize=14)
    plt.legend(loc='upper right', fontsize=12, frameon=True)
    
    # Grid and layout
    plt.grid(True, which="both", linestyle='-.', alpha=0.5)
    plt.tight_layout()
    
    # Save with high resolution
    trainvalloss_plot_fig.savefig(
        os.path.join(folder_path, 'TrainVal_lossevolution.png'),
        dpi=500, 
        bbox_inches='tight'
    )
    # plt.close()
    
    
def plot_freqsMACs_loss(history, folder_path):
    """
    Plots the data misifit loss terms
    Uses Matplotlib's internal mathtext for labels to avoid external LaTeX dependencies.
    """
    # Use a style suitable for academic publishing without requiring external LaTeX
    plt.rcParams.update({
        "text.usetex": False, # Set to False to resolve 'latex could not be found' error
        "font.family": "serif",
        "font.size": 12,
        "mathtext.fontset": "cm" # Use Computer Modern for a LaTeX-like look
    })
    
    trainvalloss_plot_fig = plt.figure(figsize=(8, 6))
    
    # Axis scales
    plt.yscale('log')
    plt.xscale('log')
    
    # Plotting data
    plt.plot(history.item()['Freqs_loss'], 
             color='black', 
             linewidth=1.5, 
             linestyle='--', 
             label=r'Frequency misfit($\mathcal{L}_{\mathrm{freq}}^{\text{val}}$)')
    
    plt.plot(history.item()['MAC_modes_loss'], 
             color='darkgray', 
             linewidth=1.5, 
             label=r'Mode shape misfit loss ($\mathcal{L}_{\mathrm{MAC}}^{\text{val}}$)')
    
    # Labels and Legend
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(r'$\mathcal{L}_{\mathrm{ELBO}}$ (log scale)', fontsize=14)
    plt.legend(loc='upper right', fontsize=12, frameon=True)
    
    # Grid and layout
    plt.grid(True, which="both", linestyle='-.', alpha=0.5)
    plt.tight_layout()
    
    # Save with high resolution
    trainvalloss_plot_fig.savefig(
        os.path.join(folder_path, 'FreqsMACs_Val_lossevolution.png'),
        dpi=500, 
        bbox_inches='tight'
    )