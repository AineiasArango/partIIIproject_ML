#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#optimizing hyperparameters with optuna
import optuna
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import optim, nn, utils, Tensor
import lightning as L
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from lightning import Trainer
from typing import List, Tuple, Dict
import os
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError
import wandb
from pytorch_lightning.loggers import WandbLogger
import pickle
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import graph_network as net

neighbour_data = np.load("LowRes_neighbour_data_norm.npy")
mass_flows = np.load("LowRes_mass_flow.npy")
global_data = np.load("LowRes_global_data.npy")
edges = pickle.load(open("LowRes_edges.pkl", "rb"))

target_mean = np.mean(mass_flows)
target_std = np.std(mass_flows)
mass_flows = (mass_flows - target_mean) / target_std

model_name = "EdgeNet"

def objective(trial):
    # Define the hyperparameters to optimize
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    valid_size = trial.suggest_categorical('valid_size', [0.1, 0.2, 0.3])
    test_size = trial.suggest_categorical('test_size', [0.1, 0.2, 0.3])
    n_layers = trial.suggest_int('n_layers', 3, 6)
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    latent_channels = trial.suggest_categorical('latent_channels', [16, 32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    
    # Train the model with these parameters
    score = net.train_model(
        mass_flows, neighbour_data, edges, global_data,
        batch_size=batch_size, valid_size=valid_size, 
        test_size=test_size, model_name=model_name,
        n_layers=n_layers, hidden_channels=hidden_channels,
        latent_channels=latent_channels, learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=42
    )
    
    return score

# Create and run the study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get the best parameters and print them
best_params = study.best_params
print("\nBest parameters found:")
for key, value in best_params.items():
    print(f"{key}: {value}")

# Print the best score achieved
print(f"\nBest score: {study.best_value}")

# If you're using wandb, log the best parameters
wandb.log({"best_hyperparameters": best_params, "best_score": study.best_value})