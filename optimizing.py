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
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from network import CellMLP, CellMLPLightning
import network as net
os.chdir("/data/ERCblackholes4/aasnha2/for_aineias/plots2")
neighbour_data = np.load("mass_flowNoBHFableLowSNEff_1024_True.npz")["neighbour_data"]
mass_flows = np.load("mass_flowNoBHFableLowSNEffHighRes_32.npz")["mass_flow"]
input_size = neighbour_data.shape[2]

target_mean = np.mean(mass_flows)
target_std = np.std(mass_flows)
mass_flows = (mass_flows - target_mean) / target_std

def objective(trial):
    # Define the hyperparameters to optimize
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    hidden_sizes = trial.suggest_categorical('hidden_sizes', [
        (64, 64),
        (128, 128),
        (256, 256),
        (128, 128, 128, 128, 128),
        (64, 64, 64, 64, 64),
        (64, 64, 64, 64, 64, 64)
    ])
    max_epochs = trial.suggest_categorical('max_epochs', [50, 100, 200, 500])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3, 0.5, 0.7])
    valid_size = trial.suggest_categorical('valid_size', [0.1, 0.2, 0.3])
    test_size = trial.suggest_categorical('test_size', [0.1, 0.2, 0.3])
    
    # Train the model with these parameters
    score = net.train_model(
        neighbour_data, mass_flows, input_size, list(hidden_sizes), max_epochs, 
        learning_rate, weight_decay, dropout_rate, valid_size, test_size, seed=42
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