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
os.chdir("/data/ERCblackholes4/aasnha2/for_aineias/plots3")
neighbour_data = np.load("LowRes_neighbour_data_norm.npy")
mass_flows = np.load("LowRes_mass_flow.npy")
global_data = np.load("LowRes_global_data.npy")
edges = pickle.load(open("LowRes_edges.pkl", "rb"))

target_mean = np.mean(mass_flows)
target_std = np.std(mass_flows)
mass_flows = (mass_flows - target_mean) / target_std

input_size = neighbour_data.shape[2]
model_name = "EdgeNet"
batch_size, valid_size, test_size = 32, 0.1, 0.2
n_layers, hidden_channels, latent_channels, learning_rate, weight_decay = 5, 64, 32, 1e-3, 0
net.train_model(mass_flows, neighbour_data, edges, global_data, batch_size=batch_size, valid_size=valid_size, test_size=test_size, model_name=model_name, n_layers=n_layers, hidden_channels=hidden_channels, latent_channels=latent_channels, learning_rate=learning_rate, weight_decay=weight_decay, seed=42)

