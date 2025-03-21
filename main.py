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
neighbour_data = np.load("mass_flowNoBHFableLowSNEff_32_True.npz")["neighbour_data"]
mass_flows = np.load("mass_flowNoBHFableLowSNEff_32_True.npz")["mass_flow"]
input_size = neighbour_data.shape[2]

target_mean = np.mean(mass_flows)
target_std = np.std(mass_flows)
mass_flows = (mass_flows - target_mean) / target_std

hidden_sizes, max_epochs, learning_rate, weight_decay, dropout_rate, valid_size, test_size = [64, 64], 100, 1e-3, 1e-4, 0.2, 0.1, 0.2
net.train_model(neighbour_data, mass_flows, input_size, hidden_sizes, max_epochs, learning_rate, weight_decay, dropout_rate, valid_size, test_size, seed=42)


