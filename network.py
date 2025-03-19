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

#cellMLP
class CellMLP(nn.Module):
    def __init__(self, input_size=4, hidden_sizes=[64, 64], dropout_rate=0.2):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
        
        # Output layer (single value per cell)
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch_size, num_cells, input_features)
        batch_size, num_cells, _ = x.shape
        
        # Reshape to process all cells through the same MLP
        x_reshaped = x.view(-1, x.shape[-1])
        
        # Process through MLP
        cell_outputs = self.network(x_reshaped)
        
        # Reshape back and sum over cells
        cell_outputs = cell_outputs.view(batch_size, num_cells)
        final_output = cell_outputs.sum(dim=1) / num_cells #take average of all cells
        
        return final_output

class CellMLPLightning(L.LightningModule):
    def __init__(self, input_size=4, hidden_sizes=[64, 64], learning_rate=1e-3, dropout_rate=0.2, target_mean=0, target_std=1, linthresh=5e-7):
        super().__init__()
        self.save_hyperparameters()
        self.model = CellMLP(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
        self.learning_rate = learning_rate
        self.target_mean = target_mean
        self.target_std = target_std
        self.linthresh = linthresh
        
        # Initialize metric tracking
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        # Calculate metrics
        r2 = self.train_r2(y_hat, y)
        mse = self.train_mse(y_hat, y)
        mae = self.train_mae(y_hat, y)
        
        # Log metrics
        self.log('loss/train', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('r2/train', r2, prog_bar=True, on_epoch=True, on_step=False)
        self.log('mse/train', mse, prog_bar=True, on_epoch=True, on_step=False)
        self.log('mae/train', mae, prog_bar=True, on_epoch=True, on_step=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        # Calculate metrics
        r2 = self.val_r2(y_hat, y)
        mse = self.val_mse(y_hat, y)
        mae = self.val_mae(y_hat, y)
        
        # Log metrics
        self.log('loss/val', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('r2/val', r2, prog_bar=True, on_epoch=True, on_step=False)
        self.log('mse/val', mse, prog_bar=True, on_epoch=True, on_step=False)
        self.log('mae/val', mae, prog_bar=True, on_epoch=True, on_step=False)

        if self.current_epoch == self.trainer.max_epochs - 1:
            plt.ioff()  # Turn off interactive mode
            fig = plt.figure(figsize=(10, 10))
            plt.scatter(y.cpu().numpy(), y_hat.detach().cpu().numpy(), alpha=0.5)
            plt.plot([y.min().cpu().numpy(), y.max().cpu().numpy()], 
                    [y.min().cpu().numpy(), y.max().cpu().numpy()], 
                    'r--', label='Perfect Prediction')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title('Predictions vs Actual')
            plt.legend()
            
            # Log figure to TensorBoard
            self.logger.experiment.log({
            "predictions_vs_actual": wandb.Image(fig)})
            plt.close(fig)  # Explicitly close the figure

            # Unnormalized plot
            fig = plt.figure(figsize=(10, 10))
            y_unnorm = y.cpu().numpy() * self.target_std + self.target_mean
            y_hat_unnorm = y_hat.detach().cpu().numpy() * self.target_std + self.target_mean
            plt.scatter(y_unnorm, y_hat_unnorm, color='blue', label='Data Points')
            plt.plot([y_unnorm.min(), y_unnorm.max()], 
                    [y_unnorm.min(), y_unnorm.max()], 
                    'r--', label='Perfect Prediction')
            plt.xlabel('True Mass flow [$\mathrm{M}_\odot/\mathrm{yr}$]', fontsize=20)
            plt.ylabel('Predicted Mass flow [$\mathrm{M}_\odot/\mathrm{yr}$]', fontsize=20)
            plt.title('True vs predicted mass flow', fontsize=26)
            plt.legend(prop={'size': 24}, frameon=False, loc='lower right')

            plt.tight_layout()
            # Set tick format to scientific notation
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            self.logger.experiment.log({
            "predictions_vs_actual_unnormalized": wandb.Image(fig)})
            plt.close(fig)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)