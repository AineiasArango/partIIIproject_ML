#----------------------------------------------------------------------
# Definition of the neural network architectures
# Author: Aineias
# Last update: 20-03-25
#----------------------------------------------------------------------

import torch
from torch.nn import Sequential, Linear, ReLU, ModuleList, Dropout
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError
import matplotlib.pyplot as plt
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from torch.utils.data import DataLoader, random_split

#--------------------------------------------
# Cell MLP architecture
#--------------------------------------------

class CellMLP(torch.nn.Module):
    def __init__(self, input_size=4, hidden_sizes=[64, 64], dropout_rate=0.2):
        super(CellMLP, self).__init__()
        layers = []
        
        # Input layer
        layers.append(Linear(input_size, hidden_sizes[0]))
        layers.append(ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(ReLU())
            layers.append(Dropout(p=dropout_rate))
        
        # Output layer
        layers.append(Linear(hidden_sizes[-1], 1))
        
        self.network = Sequential(*layers)
    
    def forward(self, x):
        batch_size, num_cells, _ = x.shape
        x_reshaped = x.view(-1, x.shape[-1])
        cell_outputs = self.network(x_reshaped)
        cell_outputs = cell_outputs.view(batch_size, num_cells)
        final_output = cell_outputs.sum(dim=1) / num_cells
        return final_output

class CellMLPLightning(pl.LightningModule):
    def __init__(self, input_size=4, hidden_sizes=[64, 64], learning_rate=1e-3, 
                 dropout_rate=0.2, target_mean=0, target_std=1, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.model = CellMLP(input_size=input_size, 
                            hidden_sizes=hidden_sizes, 
                            dropout_rate=dropout_rate)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_mean = target_mean
        self.target_std = target_std
        
        # Metrics
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        # Calculate and log metrics
        r2 = self.train_r2(y_hat, y)
        mse = self.train_mse(y_hat, y)
        mae = self.train_mae(y_hat, y)
        
        # Add batch_size parameter to logging
        batch_size = y.size(0)
        self.log('loss/train', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
        self.log('r2/train', r2, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
        self.log('mse/train', mse, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
        self.log('mae/train', mae, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
        
        if batch_idx % 100 == 0:
            wandb.log({
                "predictions": wandb.Histogram(y_hat.detach().cpu().numpy()),
                "actuals": wandb.Histogram(y.detach().cpu().numpy())
            })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        
        # Calculate metrics
        r2 = self.val_r2(y_hat, y)
        mse = self.val_mse(y_hat, y)
        mae = self.val_mae(y_hat, y)
        
        # Add batch_size parameter to logging
        batch_size = y.size(0)
        self.log('loss/val', val_loss, prog_bar=True, batch_size=batch_size)
        self.log('r2/val', r2, prog_bar=True, batch_size=batch_size)
        self.log('mse/val', mse, prog_bar=True, batch_size=batch_size)
        self.log('mae/val', mae, prog_bar=True, batch_size=batch_size)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat, y)
        
        # Store predictions and targets for visualization
        if not hasattr(self, 'test_preds'):
            self.test_preds = []
            self.test_targets = []
        self.test_preds.append(y_hat)
        self.test_targets.append(y)
        
        # Add batch_size parameter to logging
        batch_size = y.size(0)
        self.log('test_loss', test_loss, batch_size=batch_size)
        
        return test_loss
    
    def on_test_epoch_end(self):
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.test_preds)
        all_targets = torch.cat(self.test_targets)
        
        # Convert to numpy for plotting
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        
        # Create scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(targets_np, preds_np, alpha=0.5)
        plt.plot([targets_np.min(), targets_np.max()], 
                [targets_np.min(), targets_np.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted Values')
        
        # Log to wandb
        wandb.log({
            "true_vs_predicted": wandb.Image(plt),
            "predictions_vs_true": wandb.plot.scatter(
                wandb.Table(data=[[x, y] for x, y in zip(targets_np, preds_np)],
                          columns=["True Values", "Predictions"]),
                "True Values",
                "Predictions"
            )
        })
        
        plt.close()
        
        # Clear the stored predictions and targets
        self.test_preds = []
        self.test_targets = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val"
            }
        }

class CellDataModule(pl.LightningDataModule):
    def __init__(self, cell_data, mass_flows, batch_size=32, valid_size=0.1, test_size=0.1, seed=42):
        """
        Args:
            cell_data: numpy array of shape (n_samples, n_cells, n_features)
            mass_flows: numpy array of shape (n_samples,)
            batch_size: size of batches for training
            valid_size: proportion of data to use for validation
            test_size: proportion of data to use for testing
        """
        super().__init__()
        self.cell_data = torch.tensor(cell_data, dtype=torch.float32)
        self.mass_flows = torch.tensor(mass_flows, dtype=torch.float32)
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.seed = seed
    def setup(self, stage=None):
        # Create full dataset
        dataset = list(zip(self.cell_data, self.mass_flows))
            
        # Calculate splits
        n_samples = len(dataset)
        n_valid = int(self.valid_size * n_samples)
        n_test = int(self.test_size * n_samples)
        n_train = n_samples - n_valid - n_test
        
        # Split dataset
        generator = torch.Generator().manual_seed(self.seed)  # You can use any integer as seed
        self.train_data, self.val_data, self.test_data = random_split(
            dataset, 
            [n_train, n_valid, n_test],
            generator=generator
        )
        
        # Log dataset info
        wandb.config.update({
            "train_size": n_train,
            "val_size": n_valid,
            "test_size": n_test,
            "total_samples": n_samples,
            "n_cells": self.cell_data.shape[1],
            "n_features": self.cell_data.shape[2]
        })
    
    def train_dataloader(self):
        return DataLoader(self.train_data, 
                        batch_size=self.batch_size, 
                        shuffle=True,
                        num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, 
                        batch_size=self.batch_size,
                        num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, 
                        batch_size=self.batch_size,
                        num_workers=4)

def prepare_data(cell_data, mass_flows, batch_size=64, valid_size=0.1, test_size=0.2, seed=42):
    """
    Prepare the data for training the CellMLP model.
    
    Args:
        cell_data: numpy array of shape (n_samples, n_cells, n_features)
        mass_flows: numpy array of shape (n_samples,)
        batch_size: size of batches for training
    
    Returns:
        data_module: CellDataModule instance
    """
    # Create data module
    data_module = CellDataModule(
        cell_data=cell_data,
        mass_flows=mass_flows,
        batch_size=batch_size,
        valid_size=valid_size,
        test_size=test_size,
        seed=seed
    )
    
    # Setup the data module
    data_module.setup()
    
    return data_module

def train_model(cell_data, mass_flows, input_size, hidden_sizes=[64, 64], max_epochs=100, learning_rate=1e-3, weight_decay=1e-4, dropout_rate=0.2, valid_size=0.1, test_size=0.2, seed=42):
    # Initialize wandb
    wandb.init(
        project="cell-mlp-mass-flow",
        config={
            "architecture": "CellMLP",
            "dataset": "mass_flow",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "hidden_sizes": hidden_sizes,
            "dropout_rate": dropout_rate,
            "max_epochs": max_epochs
        }
    )
    
    data_module = prepare_data(cell_data, mass_flows, valid_size=valid_size, test_size=test_size, seed=seed)

    # Initialize model
    model = CellMLPLightning(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate
    )
    
    # Setup wandb logger
    wandb_logger = WandbLogger(project="cell-mlp-mass-flow", log_model=True)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='loss/val',
        dirpath='checkpoints',
        filename='cellmlp-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5,
        precision=16 if torch.cuda.is_available() else 32
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    test_results = trainer.test(model, data_module)
    
    # Close wandb run
    wandb.finish()
    
    return test_results[0]['test_loss']

#------------------------------------------------------------------------------------------------
#Deepset variable length MLP architecture
#------------------------------------------------------------------------------------------------
#This architecture can handle different neighbour numbers in the same dataset

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CosmologySnapshotDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # list of (num_cells, 5) arrays/tensors
        self.labels = labels  # list/array of scalars

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    padded_xs = pad_sequence(xs, batch_first=True)  # (B, max_cells, 5)

    # Create binary mask: 1 where real, 0 where padded
    mask = torch.zeros(padded_xs.shape[:2], dtype=torch.float)
    for i, x in enumerate(xs):
        mask[i, :x.shape[0]] = 1.0

    ys = torch.stack(ys)  # (B,)
    return padded_xs, mask, ys


import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError, R2Score
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer
import numpy as np

class DeepSet(pl.LightningModule):
    def __init__(self, input_dim=5, phi_hidden_dims=[64, 64, 64], phi_output_dim=1, 
                 rho_hidden_dims=[64], output_dim=1, lr=1e-3, dropout_rate=0, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        phi_layers = []
        # Input layer
        phi_layers.append(Linear(input_dim, phi_hidden_dims[0]))
        phi_layers.append(ReLU())
        # Hidden layers
        for i in range(len(phi_hidden_dims)-1):
            phi_layers.append(Linear(phi_hidden_dims[i], phi_hidden_dims[i+1]))
            phi_layers.append(ReLU())
            phi_layers.append(Dropout(p=self.hparams.dropout_rate))
        # Output layer
        phi_layers.append(Linear(phi_hidden_dims[-1], phi_output_dim))

        rho_layers = []
        # Input layer
        rho_layers.append(Linear(phi_output_dim, rho_hidden_dims[0]))
        rho_layers.append(ReLU())
        # Hidden layers (if any)
        for i in range(len(rho_hidden_dims)-1):
            rho_layers.append(Linear(rho_hidden_dims[i], rho_hidden_dims[i+1]))
            rho_layers.append(ReLU())
        # Output layer (using the last hidden dimension)
        rho_layers.append(Linear(rho_hidden_dims[-1], output_dim))


        self.phi = nn.Sequential(*phi_layers)
        self.rho = nn.Sequential(*rho_layers)

        # Metrics
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.test_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()

        self.test_preds = []
        self.test_targets = []

    def forward(self, x, mask):
        phi_x = self.phi(x)  # (B, N, H)
        phi_x = phi_x * mask.unsqueeze(-1)  # Masked
        summed = phi_x.sum(dim=1)  # (B, H)
        return self.rho(summed).squeeze(-1)  # (B,)

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask)
        loss = F.mse_loss(y_hat, y)
        
        # Calculate metrics
        self.train_mae.update(y_hat, y)
        
        # Log metrics
        batch_size = y.size(0)
        self.log('loss/train', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
        
        if batch_idx % 100 == 0:
            wandb.log({
                "predictions": wandb.Histogram(y_hat.detach().cpu().numpy()),
                "actuals": wandb.Histogram(y.detach().cpu().numpy())
            })
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask)
        val_loss = F.mse_loss(y_hat, y)
        
        # Calculate metrics
        self.val_mae.update(y_hat, y)
        self.val_r2.update(y_hat, y)
        
        # Log metrics
        batch_size = y.size(0)
        self.log('loss/val', val_loss, prog_bar=True, batch_size=batch_size)
        self.log('mae/val', self.val_mae.compute(), prog_bar=True, batch_size=batch_size)
        self.log('r2/val', self.val_r2.compute(), prog_bar=True, batch_size=batch_size)
        
        return val_loss

    def test_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask)
        test_loss = F.mse_loss(y_hat, y)
        
        self.test_mae.update(y_hat, y)
        self.test_r2.update(y_hat, y)
        
        # Store predictions for visualization
        self.test_preds.append(y_hat.cpu())
        self.test_targets.append(y.cpu())
        
        # Log metrics
        batch_size = y.size(0)
        self.log('loss/test', test_loss, batch_size=batch_size)
        
        return test_loss

    def on_test_epoch_end(self):
        # Log final metrics
        self.log('mae/test', self.test_mae.compute(), prog_bar=True)
        self.log('r2/test', self.test_r2.compute(), prog_bar=True)
        
        # Concatenate all predictions and targets
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)
        
        # Convert to numpy for plotting
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        
        # Create scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(targets_np, preds_np, alpha=0.5)
        plt.plot([targets_np.min(), targets_np.max()], 
                [targets_np.min(), targets_np.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted Values')
        
        # Log to wandb
        wandb.log({
            "true_vs_predicted": wandb.Image(plt),
            "predictions_vs_true": wandb.plot.scatter(
                wandb.Table(data=[[x, y] for x, y in zip(targets_np, preds_np)],
                          columns=["True Values", "Predictions"]),
                "True Values",
                "Predictions"
            )
        })
        
        plt.close()
        
        # Clear stored predictions
        self.test_preds.clear()
        self.test_targets.clear()
        self.test_mae.reset()
        self.test_r2.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val"
            }
        }

#If you have multiple pieces of data with different dimensions, just add them in a list like:
#data = list(array_64) + list(array_128) + list(array_256)
#labels = list(array_1) + list(array_2) + list(array_3)
#After inputting data and labels, you must input the hyperparameters: (input_dim, max_epochs, phi_hidden_dims, phi_output_dim, rho_hidden_dims, learning_rate, weight_decay)
def train_deepset(data, labels, input_dim=5, max_epochs=100, phi_hidden_dims=[64, 64, 64], phi_output_dim=1, rho_hidden_dims=[64], learning_rate=1e-3, weight_decay=1e-4):
    # Initialize wandb
    wandb.init(
        project="deepset-cosmology",
        config={
            "architecture": "DeepSet",
            "dataset": "cosmology",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "input_dim": input_dim,
            "phi_hidden_dims": phi_hidden_dims,
            "phi_output_dim": phi_output_dim,
            "rho_hidden_dims": rho_hidden_dims,
            "max_epochs": max_epochs
        }
    )
    
    # Create dataset and split
    dataset = CosmologySnapshotDataset(data, labels)
    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=16, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=16, collate_fn=collate_fn)
    
    # Initialize model with weight decay
    model = DeepSet(input_dim=input_dim, phi_hidden_dims=phi_hidden_dims, phi_output_dim=phi_output_dim, rho_hidden_dims=rho_hidden_dims, lr=learning_rate, weight_decay=weight_decay)
    
    # Setup wandb logger
    wandb_logger = WandbLogger(project="deepset-cosmology", log_model=True)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='loss/val',
        dirpath='checkpoints',
        filename='deepset-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5,
        precision=16 if torch.cuda.is_available() else 32
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    test_results = trainer.test(model, test_loader)
    
    # Close wandb run
    wandb.finish()
    
    return test_results[0]['loss/test']
