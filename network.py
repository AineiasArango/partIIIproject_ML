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

def prepare_data(cell_data, mass_flows, batch_size=32, valid_size=0.1, test_size=0.2, seed=42):
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