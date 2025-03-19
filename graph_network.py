#----------------------------------------------------------------------
# Definition of the neural network architectures
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------------------------
#I've taken this and tweaked it a little for my own use

import torch
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing, GCNConv, PPFConv, MetaLayer, EdgeConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_cluster import knn_graph, radius_graph
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger
import random
import matplotlib.pyplot as plt



#------------------------------
# Architectures considered:
#   DeepSet
#   PointNet
#   EdgeNet
#   EdgePointLayer (a mix of the two above)
#   Convolutional Graph Network
#   Metalayer (graph network)
#
# See pytorch-geometric documentation for more info
# pytorch-geometric.readthedocs.io/
#-----------------------------

#--------------------------------------------
# Message passing architectures
#--------------------------------------------

# PointNet layer
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, mid_channels, out_channels, use_mod=1):
        # Message passing with "max" aggregation.
        super(PointNetLayer, self).__init__('max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3, or 1 if only modulus is used).
        if use_mod:
            self.mlp = Sequential(Linear(in_channels+1, mid_channels),
                                  ReLU(),
                                  Linear(mid_channels, mid_channels),
                                  ReLU(),
                                  Linear(mid_channels, out_channels))
        else:
            self.mlp = Sequential(Linear(in_channels+3, mid_channels),
                                  ReLU(),
                                  Linear(mid_channels, mid_channels),
                                  ReLU(),
                                  Linear(mid_channels, out_channels))

        self.messages = 0.
        self.input = 0.
        self.use_mod = use_mod

    def forward(self, x, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_j, pos_i):
        # x_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.
        if self.use_mod:
            input = input[:,0]**2.+input[:,1]**2.+input[:,2]**2.
            input = input.view(input.shape[0], 1)

        if x_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([x_j, input], dim=-1)

        self.input = input
        self.messages = self.mlp(input)

        return self.messages


# Edge convolution layer
class EdgeLayer(MessagePassing):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(EdgeLayer, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Sequential(Linear(2 * in_channels, mid_channels),
                       ReLU(),
                       Linear(mid_channels, mid_channels),
                       ReLU(),
                       Linear(mid_channels, out_channels))
        self.messages = 0.
        self.input = 0.

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        input = torch.cat([x_i, x_j - x_i], dim=-1)  # tmp has shape [E, 2 * in_channels]

        self.input = input
        self.messages = self.mlp(input)

        return self.messages


# Mix of EdgeNet and PointNet, using only modulus of the distance between neighbors
class EdgePointLayer(MessagePassing):
    def __init__(self, in_channels, mid_channels, out_channels, use_mod=1):
        # Message passing with "max" aggregation.
        super(EdgePointLayer, self).__init__('max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3, or 1 if only modulus is used).
        self.mlp = Sequential(Linear(2*in_channels-2, mid_channels),
                              ReLU(),
                              Linear(mid_channels, mid_channels),
                              ReLU(),
                              Linear(mid_channels, out_channels))

        self.messages = 0.
        self.input = 0.
        self.use_mod = use_mod

    def forward(self, x, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        pos_i, pos_j = x_i[:,:3], x_j[:,:3]

        input = pos_j - pos_i  # Compute spatial relation.
        input = input[:,0]**2.+input[:,1]**2.+input[:,2]**2.
        input = input.view(input.shape[0], 1)
        input = torch.cat([x_i, x_j[:,3:], input], dim=-1)

        self.input = input
        self.messages = self.mlp(input)

        return self.messages


# Node model for the MetaLayer
class NodeModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(NodeModel, self).__init__()
        #self.node_mlp_1 = Sequential(Linear(in_channels,hidden_channels),  LeakyReLU(0.2), Linear(hidden_channels,hidden_channels),LeakyReLU(0.2), Linear(mid_channels,out_channels))
        #self.node_mlp_2 = Sequential(Linear(303,500), LeakyReLU(0.2), Linear(500,500),LeakyReLU(0.2), Linear(500,1))

        self.mlp = Sequential(Linear(in_channels*2, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, latent_channels))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index

        # define interaction tensor; every pair contains features from input and
        # output node together with
        #out = torch.cat([x[row], x[col], edge_attr], dim=1)
        out = torch.cat([x[row], x[col]], dim=1)
        #print("node pre", x.shape, out.shape)

        # take interaction feature tensor and embedd it into another tensor
        #out = self.node_mlp_1(out)
        out = self.mlp(out)
        #print("node mlp", out.shape)

        # compute the mean,sum and max of each embed feature tensor for each node
        out1 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out3 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out4 = scatter_min(out, col, dim=0, dim_size=x.size(0))[0]

        # every node contains a feature tensor with the pooling of the messages from
        # neighbors, its own state, and a global feature
        out = torch.cat([x, out1, out3, out4, u[batch]], dim=1)
        #print("node post", out.shape)

        #return self.node_mlp_2(out)
        return out

# Global model for the MetaLayer
class GlobalModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(GlobalModel, self).__init__()
        #self.global_mlp = Seq(Linear(5, 500), LeakyReLU(0.2), Linear(500,500),LeakyReLU(0.2), Linear(500,2))

        self.global_mlp = Sequential(Linear((in_channels+latent_channels*3+2)*3+2, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, latent_channels))

        print("we",(in_channels+latent_channels*3+2), (in_channels+latent_channels*3+2)*3+2)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out1 = scatter_mean(x, batch, dim=0)
        out3 = scatter_max(x, batch, dim=0)[0]
        out4 = scatter_min(x, batch, dim=0)[0]
        out = torch.cat([u, out1, out3, out4], dim=1)
        #print("global pre",out.shape, x.shape, u.shape)
        out = self.global_mlp(out)
        #print("global post",out.shape)
        return out


#--------------------------------------------
# General Graph Neural Network architecture
#--------------------------------------------
class ModelGNN(torch.nn.Module):
    def __init__(self, use_model, node_features, n_layers, hidden_channels=300, latent_channels=100, loop=False):
        super(ModelGNN, self).__init__()

        # Graph layers
        layers = []
        in_channels = node_features
        for i in range(n_layers):

            # Choose the model
            if use_model=="DeepSet":
                lay = Sequential(
                    Linear(in_channels, hidden_channels),
                	ReLU(),
                	Linear(hidden_channels, hidden_channels),
                	ReLU(),
                	Linear(hidden_channels, latent_channels))

            elif use_model=="GCN":
                lay = GCNConv(in_channels, latent_channels)

            elif use_model=="PointNet":
                lay = PointNetLayer(in_channels, hidden_channels, latent_channels)

            elif use_model=="EdgeNet":
                lay = EdgeLayer(in_channels, hidden_channels, latent_channels)
                #lay = EdgeConv(Sequential(Linear(2*in_channels, hidden_channels),ReLU(),Linear(hidden_channels, hidden_channels),ReLU(),Linear(hidden_channels, latent_channels)))  # Using the pytorch-geometric implementation, same result

            elif use_model=="EdgePoint":
                lay = EdgePointLayer(in_channels, hidden_channels, latent_channels)

            elif use_model=="MetaNet":
                if use_model=="MetaNet" and i==2:   in_channels = 610
                #lay = MetaLayer(node_model=NodeModel(in_channels, hidden_channels, latent_channels), global_model=GlobalModel(in_channels, hidden_channels, latent_channels))
                lay = MetaLayer(node_model=NodeModel(in_channels, hidden_channels, latent_channels))

            else:
                print("Model not known...")

            layers.append(lay)
            in_channels = latent_channels
            if use_model=="MetaNet":    in_channels = (node_features+latent_channels*3+2)


        self.layers = ModuleList(layers)

        #lin_in = latent_channels*3+2
        lin_in = latent_channels*3
        if use_model=="MetaNet":    lin_in = (in_channels +latent_channels*3 +2)*3 + 2
        if use_model=="MetaNet" and n_layers==3:    lin_in = 2738
        self.lin = Sequential(Linear(lin_in, latent_channels),
                              ReLU(),
                              Linear(latent_channels, latent_channels),
                              ReLU(),
                              Linear(latent_channels, 1))

        self.pooled = 0.
        self.h = 0.
        self.loop = loop
        if use_model=="PointNet" or use_model=="GCN":    self.loop = True
        self.namemodel = use_model

    def forward(self, data):
        x, pos, batch, u = data.x, data.pos, data.batch, data.u
        
        # Use provided edge_index instead of computing it
        edge_index = data.edge_index
        
        # Rest of the method remains the same
        for layer in self.layers:
            if self.namemodel=="DeepSet":
                x = layer(x)
            elif self.namemodel=="PointNet":
                x = layer(x=x, pos=pos, edge_index=edge_index)
            elif self.namemodel=="MetaNet":
                x, dumb, u = layer(x, edge_index, None, u, batch)
            else:
                x = layer(x=x, edge_index=edge_index)
            self.h = x
            x = x.relu()


        # Mix different global pooling layers
        addpool = global_add_pool(x, batch) # [num_examples, hidden_channels]
        meanpool = global_mean_pool(x, batch)
        maxpool = global_max_pool(x, batch)
        self.pooled = torch.cat([addpool, meanpool, maxpool], dim=1)
        #self.pooled = torch.cat([addpool, meanpool, maxpool, u], dim=1)

        # Final linear layer
        return self.lin(self.pooled)
    
from torch_geometric.data import Data




def create_dataset(mass_flows, neighbour_data, edges, global_data):

    dataset = []

        # For each halo in the simulation:
    for i in range(len(mass_flows)):

            # Create the graph of the halo
            # x: features (includes positions), pos: positions, u: global quantity
            graph = Data(x=torch.tensor(neighbour_data[i], dtype=torch.float32), pos=torch.tensor(neighbour_data[i][:,:3], dtype=torch.float32), y=torch.tensor(mass_flows[i], dtype=torch.float32), u=torch.tensor(global_data[i], dtype=torch.float32), edge_index=torch.tensor(edges[i], dtype=torch.long))

            dataset.append(graph)

    print("Total number of snapshots", len(dataset))

    # Number of features
    node_features = neighbour_data.shape[2]

    return dataset, node_features

from torch_geometric.data import Data, DataLoader

def split_datasets(dataset, valid_size=0.1, test_size=0.1, batch_size=32):

    random.shuffle(dataset)

    num_train = len(dataset)
    split_valid = int(np.floor(valid_size * num_train))
    split_test = split_valid + int(np.floor(test_size * num_train))

    train_dataset = dataset[split_test:]
    valid_dataset = dataset[:split_valid]
    test_dataset = dataset[split_valid:split_test]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader

class GNNLightningModule(pl.LightningModule):
    def __init__(self, model_name, node_features, n_layers, hidden_channels=300, latent_channels=100, 
                 learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize the GNN model
        self.model = ModelGNN(model_name, node_features, n_layers, 
                            hidden_channels, latent_channels)
        
    def forward(self, data):
        return self.model(data)
    
    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        y_hat = y_hat.squeeze(-1)  # Change from [9, 1] to [9]
        loss = F.mse_loss(y_hat, batch.y)
        
        # Add batch_size parameter to logging
        batch_size = batch.y.size(0)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        
        if batch_idx % 100 == 0:
            wandb.log({
                "predictions": wandb.Histogram(y_hat.detach().cpu().numpy()),
                "actuals": wandb.Histogram(batch.y.detach().cpu().numpy())
            })
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        y_hat = y_hat.squeeze(-1)
        val_loss = F.mse_loss(y_hat, batch.y)
        mae = F.l1_loss(y_hat, batch.y)
        
        # Add batch_size parameter to logging
        batch_size = batch.y.size(0)
        self.log('val_loss', val_loss, prog_bar=True, batch_size=batch_size)
        self.log('val_mae', mae, prog_bar=True, batch_size=batch_size)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        y_hat = y_hat.squeeze(-1)
        test_loss = F.mse_loss(y_hat, batch.y)
        mae = F.l1_loss(y_hat, batch.y)
        
        # Add batch_size parameter to logging
        batch_size = batch.y.size(0)
        self.log('test_loss', test_loss, batch_size=batch_size)
        self.log('test_mae', mae, batch_size=batch_size)
        
        # Store predictions and true values as instance attributes
        if not hasattr(self, 'test_preds'):
            self.test_preds = []
            self.test_targets = []
        self.test_preds.append(y_hat)
        self.test_targets.append(batch.y)
        
        return {'test_loss': test_loss, 'test_mae': mae}
    
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
        plt.plot([targets_np.min(), targets_np.max()], [targets_np.min(), targets_np.max()], 'r--', lw=2)
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
            weight_decay=self.weight_decay  # Add L2 regularization
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class GraphDataModule(pl.LightningDataModule):
    def __init__(self, mass_flows, neighbour_data, edges, global_data, 
                 batch_size=32, valid_size=0.1, test_size=0.1):
        super().__init__()
        self.mass_flows = mass_flows
        self.neighbour_data = neighbour_data
        self.edges = edges
        self.global_data = global_data
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.test_size = test_size
        
    def setup(self, stage=None):
        # Create full dataset
        dataset = []
        for i in range(len(self.mass_flows)):
            graph = Data(
                x=torch.tensor(self.neighbour_data[i], dtype=torch.float32),
                pos=torch.tensor(self.neighbour_data[i][:,:3], dtype=torch.float32),
                y=torch.tensor(self.mass_flows[i], dtype=torch.float32),
                u=torch.tensor(self.global_data[i], dtype=torch.float32),
                edge_index=torch.tensor(self.edges[i], dtype=torch.long)
            )
            dataset.append(graph)
            
        # Calculate splits
        n_samples = len(dataset)
        n_valid = int(self.valid_size * n_samples)
        n_test = int(self.test_size * n_samples)
        n_train = n_samples - n_valid - n_test
        
        # Set generator for reproducibility
        #generator = torch.Generator().manual_seed(42)  # You can use any seed number
        
        # Split dataset with the generator
        self.train_data, self.val_data, self.test_data = random_split(
            dataset, 
            [n_train, n_valid, n_test],
            #generator=generator
        )
        
        # Log dataset info
        wandb.config.update({
            "train_size": n_train,
            "val_size": n_valid,
            "test_size": n_test,
            "total_samples": n_samples,
            "random_seed": 42  # Log the seed used
        })
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=len(self.val_data))
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=len(self.test_data))

def train_model(mass_flows, neighbour_data, edges, global_data):
    # Initialize wandb with smaller network dimensions
    wandb.init(
        project="graph-nn-mass-flow",
        config={
            "architecture": "EdgeNet",
            "dataset": "mass_flow",
            "learning_rate": 1e-3,
            "weight_decay": 0,
            "batch_size": 32,
            "n_layers": 5,
            "hidden_channels": 64,    # Reduced from 300
            "latent_channels": 32,    # Reduced from 100
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau"
        }
    )
    
    # Initialize data module
    data_module = GraphDataModule(
        mass_flows=mass_flows,
        neighbour_data=neighbour_data,
        edges=edges,
        global_data=global_data,
        batch_size=32,
        valid_size=0.1,
        test_size=0.2
    )
    
    # Initialize model with smaller dimensions
    model = GNNLightningModule(
        model_name="EdgeNet",
        node_features=neighbour_data.shape[2],
        n_layers=5,
        hidden_channels=64,    # Reduced from 300
        latent_channels=32,    # Reduced from 100
        learning_rate=1e-3,
        weight_decay=0
    )
    
    # Setup wandb logger
    wandb_logger = WandbLogger(project="graph-nn-mass-flow", log_model=True)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='gnn-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5,
        accumulate_grad_batches=1,
        precision=16 if torch.cuda.is_available() else 32,
        log_every_n_steps=1  # Changed from default 50 to 1 to see logs for every batch
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)
    
    # Close wandb run
    wandb.finish()