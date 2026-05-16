import torch
from torch_geometric.nn import MessagePassing
from .multiLayerPerceptron import MLP

class MessagePassingLayer(MessagePassing):
    # Initialise with Node & Edge Feature Dimensions
    def __init__(self, node_dim, edge_dim):
        # Set Aggregation Method to Addition
        super().__init__(aggr = "add")
        # Define Shapes for Message & Update MLPs
        self.messageMLP = MLP([(2*node_dim) + edge_dim, node_dim, node_dim])
        self.updateMLP = MLP([node_dim, node_dim])

    def forward(self, x, edge_index, edge_attr):
        # Perform Message Passing Pass (Creates Messages, Aggregates & Updates Embeddings)
        return self.propagate(edge_index, x=x, edge_attr = edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate Node Embeddings (Target & Source) with Edge Features and Use Message MLP
        return self.messageMLP(torch.cat([x_i, x_j, edge_attr], dim = 1))
        
    def update(self, aggr_out, x):
        # Use Update MLP on the Aggregated Messages (Same Shape as Embeddings) and Add them to Embeddings
        return x + self.updateMLP(aggr_out)
