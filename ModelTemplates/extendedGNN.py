import torch
import torch.nn as nn
from .multiLayerPerceptron import MLP
from .messagePassing import MessagePassingLayer

class ExtendedGNN(nn.Module):
    # Initialise with Node Input, Edge Input & Encoding Dimensions (+ Horizon)
    def __init__(self, nodeInputDim, edgeInputDim, encodingDim, horizon):
        super().__init__()
        # Define Horizon
        self.horizon = horizon

        # Define Node & Edge Encoder MLPs
        self.nodeEncoder = MLP([nodeInputDim, 128, encodingDim])
        self.edgeEncoder = MLP([edgeInputDim, 128, encodingDim])

        # Define Message Passing Layers
        self.layers = nn.ModuleList()
        for i in range(3):
            self.layers.append(MessagePassingLayer(encodingDim, encodingDim))

        # Define Readout MLP
        self.readout = MLP([encodingDim, 128, horizon*2])

    def forward(self, data):
        # Gather Data from Training Example
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Pre-Process the Nodes & Edges with the Encoder MLPs
        x = self.nodeEncoder(x)
        edge_attr = self.edgeEncoder(edge_attr)

        # Combine Node Encodings with Contextual Information (through Edge Encodings)
        for layer in self.layers:
            x = layer(x ,edge_index, edge_attr)

        # Produce Predictions (Nodes, Horizon * 2) with Final Node Encodings with the Readout MLP
        out = self.readout(x)

        # Reshape Predictions -> (Nodes, Horizon, 2) and then Return
        out = out.view(-1, self.horizon, 2)
        return out