import torch
import torch.nn as nn
from .multiLayerPerceptron import MLP

class BaselineMLP(nn.Module):
    # Initialise with Node Input (+ Horizon)
    def __init__(self, nodeInputDim, horizon):
        super().__init__()
        # Define Horizon
        self.horizon = horizon

        # Define Readout MLP (Operating on Raw Node Features)
        self.readout = MLP([nodeInputDim, 128, horizon * 2])

    def forward(self, data):
        # Gather Data from Training Example (Only Require Node Features)
        x = data.x

        # Produce Predictions (Nodes, Horizon * 2) with Final Node Encodings with the Readout MLP
        out = self.readout(x)

        # Reshape Predictions -> (Nodes, Horizon, 2) and then Return
        out = out.view(-1, self.horizon, 2)
        return out
