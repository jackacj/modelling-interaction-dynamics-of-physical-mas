import torch
import torch.nn as nn
from .multiLayerPerceptron import MLP

class ExtendedMLP(nn.Module):
    # Initialise with Node Input & Encoding Dimensions (+ Horizon)
    def __init__(self, nodeInputDim, encodingDim, horizon):
        super().__init__()
        # Define Horizon
        self.horizon = horizon

        # Define Node Encoder MLP
        self.nodeEncoder = MLP([nodeInputDim, 128, encodingDim])

        # Define Readout MLP
        self.readout = MLP([encodingDim, 128, horizon * 2])

    def forward(self, data):
        # Gather Data from Training Example (Only Require Node Features)
        x = data.x

        # Pre-Process the Nodes with the Encoder MLP
        x = self.nodeEncoder(x)

        # Produce Predictions (Nodes, Horizon * 2) with Final Node Encodings with the Readout MLP
        out = self.readout(x)

        # Reshape Predictions -> (Nodes, Horizon, 2) and then Return
        out = out.view(-1, self.horizon, 2)
        return out