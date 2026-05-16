import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dims, dropout = 0.0):
        # Initialise with Linear Dimensions & Dropout
        super().__init__()
        # Define Layers
        layers = []
        for i in range(len(dims) - 1):
            # Add Layer
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < (len(dims) - 2):
                # Add Activation & Dropout to Non-Final Layers
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        # Finalise MLP Module Structure
        self.net = nn.Sequential(*layers)
    
    # Define Forward Pass
    def forward(self, x):
        return self.net(x)