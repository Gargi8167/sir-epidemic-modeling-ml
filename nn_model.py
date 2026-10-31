# This file defines the standard feed-forward neural network used to learn SIR dynamics.
# The network receives three inputs (time, beta, gamma) and predicts three outputs (S, I, R).

import torch.nn as nn  # Import PyTorch's neural-network module and layer definitions.


class SIRNet(nn.Module):
    # Define the SIR neural network by inheriting from PyTorch's base neural-network class.

    def __init__(self):
        # Define the constructor that creates the network architecture.
        super().__init__()  # Initialize the parent nn.Module class.

        # Build the neural network as a sequence of layers.
        self.net = nn.Sequential(
            nn.Linear(3, 128),  # First fully connected layer: 3 input features -> 128 hidden neurons.
            nn.Tanh(),  # Apply the Tanh activation function to introduce non-linearity.
            nn.Linear(128, 128),  # Second fully connected layer: 128 neurons -> 128 neurons.
            nn.Tanh(),  # Apply another Tanh activation function.
            nn.Linear(128, 3)  # Final fully connected layer: 128 hidden neurons -> S, I, R outputs.
        )

    def forward(self, x):
        # Define the forward pass: describe how an input tensor moves through the network.
        return self.net(x)  # Pass the input through the sequential network and return the predictions.
