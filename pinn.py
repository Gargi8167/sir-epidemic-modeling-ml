# This file defines the neural-network architecture used by the Physics-Informed Neural Network (PINN).
# Unlike the standard NN, the PINN is designed to learn outputs while respecting the SIR equations.

import torch  # Import PyTorch for tensor operations and automatic differentiation.
import torch.nn as nn  # Import PyTorch's neural-network layers and base Module class.


class PINN(nn.Module):
    # Define the PINN model as a PyTorch neural network.

    def __init__(self):
        # Define the constructor that builds the PINN architecture.
        super().__init__()  # Initialize the parent nn.Module class.

        # Create a feed-forward network that maps time t to the three SIR populations.
        self.net = nn.Sequential(
            nn.Linear(1, 64),  # First layer: one input value (time) -> 64 hidden neurons.
            nn.Tanh(),  # Apply Tanh activation to model nonlinear SIR dynamics.
            nn.Linear(64, 64),  # Second hidden layer: 64 neurons -> 64 neurons.
            nn.Tanh(),  # Apply another nonlinear activation.
            nn.Linear(64, 3)  # Output layer: predict susceptible, infected, and recovered populations.
        )

    def forward(self, t):
        # Define the forward pass using time as the input.
        return self.net(t)  # Return the network's predicted S, I, and R values.
