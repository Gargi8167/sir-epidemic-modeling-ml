# This file contains the training logic for the Physics-Informed Neural Network (PINN).
# The PINN is trained using the SIR differential equations themselves as the physics constraint.
# Automatic differentiation is used to calculate dS/dt, dI/dt, and dR/dt.

import torch  # Import PyTorch for tensors and automatic differentiation.
import torch.optim as optim  # Import PyTorch optimization algorithms such as Adam.
from src.models.pinn import PINN  # Import the PINN neural-network architecture.


def pinn_loss(model, t, beta, gamma):
    # Calculate the physics-informed loss based on how well the network satisfies the SIR equations.

    t.requires_grad = True  # Tell PyTorch to track gradients with respect to time so derivatives can be calculated.

    pred = model(t)  # Ask the PINN to predict S, I, and R for every supplied time value.

    S = pred[:, 0]  # Extract the predicted susceptible population.
    I = pred[:, 1]  # Extract the predicted infected population.
    R = pred[:, 2]  # Extract the predicted recovered population.

    # Compute derivatives using autograd
    # Calculate dS/dt using PyTorch's automatic differentiation system.
    dSdt = torch.autograd.grad(S.sum(), t, create_graph=True)[0]
    # Calculate dI/dt using automatic differentiation.
    dIdt = torch.autograd.grad(I.sum(), t, create_graph=True)[0]
    # Calculate dR/dt using automatic differentiation.
    dRdt = torch.autograd.grad(R.sum(), t, create_graph=True)[0]

    N = S + I + R  # Calculate the total predicted population.

    # SIR equations residuals
    # Each residual measures how much the network's derivative differs from the corresponding SIR equation.
    loss_S = dSdt + beta * S * I / N  # Residual of dS/dt = -beta*S*I/N.
    loss_I = dIdt - (beta * S * I / N - gamma * I)  # Residual of dI/dt = beta*S*I/N - gamma*I.
    loss_R = dRdt - gamma * I  # Residual of dR/dt = gamma*I.

    loss = (loss_S**2 + loss_I**2 + loss_R**2).mean()  # Square the residuals and average them to obtain one physics loss.

    return loss  # Return the physics-informed loss value.


def train_pinn(epochs=2000):
    # Train the PINN for the requested number of epochs and return the trained model and time tensor.

    model = PINN()  # Create a new PINN model.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Create an Adam optimizer with learning rate 0.001.

    # Time input
    # Create 200 time points between 0 and 50 and reshape them into a column tensor.
    t = torch.linspace(0, 50, 200).view(-1, 1)

    beta = 0.3  # Set the transmission-rate parameter used by the SIR physics equation.
    gamma = 0.1  # Set the recovery-rate parameter used by the SIR physics equation.

    for epoch in range(epochs):
        # Repeat the PINN optimization procedure for the requested number of epochs.

        optimizer.zero_grad()  # Clear gradients from the previous training iteration.

        loss = pinn_loss(model, t, beta, gamma)  # Calculate how well the current model satisfies the SIR equations.

        loss.backward()  # Backpropagate the physics loss through the network.
        optimizer.step()  # Update the network weights using the calculated gradients.

        if epoch % 200 == 0:
            # Print progress every 200 epochs so training can be monitored.
            print(f"Epoch {epoch}, Loss: {loss.item()}")  # Display the current epoch and physics loss.

    return model, t  # Return both the trained PINN and the time points used for training.


