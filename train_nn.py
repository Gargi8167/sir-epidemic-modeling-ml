# This file contains the training procedure for the standard SIR neural network.
# It converts the dataset into PyTorch tensors, defines the optimizer and loss function,
# repeatedly updates the model, and finally returns the trained model.

import torch  # Import PyTorch for tensor creation and numerical computation.
import torch.nn as nn  # Import PyTorch's neural-network components, including MSELoss.
import torch.optim as optim  # Import PyTorch optimization algorithms such as Adam.
from src.models.nn_model import SIRNet  # Import the SIR neural-network architecture defined in nn_model.py.


def train_nn(X, y, epochs=50):
    # Define the neural-network training function.
    # X contains input features and y contains target SIR values.
    # epochs controls how many complete passes of training are performed.

    model = SIRNet()  # Create a fresh instance of the SIR neural network.

    X = torch.tensor(X, dtype=torch.float32)  # Convert input data from NumPy/other format into float32 PyTorch tensors.
    y = torch.tensor(y, dtype=torch.float32)  # Convert target values into float32 PyTorch tensors.

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use the Adam optimizer with a learning rate of 0.001.
    loss_fn = nn.MSELoss()  # Use Mean Squared Error to measure prediction error.

    for epoch in range(epochs):
        # Repeat the optimization process for the requested number of epochs.

        optimizer.zero_grad()  # Clear gradients left over from the previous training step.

        preds = model(X)  # Pass all training inputs through the network to obtain predictions.
        loss = loss_fn(preds, y)  # Compare predictions with the true targets using MSE.

        loss.backward()  # Calculate gradients of the loss with respect to all trainable model parameters.
        optimizer.step()  # Update the model parameters using the calculated gradients.

        if epoch % 10 == 0:
            # Print training progress every 10 epochs.
            print(f"Epoch {epoch}, Loss: {loss.item()}")  # Display the current epoch number and numerical loss value.

    return model  # Return the trained neural-network model.
