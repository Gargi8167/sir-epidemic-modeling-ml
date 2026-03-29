import torch
import torch.nn as nn
import torch.optim as optim
from src.models.nn_model import SIRNet

def train_nn(X, y, epochs=50):

    model = SIRNet()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        preds = model(X)
        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model