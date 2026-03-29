import torch
import torch.optim as optim
from src.models.pinn import PINN

def pinn_loss(model, t, beta, gamma):

    t.requires_grad = True

    pred = model(t)

    S = pred[:, 0]
    I = pred[:, 1]
    R = pred[:, 2]

    # Compute derivatives using autograd
    dSdt = torch.autograd.grad(S.sum(), t, create_graph=True)[0]
    dIdt = torch.autograd.grad(I.sum(), t, create_graph=True)[0]
    dRdt = torch.autograd.grad(R.sum(), t, create_graph=True)[0]

    N = S + I + R

    # SIR equations residuals
    loss_S = dSdt + beta * S * I / N
    loss_I = dIdt - (beta * S * I / N - gamma * I)
    loss_R = dRdt - gamma * I

    loss = (loss_S**2 + loss_I**2 + loss_R**2).mean()

    return loss


def train_pinn(epochs=2000):

    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Time input
    t = torch.linspace(0, 50, 200).view(-1, 1)

    beta = 0.3
    gamma = 0.1

    for epoch in range(epochs):

        optimizer.zero_grad()

        loss = pinn_loss(model, t, beta, gamma)

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, t