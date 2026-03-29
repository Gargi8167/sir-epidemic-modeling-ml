import numpy as np
import torch

from src.simulation.deterministic import solve_sir
from src.data.generator import generate_dataset
from src.training.train_nn import train_nn
from src.training.train_pinn import train_pinn
from src.utils.plotting import plot_final_comparison


def main():
    print("Running full comparison...")

    # Initial conditions
    S0, I0, R0 = 990, 10, 0
    beta = 0.3
    gamma = 0.1

    # -------------------------
    # 1. ODE (Ground Truth)
    # -------------------------
    t, S_ode, I_ode, R_ode = solve_sir(S0, I0, R0, beta, gamma)
    ode = np.column_stack([S_ode, I_ode, R_ode])

    # -------------------------
    # 2. Neural Network
    # -------------------------
    print("Training NN...")
    X, y = generate_dataset(200)
    nn_model = train_nn(X, y)

    X_test = np.column_stack([
        t,
        np.full_like(t, beta),
        np.full_like(t, gamma)
    ])

    X_test = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        nn_pred = nn_model(X_test).detach().numpy()

    # -------------------------
    # 3. PINN
    # -------------------------
    print("Training PINN...")
    pinn_model, t_pinn = train_pinn()

    with torch.no_grad():
        pinn_pred = pinn_model(t_pinn).detach().numpy()

    t_plot = t_pinn.detach().numpy()

    # -------------------------
    # FINAL PLOT
    # -------------------------
    plot_final_comparison(
        t_plot,
        ode[:len(t_plot)],
        nn_pred[:len(t_plot)],
        pinn_pred
    )


if __name__ == "__main__":
    main()