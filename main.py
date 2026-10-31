# This is the main entry point of the project.
# It runs the deterministic ODE model, trains the neural network,
# trains the physics-informed neural network (PINN), and creates the final comparison plot.

import numpy as np  # Import NumPy for array manipulation and combining model outputs.
import torch  # Import PyTorch for tensors and inference without gradient tracking.

# Import the deterministic SIR solver from the project's deterministic-model module.
from src.simulation.deterministic import solve_sir
# Import the function that generates training data for the neural network.
from src.data.generator import generate_dataset
# Import the function that trains the standard neural network.
from src.training.train_nn import train_nn
# Import the function that trains the physics-informed neural network.
from src.training.train_pinn import train_pinn
# Import the function that plots the final ODE-versus-NN-versus-PINN comparison.
from src.utils.plotting import plot_final_comparison


def main():
    # Define the main workflow of the project.
    print("Running full comparison...")  # Tell the user that the complete comparison has started.

    # Initial conditions
    # Set the initial number of susceptible, infected, and recovered people.
    S0, I0, R0 = 990, 10, 0
    # Set beta, the transmission/infection-rate parameter.
    beta = 0.3
    # Set gamma, the recovery-rate parameter.
    gamma = 0.1

    # -------------------------
    # 1. ODE (Ground Truth)
    # -------------------------
    # Solve the deterministic SIR equations to obtain the reference trajectory.
    t, S_ode, I_ode, R_ode = solve_sir(S0, I0, R0, beta, gamma)
    # Combine the three ODE population arrays into one 2D array with columns S, I, and R.
    ode = np.column_stack([S_ode, I_ode, R_ode])

    # -------------------------
    # 2. Neural Network
    # -------------------------
    print("Training NN...")  # Inform the user that standard neural-network training is beginning.
    # Generate 200 synthetic training examples for the neural network.
    X, y = generate_dataset(200)
    # Train the SIR neural network using the generated input-output dataset.
    nn_model = train_nn(X, y)

    # Build test inputs using time, beta, and gamma for every time point.
    X_test = np.column_stack([
        t,  # First input feature: time.
        np.full_like(t, beta),  # Second input feature: beta repeated for every time point.
        np.full_like(t, gamma)  # Third input feature: gamma repeated for every time point.
    ])

    # Convert the NumPy test array into a PyTorch floating-point tensor.
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Disable gradient calculation because the neural network is only being evaluated here.
    with torch.no_grad():
        # Run the test inputs through the trained NN and convert predictions back to a NumPy array.
        nn_pred = nn_model(X_test).detach().numpy()

    # -------------------------
    # 3. PINN
    # -------------------------
    print("Training PINN...")  # Inform the user that PINN training is beginning.
    # Train the PINN and receive both the trained model and its time-input tensor.
    pinn_model, t_pinn = train_pinn()

    # Disable gradients while generating final PINN predictions.
    with torch.no_grad():
        # Evaluate the trained PINN and convert its predictions to a NumPy array.
        pinn_pred = pinn_model(t_pinn).detach().numpy()

    # Convert the PINN time tensor to a NumPy array for plotting.
    t_plot = t_pinn.detach().numpy()

    # -------------------------
    # FINAL PLOT
    # -------------------------
    # Plot the deterministic ODE solution, standard NN prediction, and PINN prediction together.
    plot_final_comparison(
        t_plot,  # Time values used on the x-axis.
        ode[:len(t_plot)],  # ODE reference values, trimmed to match the plotting length.
        nn_pred[:len(t_plot)],  # Standard NN predictions, trimmed to the same length.
        pinn_pred  # PINN predictions.
    )


if __name__ == "__main__":
    # Run main() only when this file is executed directly, not when it is imported as a module.
    main()
