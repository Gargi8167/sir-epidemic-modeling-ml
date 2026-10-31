# This file contains functions for visualizing SIR simulations and model predictions.
# Matplotlib is used to create the graphs.

import matplotlib.pyplot as plt  # Import Matplotlib's plotting interface.


def plot_comparison(t1, S1, I1, R1, t2, S2, I2, R2):
    # Plot stochastic SIR trajectories against deterministic ODE trajectories.

    plt.figure()  # Create a new figure for the comparison plot.

    # Stochastic (dashed)
    plt.plot(t1, S1, label="S (Stochastic)", linestyle="--")  # Plot stochastic susceptible population.
    plt.plot(t1, I1, label="I (Stochastic)", linestyle="--")  # Plot stochastic infected population.
    plt.plot(t1, R1, label="R (Stochastic)", linestyle="--")  # Plot stochastic recovered population.

    # Deterministic (solid)
    plt.plot(t2, S2, label="S (ODE)")  # Plot deterministic susceptible population.
    plt.plot(t2, I2, label="I (ODE)")  # Plot deterministic infected population.
    plt.plot(t2, R2, label="R (ODE)")  # Plot deterministic recovered population.

    plt.xlabel("Time")  # Label the horizontal axis as time.
    plt.ylabel("Population")  # Label the vertical axis as population.
    plt.title("Stochastic vs Deterministic SIR")  # Add a descriptive title to the graph.

    plt.legend()  # Display labels explaining each plotted curve.
    plt.grid()  # Add grid lines to make the graph easier to read.
    plt.show()  # Display the graph on screen.


def plot_prediction(t, true, pred):
    # Plot true SIR values against model-predicted SIR values.

    labels = ["S", "I", "R"]  # Store names for the three SIR compartments.

    plt.figure()  # Create a new figure.

    for i in range(3):
        # Repeat once for S, once for I, and once for R.
        plt.plot(t, true[:, i], label=f"True {labels[i]}")  # Plot the true values for the current compartment.
        plt.plot(t, pred[:, i], "--", label=f"Pred {labels[i]}")  # Plot the predicted values using a dashed line.

    plt.xlabel("Time")  # Label the x-axis.
    plt.ylabel("Population")  # Label the y-axis.
    plt.title("Model Predictions vs True Values")  # Add the graph title.

    plt.legend()  # Show the curve labels.
    plt.grid()  # Add grid lines.
    plt.show()  # Display the graph.


def plot_pinn(t, pred):
    # Plot the S, I, and R predictions produced by the PINN.

    S = pred[:, 0]  # Extract predicted susceptible values.
    I = pred[:, 1]  # Extract predicted infected values.
    R = pred[:, 2]  # Extract predicted recovered values.

    plt.figure()  # Create a new figure.
    plt.plot(t, S, label="S (PINN)")  # Plot the PINN susceptible trajectory.
    plt.plot(t, I, label="I (PINN)")  # Plot the PINN infected trajectory.
    plt.plot(t, R, label="R (PINN)")  # Plot the PINN recovered trajectory.

    plt.xlabel("Time")  # Label the x-axis.
    plt.ylabel("Population")  # Label the y-axis.
    plt.title("PINN Learned SIR Dynamics")  # Add a descriptive title.

    plt.legend()  # Show labels for the three curves.
    plt.grid()  # Add grid lines.
    plt.show()  # Display the graph.


def plot_final_comparison(t, ode, nn_pred, pinn_pred):
    # Create the final graph comparing the deterministic ODE, standard NN, and PINN.

    import matplotlib.pyplot as plt  # Import Matplotlib locally so this function can use plotting commands.

    labels = ["S", "I", "R"]  # Store the names of the three SIR compartments.

    plt.figure(figsize=(10, 6))  # Create a larger 10-by-6 inch figure for the final comparison.

    for i in range(3):
        # Repeat for susceptible, infected, and recovered populations.
        plt.plot(t, ode[:, i], label=f"ODE {labels[i]}")  # Plot the deterministic ODE reference curve.
        plt.plot(t, nn_pred[:, i], "--", label=f"NN {labels[i]}")  # Plot the standard neural-network prediction as dashed.
        plt.plot(t, pinn_pred[:, i], ":", label=f"PINN {labels[i]}")  # Plot the PINN prediction as dotted.

    plt.xlabel("Time")  # Label the horizontal axis.
    plt.ylabel("Population")  # Label the vertical axis.
    plt.title("Final Comparison: ODE vs NN vs PINN")  # Give the final comparison plot a clear title.

    plt.legend()  # Display the labels for all plotted curves.
    plt.grid()  # Add grid lines for readability.

    # Save the final comparison figure to the outputs folder.
    # NOTE: the folder must exist before saving, otherwise Matplotlib can raise a FileNotFoundError.
    plt.savefig("outputs/final_comparison.png")

    plt.show()  # Display the final graph on screen.
