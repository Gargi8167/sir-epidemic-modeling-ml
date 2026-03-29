import matplotlib.pyplot as plt


def plot_comparison(t1, S1, I1, R1, t2, S2, I2, R2):
    plt.figure()

    # Stochastic (dashed)
    plt.plot(t1, S1, label="S (Stochastic)", linestyle="--")
    plt.plot(t1, I1, label="I (Stochastic)", linestyle="--")
    plt.plot(t1, R1, label="R (Stochastic)", linestyle="--")

    # Deterministic (solid)
    plt.plot(t2, S2, label="S (ODE)")
    plt.plot(t2, I2, label="I (ODE)")
    plt.plot(t2, R2, label="R (ODE)")

    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Stochastic vs Deterministic SIR")

    plt.legend()
    plt.grid()
    plt.show()


def plot_prediction(t, true, pred):
    labels = ["S", "I", "R"]

    plt.figure()

    for i in range(3):
        plt.plot(t, true[:, i], label=f"True {labels[i]}")
        plt.plot(t, pred[:, i], "--", label=f"Pred {labels[i]}")

    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Model Predictions vs True Values")

    plt.legend()
    plt.grid()
    plt.show()


def plot_pinn(t, pred):
    S = pred[:, 0]
    I = pred[:, 1]
    R = pred[:, 2]

    plt.figure()
    plt.plot(t, S, label="S (PINN)")
    plt.plot(t, I, label="I (PINN)")
    plt.plot(t, R, label="R (PINN)")

    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("PINN Learned SIR Dynamics")

    plt.legend()
    plt.grid()
    plt.show()


def plot_final_comparison(t, ode, nn_pred, pinn_pred):
    import matplotlib.pyplot as plt

    labels = ["S", "I", "R"]

    plt.figure(figsize=(10, 6))

    for i in range(3):
        plt.plot(t, ode[:, i], label=f"ODE {labels[i]}")
        plt.plot(t, nn_pred[:, i], "--", label=f"NN {labels[i]}")
        plt.plot(t, pinn_pred[:, i], ":", label=f"PINN {labels[i]}")

    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Final Comparison: ODE vs NN vs PINN")

    plt.legend()
    plt.grid()

    plt.savefig("outputs/final_comparison.png")  # 🔥 ADD THIS
    plt.show()