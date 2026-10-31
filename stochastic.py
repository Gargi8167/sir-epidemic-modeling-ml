# This file implements a stochastic version of the SIR epidemic model.
# Instead of directly solving differential equations, it uses random binomial transitions
# to simulate infections and recoveries.

import numpy as np  # Import NumPy for random sampling, arrays, and numerical operations.


def stochastic_sir(S0, I0, R0, beta, gamma, T=50, dt=0.1):
    # Define the stochastic SIR simulation function.
    # S0, I0, R0 are the initial population counts.
    # beta is the infection/transmission rate.
    # gamma is the recovery rate.
    # T is the total simulation time.
    # dt is the size of each simulation time step.

    """
    # The simulation approach used by this function.
    Stochastic SIR simulation using binomial transitions
    """

    S, I, R = S0, I0, R0  # Initialize the current susceptible, infected, and recovered populations.
    N = S + I + R  # Calculate the total population.

    t_values = np.arange(0, T, dt)  # Create equally spaced simulation times from 0 up to T.

    S_list = []  # Create a list to store susceptible values over time.
    I_list = []  # Create a list to store infected values over time.
    R_list = []  # Create a list to store recovered values over time.

    for t in t_values:
        # Repeat the stochastic update for every simulation time point.

        # Correct probabilities
        infection_prob = beta * I / N * dt  # Estimate the probability that a susceptible person becomes infected during dt.
        recovery_prob = gamma * dt  # Estimate the probability that an infected person recovers during dt.

        # Clamp probabilities
        # Limit each probability to the mathematically valid interval [0, 1].
        infection_prob = min(max(infection_prob, 0), 1)  # Prevent the infection probability from going below 0 or above 1.
        recovery_prob = min(max(recovery_prob, 0), 1)  # Prevent the recovery probability from going below 0 or above 1.

        # Random transitions
        # Sample how many people become infected and recover during this time step.
        new_infections = np.random.binomial(int(S), infection_prob)  # Randomly sample new infections among susceptible people.
        new_recoveries = np.random.binomial(int(I), recovery_prob)  # Randomly sample recoveries among infected people.

        # Update populations
        S -= new_infections  # Remove newly infected people from the susceptible population.
        I += new_infections - new_recoveries  # Add new infections and remove people who recovered.
        R += new_recoveries  # Add newly recovered people to the recovered population.

        # Store values
        S_list.append(S)  # Save the current susceptible population.
        I_list.append(I)  # Save the current infected population.
        R_list.append(R)  # Save the current recovered population.

    # Convert the time and population lists into NumPy arrays and return them.
    return np.array(t_values), np.array(S_list), np.array(I_list), np.array(R_list)

