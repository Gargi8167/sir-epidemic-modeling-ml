import numpy as np

def stochastic_sir(S0, I0, R0, beta, gamma, T=50, dt=0.1):
    """
    Stochastic SIR simulation using binomial transitions
    """

    S, I, R = S0, I0, R0
    N = S + I + R

    t_values = np.arange(0, T, dt)

    S_list = []
    I_list = []
    R_list = []

    for t in t_values:

        # Correct probabilities
        infection_prob = beta * I / N * dt
        recovery_prob = gamma * dt

        # Clamp probabilities
        infection_prob = min(max(infection_prob, 0), 1)
        recovery_prob = min(max(recovery_prob, 0), 1)

        # Random transitions
        new_infections = np.random.binomial(int(S), infection_prob)
        new_recoveries = np.random.binomial(int(I), recovery_prob)

        # Update populations
        S -= new_infections
        I += new_infections - new_recoveries
        R += new_recoveries

        # Store values
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)

    return np.array(t_values), np.array(S_list), np.array(I_list), np.array(R_list)