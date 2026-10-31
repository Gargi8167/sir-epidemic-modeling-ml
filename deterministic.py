# This file implements the deterministic SIR epidemic model.
# It uses the SIR differential equations and SciPy's ODE solver to calculate
# how the susceptible (S), infected (I), and recovered (R) populations change over time.

import numpy as np  # Import NumPy for numerical operations and for creating the time array.
from scipy.integrate import odeint  # Import SciPy's ODE solver for solving the SIR differential equations.


def sir_ode(y, t, beta, gamma):
    # Define the system of deterministic SIR differential equations.
    # y contains the current values of S, I, and R.
    # t is the current time.
    # beta is the infection/transmission rate.
    # gamma is the recovery rate.

    """
    # Purpose of this function: it returns the three SIR derivatives.
    SIR differential equations
    """

    S, I, R = y  # Unpack the current susceptible, infected, and recovered populations.
    N = S + I + R  # Calculate the total population, assumed to remain constant.

    dSdt = -beta * S * I / N  # Calculate how quickly susceptible people become infected.
    dIdt = beta * S * I / N - gamma * I  # Calculate infections gained minus recoveries.
    dRdt = gamma * I  # Calculate how quickly infected people recover.

    return [dSdt, dIdt, dRdt]  # Return the three derivatives to the ODE solver.


def solve_sir(S0, I0, R0, beta, gamma, T=50):
    # Define a helper function that solves the SIR equations over a chosen time period.
    # S0, I0, R0 are the initial population counts.
    # beta and gamma are the epidemic parameters.
    # T is the final simulation time and defaults to 50.

    """
    # What this function does?
    Solve SIR ODE system
    """

    t = np.linspace(0, T, 200)  # Create 200 evenly spaced time points from 0 to T.
    initial_conditions = [S0, I0, R0]  # Put the three initial population values into one list.

    solution = odeint(sir_ode, initial_conditions, t, args=(beta, gamma))  # Numerically solve the SIR ODE system.

    S = solution[:, 0]  # Extract the susceptible population from the solver output.
    I = solution[:, 1]  # Extract the infected population from the solver output.
    R = solution[:, 2]  # Extract the recovered population from the solver output.

    return t, S, I, R  # Return time and all three population trajectories.
