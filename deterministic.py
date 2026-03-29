import numpy as np
from scipy.integrate import odeint

def sir_ode(y, t, beta, gamma):
    """
    SIR differential equations
    """
    S, I, R = y
    N = S + I + R

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I

    return [dSdt, dIdt, dRdt]


def solve_sir(S0, I0, R0, beta, gamma, T=50):
    """
    Solve SIR ODE system
    """

    t = np.linspace(0, T, 200)
    initial_conditions = [S0, I0, R0]

    solution = odeint(sir_ode, initial_conditions, t, args=(beta, gamma))

    S = solution[:, 0]
    I = solution[:, 1]
    R = solution[:, 2]

    return t, S, I, R