# Learning SIR Epidemic Dynamics Using Machine Learning and PINNs

## Overview

This project explores epidemic modelling using the classical **Susceptible–Infected–Recovered (SIR) model**, conventional Machine Learning and Physics-Informed Neural Networks (PINNs).

The project implements and examines four approaches:

- **Deterministic SIR Model:** Uses differential equations to model epidemic progression.
- **Stochastic SIR Simulation:** Introduces randomness into infection and recovery events using binomial sampling.
- **Neural Network:** Uses a feed-forward neural network to explore data-driven learning of SIR dynamics.
- **Physics-Informed Neural Network (PINN):** Incorporates the SIR differential equations into the neural network's loss function.

The main objective is to explore and compare mathematical modelling, data-driven learning and physics-informed learning approaches.

**Project Status:** Exploratory implementation. The neural network and PINN outputs require further refinement and do not currently reproduce the expected SIR trajectories accurately.

---

## Project Motivation

Mathematical models are widely used to understand the spread of infectious diseases.

The classical SIR model provides a mathematical framework for describing how individuals move between susceptible, infected and recovered compartments.

This project explores the following question:

> Can machine-learning models learn or approximate epidemic dynamics, and how can incorporating mathematical constraints influence the learning process?

The project investigates this through numerical simulation, neural networks and physics-informed learning.

---

## The SIR Model

The SIR model divides a population into three compartments:

- **S — Susceptible:** Individuals who can potentially become infected.
- **I — Infected:** Individuals currently infected.
- **R — Recovered:** Individuals who have recovered.

The total population is:

**N = S + I + R**

The model is represented by three differential equations:

dS/dt = −βSI/N

dI/dt = βSI/N − γI

dR/dt = γI

Where:

- **β (Beta):** Transmission rate.
- **γ (Gamma):** Recovery rate.
- **N:** Total population.

These equations describe how the population in each compartment changes over time.

---

## Project Methodology

### 1. Deterministic SIR Model

The deterministic model solves the SIR differential equations numerically.

The implementation uses:

- Python
- NumPy
- SciPy's `odeint()`

The model produces susceptible, infected and recovered population trajectories over time.

The deterministic solution is used as a reference for examining the neural-network and PINN outputs.

**Implementation:** `deterministic.py`

### 2. Stochastic SIR Simulation

The stochastic implementation introduces randomness into epidemic progression.

At each simulation step:

- Susceptible individuals may become infected.
- Infected individuals may recover.
- Binomial random sampling determines the number of infection and recovery events.

Unlike the deterministic model, repeated stochastic simulations can produce different trajectories.

**Implementation:** `stochastic.py`

### 3. Neural Network

A conventional feed-forward neural network is implemented using PyTorch.

The network uses three input features:

**[Time, β, γ]**

It produces three outputs:

**[S, I, R]**

The architecture consists of:

- Input layer: 3 neurons
- Hidden layer 1: 128 neurons with Tanh activation
- Hidden layer 2: 128 neurons with Tanh activation
- Output layer: 3 neurons

The implementation uses:

- PyTorch
- Adam optimizer
- Mean Squared Error (MSE) loss
- Backpropagation

**Implementation:**
- `nn_model.py`
- `train_nn.py`

The objective is to explore whether a neural network can learn the relationship between epidemic parameters, time and population dynamics.

### 4. Physics-Informed Neural Network (PINN)

A Physics-Informed Neural Network incorporates known mathematical equations into the learning process.

Unlike a conventional neural network, which learns primarily from training data, a PINN also uses mathematical constraints to guide its predictions.

In this project, the PINN takes time as input and predicts:

**[S, I, R]**

Its architecture consists of:

- Input layer: 1 neuron
- Hidden layer 1: 64 neurons with Tanh activation
- Hidden layer 2: 64 neurons with Tanh activation
- Output layer: 3 neurons

**Implementation:**
- `pinn.py`
- `train_pinn.py`

### Automatic Differentiation

The PINN uses PyTorch automatic differentiation to calculate the time derivatives of its outputs:

- dS/dt
- dI/dt
- dR/dt

These derivatives are substituted into the SIR differential equations to calculate physics residuals.

For example:

For example, the susceptible equation requires:

dS/dt + βSI/N = 0

Similarly:

dI/dt − (βSI/N − γI) = 0

and:

dR/dt − γI = 0

The differences between the predicted derivatives and the expected SIR dynamics form the physics residuals.

The PINN minimizes these residuals during training.


**Current implementation note:** The PINN training currently focuses on physics-residual loss. Explicit initial-condition enforcement and a supervised data loss are not included in the current formulation.

---

## Experimental Parameters

The deterministic SIR experiment uses the following initial conditions and parameters:

| Parameter | Value |
|---|---:|
| Initial susceptible population | 990 |
| Initial infected population | 10 |
| Initial recovered population | 0 |
| Transmission rate (β) | 0.3 |
| Recovery rate (γ) | 0.1 |
| Simulation duration | 50 time units |
| Time points | 200 |

These values define the initial conditions for the deterministic SIR experiment.

---

## Results and Observations

The project includes a final comparison of the deterministic SIR model, neural network and PINN outputs.

### Deterministic SIR Output

The deterministic model produces the expected epidemic progression:

- Susceptible population decreases over time.
- Infected population initially increases, reaches a peak and subsequently decreases.
- Recovered population increases over time.

### Neural Network and PINN Outputs

In the current comparison, the neural network and PINN outputs remain close to zero relative to the deterministic trajectories.

They do not accurately reproduce the expected SIR population dynamics.

### Interpretation

The current results demonstrate the implementation and comparison of different modelling approaches, but do not establish successful neural-network-based epidemic prediction.

Further refinement of the training process and model formulation is required.

---

## Limitations

The current implementation has the following limitations:

1. The neural network and PINN outputs do not accurately reproduce the deterministic SIR trajectories.
2. Initial conditions are not explicitly enforced in the PINN loss function.
3. PINN training currently uses physics-residual loss without a supervised data loss.
4. The stochastic simulator is implemented separately and is not integrated into the final ODE–NN–PINN comparison.
5. The dataset-generation component and complete end-to-end execution workflow require further verification.
6. A quantitative evaluation of the final NN and PINN predictions has not been established.
7. The experiments use a simplified SIR model and do not represent the full complexity of real-world epidemics.

---

## Future Improvements

Possible extensions include:

1. **Initial-Condition Enforcement:** Incorporate initial S, I and R values into the PINN loss function.
2. **Combined Loss Function:** Explore combining supervised data loss with physics-residual loss.
3. **Dataset Generation:** Develop and validate a complete synthetic dataset-generation pipeline.
4. **Model Evaluation:** Compare predictions quantitatively using metrics such as Mean Squared Error.
5. **Stochastic Model Integration:** Include stochastic trajectories in the overall model comparison.
6. **Parameter Estimation:** Explore learning transmission and recovery parameters from epidemic observations.
7. **SEIR Extension:** Extend the framework by introducing an Exposed compartment.
8. **Real-World Data:** Investigate the application of the framework to publicly available epidemiological datasets.

---

## Technologies Used

**Programming Language**
- Python

**Machine Learning**
- PyTorch
- Feed-forward Neural Networks
- Physics-Informed Neural Networks
- Automatic Differentiation

**Scientific Computing**
- NumPy
- SciPy

**Visualization**
- Matplotlib

---

## Project Structure

The project contains the following main components:

```text
sir-epidemic-modeling-ml/
│
├── main.py
├── deterministic.py
├── stochastic.py
├── nn_model.py
├── train_nn.py
├── pinn.py
├── train_pinn.py
├── plotting.py
├── metrics.py
├── outputs/
│   └── final_comparison.png
│
└── README.md
```

*Note: The dataset-generation component and complete execution workflow require further verification. This structure describes the intended main project components.*

---

## Conclusion

This project is an exploratory study of epidemic modelling using mathematical equations, stochastic simulation, conventional neural networks and Physics-Informed Neural Networks.

The deterministic SIR model provides the mathematical reference, while the neural network and PINN implementations explore data-driven and physics-informed approaches to learning epidemic dynamics.

The current comparison highlights the limitations of the implemented learning models and provides directions for further improvement.

The project provided practical exposure to differential equations, stochastic simulation, neural networks, automatic differentiation and physics-informed machine learning.

---

## Author

**Gargi**

Exploring Machine Learning, Mathematical Modelling and Physics-Informed Neural Networks.
