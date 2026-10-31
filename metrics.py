# This file contains evaluation metrics used to measure model prediction error.
# At present, it implements Mean Squared Error (MSE).

import numpy as np  # Import NumPy so that array-based mathematical calculations can be performed.


def mse(y_true, y_pred):
    # Define a function to calculate Mean Squared Error between actual and predicted values.

    """
    # The metric implemented by this function.
    Mean Squared Error
    """

    return np.mean((y_true - y_pred) ** 2)  # Compute squared errors, average them, and return the resulting MSE.
