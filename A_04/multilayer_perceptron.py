"""Multilayer Perceptron (forward propagation only).

Architecture
    Input layer   : 3 neurons
    Hidden layer 1: 5 neurons   (sigmoid)
    Hidden layer 2: 5 neurons   (sigmoid)
    Hidden layer 3: 2 neurons   (sigmoid)
    Output layer 4: 2 neurons   (ReLU)

Weights and biases are fixed, hand-chosen values so results are reproducible.
"""

import numpy as np


class MultilayerPerceptron:
    """A 3-5-5-2-2 fully connected network with forward propagation."""

    def __init__(self):
        self.network = {}
        self.init_network()

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------
    @staticmethod
    def sigmoid(x):
        """Sigmoid: 1 / (1 + e^-x), applied element-wise."""
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def relu(x):
        """ReLU: max(0, x), applied element-wise."""
        return np.maximum(0, x)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def init_network(self):
        """Set the weights and biases (any values are allowed by the assignment).

        Shapes follow  (neurons in previous layer, neurons in this layer):
            W1: (3, 5)  b1: (5,)
            W2: (5, 5)  b2: (5,)
            W3: (5, 2)  b3: (2,)
            W4: (2, 2)  b4: (2,)
        """
        net = {}

        net["W1"] = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0],
            [-0.1, -0.2, -0.3, -0.4, -0.5],
        ])
        net["b1"] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        net["W2"] = np.array([
            [0.1, -0.2, 0.3, -0.4, 0.5],
            [0.2, 0.3, -0.4, 0.5, -0.6],
            [-0.3, 0.4, 0.5, -0.6, 0.7],
            [0.4, -0.5, 0.6, 0.7, -0.8],
            [-0.5, 0.6, -0.7, 0.8, 0.9],
        ])
        net["b2"] = np.array([0.1, -0.1, 0.1, -0.1, 0.1])

        net["W3"] = np.array([
            [0.3, -0.7],
            [0.5, 0.2],
            [-0.4, 0.6],
            [0.8, -0.1],
            [-0.2, 0.4],
        ])
        net["b3"] = np.array([0.1, 0.2])

        # Output layer: chosen so the second output is positive for some inputs
        # and clipped to 0 by ReLU for others.
        net["W4"] = np.array([
            [0.9, 2.0],
            [0.4, -2.0],
        ])
        net["b4"] = np.array([0.05, -0.09])

        self.network = net

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def forward(self, x, return_all=False):
        """Propagate input ``x`` (shape (3,) or (N, 3)) to the output layer.

        If ``return_all`` is True, also return the activations of every layer
        as a list [input, layer1, layer2, layer3, output].
        """
        net = self.network
        x = np.asarray(x, dtype=float)

        a0 = x
        a1 = self.sigmoid(a0 @ net["W1"] + net["b1"])   # hidden layer 1
        a2 = self.sigmoid(a1 @ net["W2"] + net["b2"])   # hidden layer 2
        a3 = self.sigmoid(a2 @ net["W3"] + net["b3"])   # hidden layer 3
        y = self.relu(a3 @ net["W4"] + net["b4"])       # output layer (ReLU)

        if return_all:
            return y, [a0, a1, a2, a3, y]
        return y
