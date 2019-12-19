"""
@author: Prashant Joshi
@date-time: 12-19-2019  10:27 PM
"""

from NeuralNetwork import NeuralNetwork
from activation import Sigmoid, Tanh, ReLu

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])



if __name__ == "__main__":
    nn = NeuralNetwork(input_units=2, hidden_units=2, output_units=1,
                       hidden_activation=Tanh, output_activation=Sigmoid, learning_rate=0.1)

    epochs = 10000
    print("Initial Prediction")
    for i in range(4):
        print(f"y{i}: {nn.predict(X[i].reshape(1, 2))}")

    for i in range(epochs):
        for j in range(4):
            nn.train(X[j].reshape(1, 2), y[j])

    print("\nAfter training for {} epochs".format(epochs))
    for i in range(4):
        print(f"y{i}: {nn.predict(X[i].reshape(1, 2))}")