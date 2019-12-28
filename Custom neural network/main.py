"""
@author: Prashant Joshi
@date-time: 12-19-2019  10:27 PM
"""

from NeuralNetwork import NeuralNetwork
from activation import Sigmoid, Tanh, ReLu, Softmax

import numpy as np

X = np.array([[0, 0, 1], [2, 3, 8], [2, 1, 3]])
y = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])



if __name__ == "__main__":
    nn = NeuralNetwork(input_units=3, hidden_units=4, output_units=3,
                       hidden_activation=Tanh, output_activation=Softmax, learning_rate=0.1)

    epochs = 1000
    print("Initial Prediction")
    for i in range(3):
        print(f"y{i}: {nn.predict(X[i].reshape(1, 3))}")

    for i in range(epochs):
        for j in range(3):
            nn.train(X[j].reshape(1, 3), y[j])

    print("\nAfter training for {} epochs".format(epochs))
    for i in range(3):
        print(f"y{i}: {nn.predict(X[i].reshape(1, 3))}")