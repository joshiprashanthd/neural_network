from NeuralNetwork import NeuralNetwork
from activation import Sigmoid, Tanh

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

if __name__ == "__main__":
    nn = NeuralNetwork(input_units=2, hidden_units=2, output_units=1,
                       hidden_activation=Sigmoid, output_activation=Sigmoid, learning_rate=0.1)

    for i in range(4):
        print(f"y{i}: {nn.predict(X[i].reshape(1, 2))}")

    for i in range(500):
        for j in range(4):
            nn.train(X[j].reshape(1, 2), y[j])

    for i in range(4):
        print(f"y{i}: {nn.predict(X[i].reshape(1, 2))}")
