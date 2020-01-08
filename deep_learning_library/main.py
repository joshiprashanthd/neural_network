# Author: Prashant Joshi
# Date time: 1-4-2020

import numpy as np
import layers
import activations
import models
import activations
import loss
import optimizers

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2, 1))
y = np.array([0, 1, 1, 0]).reshape((4, 1, 1))

if __name__ == "__main__":

    network = models.Sequential()
    network.add(layers.Dense(3, activation=activations.ReLu(), input_shape=(2, 1)))
    network.add(layers.Dense(1, activation=activations.Sigmoid()))

    network.compile(loss=loss.MSE(), optimizer=optimizers.SGD(lr=0.1))

    network.fit(X, y, epochs=1000, batch_size=4)

    y_pred = network.predict(X)

    print(y_pred)
