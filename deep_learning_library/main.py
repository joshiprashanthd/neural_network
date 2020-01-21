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
    
    model = models.Sequential()
    model.add(layers.Dense(2, activation = activations.Tanh(), input_shape=(2, 1)))
    # model.add(layers.Dropout(0.01))
    model.add(layers.Dense(1, activation=activations.Sigmoid()))
    
    model.compile(loss=loss.MSE(), optimizer=optimizers.SGD(lr=0.1, momentum=0.9))
    
    print("Initial Prediction: ")
    print(model.predict(X))
    
    
    model.fit(X, y, batch_size=4, epochs=1000)
    
    print(model.predict(X))