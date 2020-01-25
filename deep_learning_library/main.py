# Author: Prashant Joshi
# Date time: 1-4-2020

import numpy as np
import layers
import activations
import models
import activations
import loss
import optimizers

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical

# dataset = load_iris()

# X = dataset.data.reshape((-1, 4, 1))
# y = dataset.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# X_train = X_train.reshape((-1, 4, 1))
# X_test = X_test.reshape((-1, 4, 1))
# y_train = to_categorical(y_train)
# y_train = y_train.reshape((-1, 3, 1))
# y_test = to_categorical(y_test)
# y_test = y_test.reshape((-1, 3, 1))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((-1, 2, 1))
y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]).reshape((-1, 3, 1))

if __name__ == "__main__":
    
    model = models.Sequential()
    model.add(layers.Dense(2, activation = activations.ReLu(), input_shape=X.shape[1:], weight_init_method="uniform"))
    # model.add(layers.Dense(4, activation=activations.ReLu()))
    model.add(layers.Dense(3, activation=activations.Softmax()))
    
    model.compile(loss=loss.CategoricalCrossEntropy(), optimizer=optimizers.SGD(lr=0.1, momentum=0))
    
    # print("Initial Prediction: ")
    # print(model.predict(X))
    
    model.fit(X, y, batch_size=4, epochs=100, shuffle=True)
    
    # print(y_test[:10])
    # print(X_train[:10])
    print(model.predict(X))