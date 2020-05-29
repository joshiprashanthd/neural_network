# Author: Prashant Joshi
# Date time: 1-4-2020

import numpy as np
import deep_learning_library.layers as layers
import deep_learning_library.models as models
import deep_learning_library.activations as activations
import deep_learning_library.loss as loss
import deep_learning_library.optimizers as optimizers

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

dataset = load_iris()

X = dataset.data.reshape((-1, 4, 1))
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = X_train.reshape((-1, 4, 1))
X_test = X_test.reshape((-1, 4, 1))
y_train = to_categorical(y_train)
y_train = y_train.reshape((-1, 3, 1))
y_test = to_categorical(y_test)
y_test = y_test.reshape((-1, 3, 1))

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((-1, 2, 1))
# y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]).reshape((-1, 2, 1))

if __name__ == "__main__":
    
    model = models.Sequential()
    model.add(layers.Dense(16, activation=activations.ReLu(), input_shape=X_train.shape[1:], weight_init_method="relu"))
    model.add(layers.Dense(16, activation=activations.ReLu(), weight_init_method="relu"))
    model.add(layers.Dense(3, activation=activations.Sigmoid()))
    
    # model.compile(loss=loss.MSE(), optimizer=optimizers.SGD(lr=0.01, momentum=0.9)) #loss after 1000 epoch: 40.002
    # model.compile(loss=loss.MSE(), optimizer=optimizers.ADAGRAD()) # loss after 1000 epoch: 6.56
    # model.compile(loss=loss.MSE(),optimizer=optimizers.RMSPROP()) # loss after 1000 epoch: 6.003
    model.compile(loss=loss.MSE(), optimizer=optimizers.ADAM()) # loss after 1000 epoch: 4.204

    model.fit(X_train, y_train, batch_size=80, epochs=1000, shuffle=False, verbose=True)

    y_pred = model.predict(X_test)
    
    y_test_alt = np.array([np.argmax(y_test[i]) for i in range(len(y_test))])
    y_pred_alt = np.array([np.argmax(y_pred[i]) for i in range(len(y_pred))])
    
    print("y_test_alt: ", y_test_alt[:10])
    print("y_pred_alt: ", y_pred_alt[:10])
    
    print(confusion_matrix(y_test_alt, y_pred_alt))