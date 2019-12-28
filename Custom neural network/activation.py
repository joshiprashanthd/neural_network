"""
A activation represents a mapping a original value into a different form
"""

import numpy as np
from tensor import Tensor


class Activation:
    """
    An asbtract class for loss functions
    """

    @staticmethod
    def func(X: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def grad(X: Tensor) -> Tensor:
        raise NotImplementedError


class Sigmoid(Activation):
    """
    A sigmoid function just outputs values in range [0, 1]
    """

    @staticmethod
    def func(X: Tensor) -> Tensor:
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def grad(X: Tensor) -> Tensor:
        return Sigmoid.func(X) * (1 - Sigmoid.func(X))


class Tanh(Activation):
    """
    Represents a Hyperbolic Tangent Activation Function
    """

    @staticmethod
    def func(X: Tensor) -> Tensor:
        return np.tanh(X)

    @staticmethod
    def grad(X: Tensor) -> Tensor:
        return 1 - (np.tanh(X) ** 2)


class ReLu(Activation):
    """
    A ReLu (Rectified Linear Units) transform every element to the result of max(0, x)
    """

    @staticmethod
    def _max(x: float) -> float:
        if x < 0:
            return 0
        else:
            return x

    @staticmethod
    def func(X: Tensor) -> Tensor:
        return np.vectorize(ReLu._max)(X)

    @staticmethod
    def grad(X: Tensor) -> Tensor:
        copy_X = X.copy()
        copy_X[copy_X < 0] = 0
        copy_X[copy_X != 0] = 1
        return copy_X
    
class Softmax(Activation):

    @staticmethod
    def func(X: Tensor) -> Tensor:
        return np.exp(X) / np.sum(np.exp(X))

    @staticmethod
    def grad(X: Tensor) -> Tensor:
        s = Softmax.func(X)
        s = s.reshape(-1, 1)
        return (np.diagflat(s) - np.dot(s, s.T)) @ X


class Identity(Activation):

    @staticmethod
    def func(X: Tensor) -> Tensor:
        return X

    @staticmethod
    def grad(X: Tensor) -> Tensor:
        return np.ones(X.shape, dtype=X.dtype)
