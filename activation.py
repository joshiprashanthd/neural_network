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
    def vectGrad(X: Tensor) -> Tensor:
        return Sigmoid.func(X) * (1 - Sigmoid.func(X))


class Tanh(Activation):
    """
    Represents a Hyperbolic Tangent Activation Function
    """

    @staticmethod
    def func(X: Tensor) -> Tensor:
        return np.tanh(X)

    @staticmethod
    def vectGrad(X: Tensor) -> Tensor:
        return 1 - (np.tanh(X) ** 2)

class ReLu(Activation):
    """
    A ReLu (Rectified Linear Units) transform every element to the result of max(0, x)
    """
    
    @staticmethod
    def func(X: Tensor) -> Tensor:
        return np.max(0, X)
    
    @staticmethod
    def grad(X: Tensor) -> Tensor:
        copy_X = X.copy()
        copy_X[copy_X < 0] = 0
        return copy_X


class Identity(Activation):
    
    @staticmethod
    def func(X: Tensor) -> Tensor:
        return X

    @staticmethod
    def grad(X: Tensor) -> Tensor:
        return np.ones(X.shape, dtype=X.dtype)