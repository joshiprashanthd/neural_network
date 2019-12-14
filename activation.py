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
    def func(X: float) -> float:
        raise NotImplementedError

    @staticmethod
    def vectFunc(X: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def grad(X: float) -> float:
        raise NotImplementedError

    @staticmethod
    def vectGrad(X: Tensor) -> Tensor:
        raise NotImplementedError


class Sigmoid(Activation):
    """
    A sigmoid function just outputs values in range [0, 1]
    """

    @staticmethod
    def func(X: float) -> float:
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def vectFunc(X: Tensor) -> Tensor:
        vectorize_func = np.vectorize(Sigmoid.func)
        return vectorize_func(X)

    @staticmethod
    def grad(X: float) -> float:
        return Sigmoid.func(X) * (1 - Sigmoid.func(X))

    @staticmethod
    def vectGrad(X: Tensor) -> Tensor:
        return np.vectorize(Sigmoid.grad)(X)


class Tanh(Activation):
    """
    Represents a Hyperbolic Tangent Activation Function
    """


    @staticmethod
    def func(X: float) -> float:
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

    @staticmethod
    def vectFunc(X: Tensor) -> Tensor:
        vectorize_func = np.vectorize(Tanh.func)
        return vectorize_func(X)

    @staticmethod
    def grad(X: float) -> float:
        return 1 - (Tanh.func(X)) ** 2

    @staticmethod
    def vectGrad(X: Tensor) -> Tensor:
        return np.vectorize(Tanh.grad)(X)


class Identity(Activation):
    @staticmethod
    def func(X: float) -> float:
        return X

    @staticmethod
    def vectFunc(X: Tensor) -> Tensor:
        return X

    @staticmethod
    def grad(X: float) -> float:
        return 1

    @staticmethod
    def vectGrad(X: Tensor) -> Tensor:
        return np.vectorize(Identity.grad)(X)