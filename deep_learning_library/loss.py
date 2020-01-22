"""
A loss function is use to optimize the model, by tuning the hyperparameters. The Stochastic Gradient
Descent is the main approach, which is used in this library
"""
from tensor import Tensor
import numpy as np

class Loss:
    """
    A base class loss functions.
    """

    def loss(self, target: Tensor, predicted: Tensor) -> Tensor:
        raise NotImplementedError

    def grad(self, target: Tensor, predicted: Tensor) -> Tensor:
        raise NotImplementedError


class Normal(Loss):
    def loss(self, target: Tensor, predicted: Tensor) -> Tensor:
        return np.sum(target - predicted)

    def grad(self, target: Tensor, predicted: Tensor) -> Tensor:
        return -(target - predicted)


class MSE(Loss):
    def loss(self, target: Tensor, predicted: Tensor) -> Tensor:
        return np.sum((target - predicted) ** 2)

    def grad(self, target: Tensor, predicted: Tensor) -> Tensor:
        return -(2 / len(target)) * (target - predicted)

class BinaryCrossEntropy(Loss):
    def loss(self, target: Tensor, predicted: Tensor) -> Tensor:
        return -np.sum(target * np.log(predicted+0.001) + (1 - target) * np.log(1 - predicted+0.001), axis=0) / len(target)

    def grad(self, target: Tensor, predicted: Tensor) -> Tensor:
        return -((target / (predicted + 0.001)) - (1 - target) / ((1 - predicted) + 0.001)) / len(target)