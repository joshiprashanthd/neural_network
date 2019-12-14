"""
A loss function determines how far our predictions are from target outputs
"""
from tensor import Tensor
import numpy as np


class Loss:

    @staticmethod
    def loss(target: Tensor, predicted: Tensor) -> Tensor:
        """
        @param target:
        @param predicted:
        @return: Distance from target to predicted
        """
        raise NotImplementedError

    @staticmethod
    def grad(target: Tensor, predicted: Tensor) -> Tensor:
        """
        @param target
        @param predicted
        """
        raise NotImplementedError


class MSE(Loss):
    @staticmethod
    def loss(target: Tensor, predicted: Tensor) -> Tensor:
        print("LOSS: \n", (target - predicted) ** 2)
        return (target - predicted) ** 2

    @staticmethod
    def grad(target: Tensor, predicted: Tensor) -> Tensor:
        return 2 * (target - predicted)


class Normal(Loss):
    @staticmethod
    def loss(target: Tensor, predicted: Tensor) -> Tensor:
        return target - predicted

    @staticmethod
    def grad(target: Tensor, predicted: Tensor) -> Tensor:
        return np.ones(target.shape)
