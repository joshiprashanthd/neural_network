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
    
    def __init__(self, epsilon: float = 0.001) -> None:
        self.epsilon = epsilon
    
    def loss(self, target: Tensor, predicted: Tensor) -> Tensor:
        return -np.sum(target * np.log(predicted + self.epsilon) + (1 - target) * np.log(1 - predicted + self.epsilon), axis=0) / len(target)

    def grad(self, target: Tensor, predicted: Tensor) -> Tensor:
        return -((target / (predicted + self.epsilon)) - (1 - target) / ((1 - predicted) + self.epsilon)) / len(target)
    
class CategoricalCrossEntropy(Loss):
    
    def __init__(self, epsilon: float = 0.001) -> None:
        self.epsilon = epsilon
    
    def loss(self, target: Tensor, predicted: Tensor) -> Tensor:
        return -np.sum(target * np.log(predicted + self.epsilon))
    
    def grad(self, target: Tensor, predicted: Tensor) -> Tensor:
        # print("PREDICTED: ",predicted)
        # print("TARGET: ", target)
        y = -(target / (predicted + self.epsilon)) / len(target)
        # print("RETURN VALUE: ", y)
        return y