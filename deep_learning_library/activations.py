
# Author: Prashant Joshi
# time: 12/29/2019 5:46 PM

"""
Activation perform non-linear operations on the output values, calculated by W * X + B
"""

from tensor import Tensor
import numpy as np


class Activation:
    """
    The base class for the activation function.
    """

    def func(self, input_tensor: Tensor) -> Tensor:
        raise NotImplementedError

    def grad(self, input_tensor: Tensor) -> Tensor:
        raise NotImplementedError


class Sigmoid(Activation):
    """
    A sigmoid function takes any float value and squashed it into a float value which ranges between 0 and 1
    """

    def func(self, input_tensor: Tensor) -> Tensor:
        """
        :param input_tensor: Tensor
        :return: A tensor having values transformed by Sigmoid function
        :rtype: Tensor

        Transforms values of {input_tensor} using Sigmoid Function
        """

        return 1 / (1 + np.exp(-input_tensor))

    def grad(self, input_tensor: Tensor) -> Tensor:
        """
        :param input_tensor: Tensor
        :return: A tensor whose value transform using gradient of Sigmoidal Function
        :rtype: Tensor

        Values are converted into grad values of Sigmoidal function
        """

        return self.func(input_tensor) * (1 - self.func(input_tensor))


class Tanh(Activation):
    """
    Tanh activation function represents simply the hyperbolic tangent function
    """

    def func(self, input_tensor: Tensor) -> Tensor:
        """

        :param input_tensor:
        :return:  tensor whose value transform using gradient of Tanh Function
        :rtype: Tensor

        Values are transformed using np.tanh function
        """

        return np.tanh(input_tensor)

    def grad(self, input_tensor: Tensor) -> Tensor:
        """
        :param input_tensor:
        :return: A tensor whose value transform using gradient of Tanh Function
        :rtype: Tensor

        Values are grad equivalent of the Tanh function
        """
        return 1 - (np.tanh(input_tensor) ** 2)


class ReLu(Activation):

    def func(self, input_tensor: Tensor) -> Tensor:
        copy_tensor = input_tensor.copy()
        copy_tensor[copy_tensor <= 0] = 0
        return copy_tensor

    def grad(self, input_tensor: Tensor) -> Tensor:
        copy_tensor = input_tensor.copy()
        copy_tensor[copy_tensor <= 0] = 0
        copy_tensor[copy_tensor > 0] = 1
        return copy_tensor


class Softmax(Activation):

    def func(self, input_tensor: Tensor) -> Tensor:
        return np.exp(input_tensor) / np.sum(np.exp(input_tensor), axis=0)

    def grad(self, input_tensor: Tensor) -> Tensor:
        s = self.func(input_tensor)
        s = s.reshape((len(s), s.shape[2], s.shape[1]))
        return np.array([(np.diagflat(s[i]) - np.dot(s[i], s[i].T)) @ input_tensor[i] for i in range(len(input_tensor))])


class Identity(Activation):

    def func(self, input_tensor: Tensor) -> Tensor:
        return input_tensor

    def grad(self, input_tensor: Tensor) -> Tensor:
        return np.ones(input_tensor.shape)
