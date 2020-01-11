from tensor import Tensor
from typing import Dict, Iterable
from activations import Activation
import numpy as np
import activations


class Layer:
    def __init__(self):
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.output_size = None
        self.input_shape = None
        self.inputs = None
        self.activation = None

    def set_input_shape(self, input_size: int) -> None:
        raise NotImplementedError

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, error: Tensor) -> Tensor:
        raise NotImplementedError

    def raw_output(self) -> Tensor:
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, output_size: int = None, activation: Activation = None, input_shape: tuple = None) -> None:
        super().__init__()
        assert (output_size is not None), Exception("Output size is not defined")
        assert (activation is not None), Exception("Activation is not defined")

        self.params: Dict[str, Tensor] = {}
        self.grad: Dict[str, Tensor] = {}
        self.output_size = output_size
        self.activation = activation
        self.input_shape = input_shape
        self.inputs = None
        self.outputs = None
        self.raw_outputs = None

        if self.input_shape is not None:
            self.init_param()

    def init_param(self):
        self.params = {
            'w': np.random.uniform(size=(self.output_size, self.input_shape[0])),
            'b': np.random.uniform(size=(self.output_size, 1))
        }

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        self.outputs = np.array([self.activation.func(self.params['w'] @ self.inputs[i] + self.params['b']) for i in range(len(self.inputs))])
        return self.outputs

    def backward(self, error: Tensor) -> Tensor:
        assert (self.inputs is not None), Exception("Inputs are not initialized")

        self.raw_output()

        self.grads['b'] = np.sum(error * self.activation.grad(self.raw_outputs), axis=0)
        self.grads['w'] = np.sum((error * self.activation.grad(self.raw_outputs)) @ self.inputs.reshape(len(self.inputs), self.inputs.shape[2], self.inputs.shape[1]), axis=0)
        return self.params['w'].T @ error

    def set_input_shape(self, input_size: int) -> None:
        self.input_shape = (input_size, 1)
        self.init_param()

    def raw_output(self) -> Tensor:
        assert (self.inputs is not None), Exception("Inputs are not initialized")

        self.raw_outputs = self.params['w'] @ self.inputs + self.params['b']
        return self.raw_outputs

class Dropout(Layer):

    def __init__(self, fraction: float, input_shape: tuple = None):
        super().__init__()
        assert (0 <= fraction < 1), Exception("Fraction should be in range [0, 1)")

        self.fraction = fraction
        self.input_shape = input_shape
        self.output_size = None

    def set_input_shape(self, input_size: int) -> None:
        self.input_shape = (input_size, 1)
        self.output_size = input_size

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        size = self.input_shape[0]  # counting number of columns
        indexes = [np.random.choice(np.arange(size), p=activations.Softmax().func(np.random.rand(size))) for _ in
                   range(np.math.floor(self.fraction * size))]
        self.inputs[:, indexes] = 0
        return self.inputs

    def backward(self, error: Tensor) -> Tensor:
        return error

    def raw_output(self) -> Tensor:
        return self.inputs