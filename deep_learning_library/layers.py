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
        self.logits = None

    def set_input_shape(self, input_size: int) -> None:
        raise NotImplementedError

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, error: Tensor) -> Tensor:
        raise NotImplementedError

    def init_logits(self) -> Tensor:
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, output_size: int = None, activation: Activation = None, input_shape: tuple = None, weight_init_method: str = "uniform") -> None:
        super().__init__()
        assert (output_size is not None), Exception("Output size is not defined")
        assert (activation is not None), Exception("Activation is not defined")

        self.params: Dict[str, Tensor] = {}
        self.grad: Dict[str, Tensor] = {}
        self.output_size = output_size
        self.activation = activation
        self.input_shape = input_shape
        self.weight_init_method = weight_init_method
        self.inputs = None
        self.outputs = None
        self.logits = None

        if self.input_shape is not None:
            self.init_param()

    def init_param(self):
        if self.weight_init_method is "relu_optimized":
            self.params = {
                'w': np.random.randn(self.output_size, self.input_shape[0]) * np.sqrt(1/self.input_shape[0]),
                'b': np.random.randn(self.output_size, 1) * np.sqrt(1/self.input_shape[0])
            }
        elif self.weight_init_method == "uniform":
            self.params = {
                'w': np.random.uniform(size=(self.output_size, self.input_shape[0])) * np.sqrt(1/self.input_shape[0]),
                'b': np.random.uniform(size=(self.output_size, 1)) * np.sqrt(1/self.input_shape[0])
            }
        elif self.weight_init_method == "normal":
            self.params = {
                'w': np.random.normal(size=(self.output_size, self.input_shape[0])) * np.sqrt(1/self.input_shape[0]),
                'b': np.random.normal(size=(self.output_size, 1)) * np.sqrt(1/self.input_shape[0])
            }

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        self.outputs = np.array([self.activation.func(self.params['w'] @ self.inputs[i] + self.params['b']) for i in range(len(self.inputs))])
        return self.outputs

    def backward(self, error: Tensor) -> Tensor:
        assert (self.inputs is not None), Exception("Inputs are not initialized")

        self.init_logits()

        self.grads['b'] = np.sum(error * self.activation.grad(self.logits), axis=0)
        self.grads['w'] = np.sum((error * self.activation.grad(self.logits)) @ self.inputs.reshape(len(self.inputs), self.inputs.shape[2], self.inputs.shape[1]), axis=0)
        return self.params['w'].T @ error

    def set_input_shape(self, input_size: int) -> None:
        self.input_shape = (input_size, 1)
        self.init_param()

    def init_logits(self) -> Tensor:
        assert (self.inputs is not None), Exception("Inputs are not initialized")

        self.logits = self.params['w'] @ self.inputs + self.params['b']
        return self.logits

class Dropout(Layer):

    def __init__(self, fraction: float, input_shape: tuple = None):
        super().__init__()
        assert (0 <= fraction < 1), Exception("Fraction should be in range [0, 1)")

        self.fraction = fraction
        self.input_shape = input_shape
        self.output_size = None
        self.logits = None

    def set_input_shape(self, input_size: int) -> None:
        self.input_shape = (input_size, 1)
        self.output_size = input_size

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        
        self.init_logits()
        
        keep_prob = 1 - self.fraction
        
        d = np.random.rand(self.inputs.shape[1], self.inputs.shape[2]) < keep_prob
        self.inputs *= d
        self.inputs /= keep_prob
        
        return self.inputs

    def backward(self, error: Tensor) -> Tensor:
        return error
    
    def init_logits(self) -> Tensor:
        self.logits = self.inputs
        return self.inputs