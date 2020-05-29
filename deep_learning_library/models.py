from deep_learning_library.tensor import Tensor
import numpy as np
from deep_learning_library.layers import Dense, Layer
from deep_learning_library.batch_iterator import BatchIterator
from deep_learning_library.loss import Loss, MSE, Normal
from typing import Iterable, List, Tuple
from deep_learning_library.optimizers import Optimizer


class Sequential:
    def __init__(self, layers: List[Layer] = None) -> None:
        if layers is None:
            layers = []
        self.layers = layers
        self.is_compiled = False
        self.optimizer = None
        self.loss = None

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def fit(self, inputs: Tensor, targets: Tensor, epochs=5, batch_size=32, shuffle=False, verbose=True) -> None:
        assert (inputs.ndim >= 3), Exception("Batch dimension must be provided for inputs")
        assert (targets.ndim >= 3), Exception("Batch dimension must be provided for targets")
        assert (batch_size <= len(inputs)), Exception(
            "Batch must be less than or equal to the batches of inputs tensor")

        batch_iterator = BatchIterator(inputs, targets, shuffle=shuffle, batch_size=batch_size)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in batch_iterator.batches():
                predicted = self.predict(batch.inputs)
                epoch_loss += self.loss.loss(batch.targets, predicted)
                error = self.loss.grad(batch.targets, predicted)
                self.backprop(error)
                self.optimizer.step(self)
            if verbose: 
                print("EPOCH {} LOSS: {}".format(epoch, epoch_loss))

    def predict(self, inputs: Tensor) -> Tensor:
        assert self.is_compiled, Exception("Models must be compiled before prediction could be done")
        assert (inputs.ndim >= 3), Exception("Batch dimension must be provided for inputs")

        output = self.layers[0].forward(inputs)
        for layer in self.layers[1:]:
            output = layer.forward(output)
        return output

    def compile(self, loss: Loss = None, optimizer: Optimizer = None) -> None:
        assert (loss is not None), Exception("Loss function must be specified")
        assert (optimizer is not None), Exception("Optimizer must be specified")
        assert (len(self.layers) != 0), Exception("Layers must be specified before compiling the model")

        self.optimizer = optimizer
        self.loss = loss

        # Getting output size of the first layer. This input size is used to determine input shape of subsequent layers
        input_size = self.layers[0].output_size
        self.layers[0].id = 1
        i = 2

        # Setting input shape of all but first layers
        for layer in self.layers[1:]:
            layer.set_input_shape(input_size)
            input_size = layer.output_size
            layer.id = i
            i += 1

        self.is_compiled = True
        
    def evaluate(self, y_true: Tensor, y_pred: Tensor) -> Tuple[float, float]:
        assert(y_true.shape == y_pred.shape), Exception("Prediction and True valued tensor must have the same dimension")

    def backprop(self, error: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            error = layer.backward(error)
        return error

    def params_grads(self) -> Iterable[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for key in layer.params.keys():
                grad = layer.grads[key]
                param = layer.params[key]
                yield param, grad
