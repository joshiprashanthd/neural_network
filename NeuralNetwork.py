import numpy as np
from activation import Activation, Identity
from loss import Loss, MSE, Normal

from tensor import Tensor


class NeuralNetwork:
    """
    Represents a Neural Network consist of input_layer -> hidden_layer -> output_layer
    """

    def __init__(self, input_units: int = 0, hidden_units: int = 0, output_units: int = 0, learning_rate: float = 0.01,
                 hidden_activation: Activation = Identity, output_activation: Activation = Identity,
                 loss: Loss = Normal) -> None:
        self._weights_IH = np.random.randn(hidden_units, input_units)
        self._weights_HO = np.random.randn(output_units, hidden_units)
        self._bias_IH = np.random.randn(hidden_units, 1)
        self._bias_HO = np.random.randn(output_units, 1)
        self._learning_rate = learning_rate
        self._output_activation = output_activation
        self._hidden_activation = hidden_activation
        self.loss = loss

    def train(self, X: Tensor = None, y: Tensor = None) -> None:
        output = self.predict(X)
        hidden_output = self._hidden_output(X)

        output_error = self.loss.loss(y.reshape(-1, 1), output)
        hidden_error = self._weights_HO.T @ output_error

        delta_output = self._output_activation.vectGrad(output)
        delta_hidden_output = self._hidden_activation.vectGrad(hidden_output)

        self._bias_IH += ((self._learning_rate * hidden_error) * delta_hidden_output)
        self._bias_HO += ((self._learning_rate * output_error) * delta_output)

        self._weights_IH += ((self._learning_rate * hidden_error) * delta_hidden_output) @ X
        self._weights_HO += ((self._learning_rate * output_error) * delta_output) @ hidden_output.T

    def _hidden_output(self, X: Tensor = None) -> Tensor:
        assert X.ndim == 2, Exception("X must be a row vector")
        X = X.reshape(-1, 1)
        predict_IH = self._hidden_activation.vectFunc(self._weights_IH @ X + self._bias_IH)
        return predict_IH

    def predict(self, X: Tensor = None) -> Tensor:
        assert X.ndim == 2, Exception("X must be a row vector")
        X = X.reshape(-1, 1)
        predict_IH = self._hidden_activation.vectFunc(self._weights_IH @ X + self._bias_IH)
        predict_HO = self._output_activation.vectFunc(self._weights_HO @ predict_IH + self._bias_HO)
        return predict_HO
