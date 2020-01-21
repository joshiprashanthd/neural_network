class Optimizer:
    def step(self, network=None) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0):
        self.momentum = momentum
        self.lr = lr
        self.prev_grads = []

    def step(self, network=None) -> None:
        if len(self.prev_grads) == 0:
            self.prev_grads = [0] * (2 * len(network.layers))
        
        for i, (param, grad) in enumerate(network.params_grads()):
            self.prev_grads[i] = (self.lr * grad) + (self.momentum * self.prev_grads[i])
            param -= self.prev_grads[i]
            