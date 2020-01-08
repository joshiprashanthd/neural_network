class Optimizer:
    def step(self, network=None) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, network=None) -> None:
        for param, grad in network.params_grads():
            param -= self.lr * grad
