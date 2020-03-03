import numpy as np

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
            

class ADAGRAD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0, epsilon: float = 1e-6):
        self.momentum = momentum
        self.lr = lr
        self.prev_grads = []
        self.epsilon = epsilon
        
    def step(self, network=None) -> None:
        if len(self.prev_grads) == 0:
            self.prev_grads = [0] * (2 * len(network.layers))
            
        for i, (param, grad) in enumerate(network.params_grads()):
            self.prev_grads[i] = self.prev_grads[i] + grad**2
            param -= (self.lr / np.sqrt(self.prev_grads[i] + self.epsilon)) * grad
            
class RMSPROP(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0, epsilon: float = 1e-6, beta: float = 0.9):
        self.momentum = momentum
        self.lr = lr
        self.prev_grads = []
        self.beta = beta
        self.epsilon = epsilon
        
    def step(self, network=None) -> None:
        if len(self.prev_grads) == 0:
            self.prev_grads = [0] * (2 * len(network.layers))
            
        for i, (param, grad) in enumerate(network.params_grads()):
            self.prev_grads[i] = self.beta * self.prev_grads[i] + (1 - self.beta) * grad**2
            param -= (self.lr / np.sqrt(self.prev_grads[i] + self.epsilon)) * grad

class ADAM(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0, epsilon: float = 1e-6, beta1: float = 0.9, beta2: float = 0.999):
        self.momentum = momentum
        self.lr = lr
        self.prev_grads = []
        self.prev_velo = []
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.timestep = 0
    
    def step(self, network=None) -> None:
        if len(self.prev_grads) == 0:
            self.prev_grads = [0] * (2 * len(network.layers))
            
        if len(self.prev_velo) == 0:
            self.prev_velo = [0] * (2 * len(network.layers))
        
        self.timestep += 1
        
        for i, (param, grad) in enumerate(network.params_grads()):
            self.prev_grads[i] = self.beta1 * self.prev_grads[i] + (1 - self.beta1) * grad
            self.prev_velo[i] = self.beta2 * self.prev_velo[i] + (1 - self.beta2) * grad**2
            m_hat = self.prev_grads[i] / (1 - self.beta1 ** self.timestep)
            v_hat = self.prev_velo[i] / (1 - self.beta2 ** self.timestep)
            param -= (self.lr / np.sqrt(v_hat + self.epsilon)) * m_hat