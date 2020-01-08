from typing import NamedTuple, Iterable
from tensor import Tensor
import numpy as np

Batch = NamedTuple('Batch', inputs=Tensor, targets=Tensor)

class BatchIterator:
    def __init__(self, inputs: Tensor, targets: Tensor, batch_size=32, shuffle=False):
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def batches(self) -> Iterable[Batch]:
        starts = np.arange(0, len(self.inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = self.inputs[start:end]
            batch_targets = self.targets[start:end]
            yield Batch(inputs=batch_inputs, targets=batch_targets)
