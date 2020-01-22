bce = loss.BinaryCrossEntropy()

print(bce.grad(X, y))