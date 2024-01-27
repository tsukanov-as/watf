import numpy as np

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def sigmoid_deriv(v):
    return v * (1 - v)

LR = 0.01

class Сlassifier:
    def __init__(self, Y, X):
        self.W = np.zeros((Y, X))
        self.B = np.zeros(Y)

    def tune(self, Y, X):
        P = sigmoid(self.W @ X + self.B)
        delta = (Y - P) * sigmoid_deriv(P) * LR
        self.W += np.einsum('i,j->ij', delta, X)
        self.B += delta

    def tune_all(self, Y, X):
        Y = np.eye(self.W.shape[0])[Y]
        for i in range(len(X)):
            self.tune(Y[i], X[i])

    def pred(self, X):
        return sigmoid(self.W @ X + self.B).argmax()

    def test_all(self, Y, X):
        total = 0
        for i in range(len(X)):
            if self.pred(X[i]) == Y[i]:
                total += 1
        return total/len(X)
