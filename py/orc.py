import numpy as np

# OR classifier: https://github.com/tsukanov-as/neuron

class Сlassifier:
    def __init__(self, Y, X):
        self.W = np.zeros((Y, X))
        self.T = np.full(X, 0.000001)

    def tune(self, Y, X):
        P = self.pred(X)
        if P != Y:
            self.W[Y] += X
            self.W[P] = (self.W[P] - X/2).clip(0)
            self.T += X

    def tune_all(self, Y, X):
        for i in range(len(X)):
            self.tune(Y[i], X[i])
    
    def pred(self, X):
        return self.calc(X).argmax()
    
    def calc(self, X):
        return 1 - np.prod(1 - self.W / self.T * X, axis=1)

    def test_all(self, Y, X):
        total = 0
        for i in range(len(X)):
            if self.pred(X[i]) == Y[i]:
                total += 1
        return total/len(X)
