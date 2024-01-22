import numpy as np

class Сlassifier:
    def __init__(self, Y, X):
        self.W = np.zeros((Y, X))

    def pred(self, X):
        return (self.W @ X).argmax()

    def tune(self, Y, X):
        P = self.pred(X)
        if P != Y:
            self.W[Y] += X
            self.W[P] -= X/2
            return True
        return False

    def tune_all(self, Y, X):
        total = 0
        for i in range(len(X)):
            if self.tune(Y[i], X[i]):
                total += 1
        return total

    def test_all(self, Y, X):
        total = 0
        for i in range(len(X)):
            if self.pred(X[i]) == Y[i]:
                total += 1
        return total/len(X)
