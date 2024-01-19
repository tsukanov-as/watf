import numpy as np

class Сlassifier:
    def __init__(self, L, F):
        self.W = np.zeros((L, F))
        self.B = np.zeros(L)

    def tune(self, L, F):
        delta = L - np.heaviside(self.W @ F + self.B, 0)
        self.W += np.einsum('i,j->ij', delta, F)
        self.B += delta

    def tune_all(self, L, F):
        L = np.eye(self.W.shape[0])[L]
        for i in range(len(F)):
            self.tune(L[i], F[i])

    def pred(self, F):
        return (self.W @ F + self.B).argmax()

    def test_all(self, L, F):
        total = 0
        for i in range(len(F)):
            if self.pred(F[i]) == L[i]:
                total += 1
        return total/len(F)
