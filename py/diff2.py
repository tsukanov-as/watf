import numpy as np

class Сlassifier:
    def __init__(self, L, F):
        self.W = np.zeros((L, F))
        self.T = np.ones((L, 1))

    def feed(self, L, F):
        self.W[L] += F
        self.T[L] += 1

    def tune(self, L, F):
        P = self.pred(F)
        if P != L:
            self.feed(L, F)
            self.W[P] = (self.W[P] - F/2).clip(0)
            return True
        return False

    def tune_all(self, L, F):
        total = 0
        for i in range(len(F)):
            if self.tune(L[i], F[i]):
                total += 1
        return total

    def diff(self, F):
        w = self.W / self.T
        return ((w - F)**2).sum(axis=1)

    def pred(self, F):
        return self.diff(F).argmin()

    def test_all(self, L, F):
        total = 0
        for i in range(len(F)):
            if self.pred(F[i]) == L[i]:
                total += 1
        return total/len(F)
