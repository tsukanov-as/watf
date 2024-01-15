import numpy as np

class Сlassifier:
    def __init__(self, L, F):
        self.W = np.zeros((L, F))
        self.T = np.ones(F)

    def feed(self, L, F):
        self.W[L] += F
        self.T += F

    def tune(self, L, F):
        if self.pred(F) != L:
            self.feed(L, F)
            return True
        return False

    def tune_all(self, L, F):
        total = 0
        for i in range(len(F)):
            if self.tune(L[i], F[i]):
                total += 1
        return total

    def watf(self, F):
        w = self.W / self.T
        return w @ F

    def pred(self, F):
        return self.watf(F).argmax()

    def test_all(self, L, F):
        total = 0
        for i in range(len(F)):
            if self.pred(F[i]) == L[i]:
                total += 1
        return total/len(F)
