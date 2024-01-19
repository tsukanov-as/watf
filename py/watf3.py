import numpy as np

class Сlassifier:
    def __init__(self, L, F):
        self.W = np.zeros((L, F))

    def feed(self, L, F):
        self.W[L] += F

    def tune(self, L, F):
        P = self.pred(F)
        if P != L:
            self.feed(L, F)
            self.feed(P, -F*0.5)
            return True
        return False

    def tune_all(self, L, F):
        total = 0
        for i in range(len(F)):
            if self.tune(L[i], F[i]):
                total += 1
        return total

    def watf(self, F):
        return self.W @ F

    def pred(self, F):
        return self.watf(F).argmax()

    def test_all(self, L, F):
        total = 0
        for i in range(len(F)):
            if self.pred(F[i]) == L[i]:
                total += 1
        return total/len(F)
