import numpy as np

class Сlassifier:
    def __init__(self, L, F):
        self.W = np.zeros((L, F))

    def feed(self, L, F):
        self.W[L] += F

    def tune(self, L, F):
        if self.pred(F) != L:
            self.feed(L, F)

    def watf(self, F):
        return self.W @ F
    
    def pred(self, F):
        return self.watf(F).argmax()
