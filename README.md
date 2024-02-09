# Weights @ Features

[Multiclass perceptron](https://en.wikipedia.org/wiki/Perceptron#Multiclass_perceptron)

```python
import numpy as np

class Сlassifier:
    def __init__(self, Y, X):
        self.W = np.zeros((Y, X))

    def tune(self, Y, X):
        P = self.pred(X)
        if P != Y:
            self.W[Y] += X
            self.W[P] -= X/2

    def pred(self, X):
        return (self.W @ X).argmax()
```
