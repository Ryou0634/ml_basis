from typing import Callable
import numpy as np


class DataGenerator:
    """
    Define function, and generate data with uniform distribution.
    """

    def __init__(self,
                 function: Callable,
                 input_dim: int = 1,
                 feature_size: int = 1):
        self.function = function
        self.input_dim = input_dim
        self.feature_size = feature_size

    def __call__(self,
                 sample_size: int,
                 low: int = 0,
                 high: int = 1):
        X = []
        ys = []
        for _ in range(sample_size):
            x = np.random.uniform(low, high, size=self.input_dim)
            y = float(self.function(x))
            X.append(x)
            ys.append(y)

        return np.stack(X), np.stack(ys)
