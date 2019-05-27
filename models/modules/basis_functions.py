import numpy as np


class Polynomial:
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, xs):
        return xs ** self.dim


class Sin:
    def __init__(self, coefficient: float):
        self.coefficient = coefficient

    def __call__(self, xs):
        return np.sin(xs * self.coefficient)


class Cos:
    def __init__(self, coefficient: float):
        self.coefficient = coefficient

    def __call__(self, xs):
        return np.cos(xs * self.coefficient)
