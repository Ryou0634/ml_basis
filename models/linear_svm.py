import random
import numpy as np


class LinearSVM:
    """
    Optimize Hinge loss by stochastic gradient.

    Inputs
    ----------
    xs : np.array (batch_size, feature_size)
    ys : np.array (batch_size)

    Returns
    -------

    """

    def __init__(self,
                 l2_regularization: float = 0.):
        self.l2_regularization = l2_regularization
        self.parameter = None

    def __call__(self, xs: np.array) -> np.array:
        """
        Perform prediction.
        The output is y = w * phi(x)
        """
        if self.parameter is None:
            assert Exception("The parameter is None. You have to perform `fit` before regression.")

        # concatenate 1 as bias term
        xs = np.pad(xs, [(0, 0), (0, 1)], mode='constant', constant_values=1)

        return xs.dot(self.parameter[:, np.newaxis])

    def predict(self, xs: np.array) -> np.array:
        """
        Perform classification.

        Parameters
        ----------
        xs : np.array (batch_size, feature_size)

        Returns
        -------
        labels : np.array (batch_size, )
            Binary labels (True or False).
        """
        score = self(xs)
        return score > 0

    def _init_parameter(self, feature_size: int):
        self.parameter = np.random.random(size=feature_size)

    def _calc_gradient(self, xs: np.array, ys: np.array):
        batch_size, feature_size = xs.shape
        hinge = (1 - xs.dot(self.parameter[:, np.newaxis]) * ys[:, np.newaxis]).flatten()
        grad = - xs * ys[:, np.newaxis] + 2 * self.l2_regularization * self.parameter
        grad = grad[hinge >= 0].sum(axis=0) / batch_size

        if grad.shape == (0,):
            grad = np.zeros(feature_size)

        if self.l2_regularization:
            grad += 2 * self.l2_regularization * self.parameter

        return grad

    def fit(self,
            xs: np.array,
            ys: np.ndarray,
            stepsize: float = 0.001,
            batch_size: int = 1,
            max_iter: int = 100,
            eta: float = 0.001):

        # concatenate 1 as bias term
        xs = np.pad(xs, [(0, 0), (0, 1)], mode='constant', constant_values=1)

        if self.parameter is None:
            feature_size = xs.shape[1]
            self._init_parameter(feature_size)

        for i in range(max_iter):
            old_parameter = self.parameter.copy()
            for batch_xs, batch_ys in batch_generator(xs, ys, batch_size=batch_size):
                self.parameter -= stepsize * self._calc_gradient(batch_xs, batch_ys)

            if ((self.parameter - old_parameter) ** 2).sum() < eta:
                break


def batch_generator(xs: np.array,
                    ys: np.array,
                    batch_size: int = 1,
                    shuffle: bool = True):
    num_instance = xs.shape[0]
    start_idx_list = [i * batch_size for i in range(num_instance // batch_size)]

    if shuffle:
        random.shuffle(start_idx_list)
    for start in start_idx_list:
        end = min(start + batch_size, num_instance)
        yield xs[start:end], ys[start:end]
