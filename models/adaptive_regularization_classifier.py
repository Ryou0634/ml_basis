import numpy as np


class AdaptiveRegularizationClassifier:
    """
    Optimize Hinge loss by stochastic gradient.

    Inputs
    ----------
    xs : np.array (batch_size, feature_size)
    ys : np.array (batch_size)

    Returns
    -------

    """

    def __init__(self):
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
        self.variance = np.random.normal(size=(feature_size, feature_size)) ** 2

    def _update_param(self, x: np.array, y: np.array, gamma: float):

        beta = x.dot(self.variance).dot(x) + gamma

        x_dot_var = x.dot(self.variance)

        parameter_update = (y * max(0, 1 - x.dot(self.parameter) * y) * x_dot_var) / beta
        variance_update = - np.outer(x_dot_var, x_dot_var) / beta

        self.parameter += parameter_update
        self.variance += variance_update

    def fit(self,
            xs: np.array,
            ys: np.ndarray,
            gamma: float = 0.1,
            max_iter: int = 100,
            eta: float = 0.001):

        # concatenate 1 as bias term
        xs = np.pad(xs, [(0, 0), (0, 1)], mode='constant', constant_values=1)

        if self.parameter is None:
            feature_size = xs.shape[1]
            self._init_parameter(feature_size)

        for i in range(max_iter):
            old_parameter = self.parameter.copy()
            for x, y in zip(xs, ys):
                self._update_param(x, y, gamma)

            if ((self.parameter - old_parameter) ** 2).sum() < eta:
                break
