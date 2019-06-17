import numpy as np
import tqdm

from .modules.solvers import solve_normal_equation, alternating_direction_method_of_multipliers


class LeastSquareProbabilisticGaussKernelClassifier:
    """
    Perform linear classification by Gauss kernel.

    Inputs
    ----------
    xs : np.array (batch_size, feature_size)
    ys : np.array (batch_size)

    Returns
    -------

    """

    def __init__(self,
                 width: float,
                 n_class: int,
                 l2_regularization: float = 0.):
        self.width = width
        self.n_class = n_class
        self.l2_regularization = l2_regularization
        self.kernels = []
        self.parameter = None
        self.num_params = None

    def __call__(self, xs: np.array) -> np.array:
        if len(self.kernels) == 0:
            assert Exception("The kernels is None. You have to perform `fit` before classification.")
        phi = self._calc_design_matrix(xs)
        return self.parameter.dot(phi.T)

    def predict(self, xs: np.array) -> np.array:
        """ Classification """
        score = self(xs)
        return score > 0

    def fit(self, xs: np.array, ys: np.ndarray):
        assert xs.ndim == 2
        assert ys.ndim == 1
        assert len(set(ys)) <= self.n_class

        self._set_kernels(xs)
        phi = self._calc_design_matrix(xs)

        parameter_list = []
        for c in range(self.n_class):
            param = solve_normal_equation(phi, ys == c, self.l2_regularization)
            parameter_list.append(param)
        self.parameter = np.stack(parameter_list)

    def _set_kernels(self, xs: np.array):
        self.num_params = len(xs)
        self.kernels = xs

    def _calc_design_matrix(self, xs: np.array):
        phi = [self._calc_gauss(x, self.kernels) for x in tqdm.tqdm(xs)]
        return np.vstack(phi)

    def _calc_gauss(self, x: np.array, kernels: np.array):
        denom = ((x - kernels) ** 2)
        if denom.ndim == 2:
            denom = denom.sum(axis=1)
        return np.exp(- denom / (2 * (self.width ** 2)))
