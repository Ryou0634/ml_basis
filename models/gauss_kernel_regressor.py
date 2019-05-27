import numpy as np
import tqdm

from .modules.solvers import solve_normal_equation, alternating_direction_method_of_multipliers


class GaussKernelRegressor:
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
                 l1_regularization: float = 0.,
                 l2_regularization: float = 0.):
        self.width = width
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.kernels = []
        self.parameter = None
        self.num_params = None

    def __call__(self, xs: np.array):
        if len(self.kernels) == 0:
            assert Exception("The kernels is None. You have to perform `fit` before classification.")
        phi = self._calc_design_matrix(xs)
        return self.parameter.dot(phi.T)

    def fit(self, xs: np.array, ys: np.ndarray):
        assert xs.ndim == 2
        assert ys.ndim == 1

        self._set_kernels(xs)
        phi = self._calc_design_matrix(xs)

        if self.l1_regularization:
            self.parameter = alternating_direction_method_of_multipliers(phi, ys, self.l1_regularization)
        else:
            self.parameter = solve_normal_equation(phi, ys, self.l2_regularization)

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
