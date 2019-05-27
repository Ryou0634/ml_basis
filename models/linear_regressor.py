import numpy as np
from typing import List, Callable

from .modules.solvers import alternating_direction_method_of_multipliers, solve_normal_equation


class LinearRegressor:
    """
    Perform linear classification.

    Inputs
    ----------
    xs : np.array (batch_size, feature_size)
    ys : np.array (batch_size)

    Returns
    -------

    """

    def __init__(self,
                 basis_functions: List[Callable],
                 l1_regularization: float = 0.,
                 l2_regularization: float = 0.):
        self.basis_functions = basis_functions
        if l1_regularization != 0 and l2_regularization != 0:
            assert Exception("The model currently does not support apply both l1 and l2 regularizations.")
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.parameter = None

    def __call__(self, xs: np.array):
        """
        Perform prediction.
        The output is y = w * phi(x)
        """
        if self.parameter is None:
            assert Exception("The parameter is None. You have to perform `fit` before regression.")
        phi = self._calc_design_matrix(xs)
        return self.parameter.dot(phi.T)

    def fit(self, xs: np.array, ys: np.ndarray):

        # calculate design matrix
        phi = self._calc_design_matrix(xs)
        if self.l1_regularization:
            self.parameter = alternating_direction_method_of_multipliers(phi, ys, self.l1_regularization)
        else:
            self.parameter = solve_normal_equation(phi, ys, self.l2_regularization)

    def _calc_design_matrix(self, xs):
        phi = np.hstack([func(xs) for func in self.basis_functions])
        return phi
