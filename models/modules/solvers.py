import numpy as np


def solve_normal_equation(phi: np.ndarray,
                          targets: np.ndarray,
                          l2_regularization: float = 0.) -> np.ndarray:
    """

    Parameters
    ----------
    phi : np.array (batch_size, feature_size)
        design matrix.
    targets : np.array (batch_size)
        target values.
    l2_regularization : float
        regularization term.

    Returns
    -------
    parameter : np.array (feature_size)

    """
    batch_size, num_params = phi.shape
    return np.linalg.solve(phi.T.dot(phi) + l2_regularization * np.eye(num_params),
                           phi.T.dot(targets[:, None])).reshape(-1)


def alternating_direction_method_of_multipliers(phi: np.ndarray,
                                                ys: np.ndarray,
                                                l1_regularization: float = 0.,
                                                max_iter: int = 1000,
                                                eps: float = 0.0001) -> np.ndarray:
    """
    A method to optimize cost function with l1 regularization.
    Iteratively, update parameter until it reaches convergence.
    """

    batch_size, num_params = phi.shape
    theta = np.random.rand(num_params)
    z = np.random.rand(num_params)
    u = np.random.rand(num_params)

    for i in range(max_iter):
        old_theta = theta
        # update theta
        term1 = np.linalg.inv((phi.T.dot(phi) + np.eye(phi.shape[1])))
        term2 = phi.T.dot(ys) + z - u
        theta = term1.dot(term2)

        if ((theta - old_theta) ** 2).sum() < eps:
            # convergence
            break

        # update z
        tmp1 = theta + u - l1_regularization
        tmp2 = - theta - u - l1_regularization
        z = np.where(tmp1 < 0, 0, tmp1) - np.where(tmp2 < 0, 0, tmp2)

        # update u
        u = u + theta - z

    return theta
