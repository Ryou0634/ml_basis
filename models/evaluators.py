import numpy as np


def evaluate_regression_model(model, xs, ys):
    predicted = model(xs)
    error = ((predicted - ys) ** 2).sum()
    return error


def cross_split(lis: np.array, k: int):
    assert len(lis) >= k

    test_size = -(-len(lis) // k)  # round up
    for i in range(0, len(lis), test_size):
        j = i + test_size
        test_set = lis[i:j]
        train_set = np.concatenate([lis[:i], lis[j:]])
        yield train_set, test_set


def cross_validation(model, xs, ys, k=10):
    error_list = []
    for (train_xs, test_xs), (train_ys, test_ys) in zip(cross_split(xs, k), cross_split(ys, k)):
        model.fit(np.array(train_xs), np.array(train_ys))
        error = evaluate_regression_model(model, test_xs, test_ys)
        error_list.append(error)

    return sum(error_list) / k
