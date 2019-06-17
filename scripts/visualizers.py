import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def draw_line(data, n_plots=None):
    """
    Draw a line graph given from an array.
    If n_plots is smaller that the size of data, the graph will be smoothed.

    Parameters
    ----------
    data : List
        Data points
    n_plots : int
        a number of plots
    """

    if n_plots is None:
        n_plots = len(data)
    batches = np.array_split(range(len(data)), n_plots)
    ary = np.array(data)
    avgs = [np.mean(ary[i]) for i in batches]
    plt.figure()
    plt.plot(range(len(avgs)), avgs)
    plt.show()
    return


def draw_scatter(features, labels, classifier=None):
    """
    Draw a scatter plot given 2d-features and its labels.
    If classfier provided, it shows the separating hyperplane.

    Parameters
    ----------
    features : 2d-array (n_data, 2)
        Data points
    labels : array
        Labels represented by integers.
    classifier:
        A classifier with predict() method which takes features and outputs its labels.
    """
    n_labels = len(set(labels))

    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
    cmap = ListedColormap(colorlist[:n_labels])

    divided_features = [[] for _ in range(n_labels)]
    for feature, label in zip(features, labels):
        divided_features[label].append(feature)

    for i in range(n_labels):
        np_fs = np.array(divided_features[i])
        plt.scatter(np_fs[:, 0], np_fs[:, 1], color=colorlist[i])

    if classifier:
        x1_min, x1_max = features[:, 0].min(), features[:, 0].max()
        resolution = (x1_max - x1_min) / 200
        margin = resolution * 10
        x1_min, x1_max = x1_min - margin, x1_max + margin
        x2_min, x2_max = features[:, 1].min() - margin, features[:, 1].max() + margin
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.show()
    return


def plot_regression(xs, ys, classifier=None, resolution: int = 1000):
    """
    Draw a graph.

    Parameters
    ----------
    xs : 2d-array (n_data, 1)
        Data points
    ys : array (n_data, )
        Traget values
    classifier:
        A classifier with __call__() method which takes features and outputs the target value.
    """
    assert xs.shape[1] == 1
    assert ys.ndim == 1

    plt.scatter(xs, ys)

    if classifier:
        classifier_xs = np.linspace(start=min(xs), stop=max(xs), num=resolution).reshape(-1, 1)
        classifier_ys = classifier(classifier_xs)
        plt.plot(classifier_xs, classifier_ys)

    plt.show()
    return
