"""One-dimensional kernel density estimate plots."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian, convolve, convolve2d  # pylint: disable=no-name-in-module
from scipy.sparse import coo_matrix
from scipy.stats import entropy


def plot_kde(values, values2=None, contour=True, ax=None):
    """2D KDE plot taking into account boundary conditions.

    The code was adapted from arviz library

    Parameters
    ----------
    values : array-like
        Values to plot
    values2 : array-like, optional
        Values to plot. If present, a 2D KDE will be estimated
    contour : bool
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE. Defaults to True.
    ax : matplotlib axes

    Returns
    -------
    ax : matplotlib axes

    """
    if ax is None:
        ax = plt.gca()

    else:

        gridsize = (128, 128) if contour else (256, 256)

        density, xmin, xmax, ymin, ymax = _fast_kde_2d(values, values2, gridsize=gridsize)
        g_s = complex(gridsize[0])
        x_x, y_y = np.mgrid[xmin:xmax:g_s, ymin:ymax:g_s]

        ax.grid(False)
        if contour:
            qcfs = ax.contourf(x_x, y_y, density, antialiased=True)
            qcfs.collections[0].set_alpha(0)
        else:
            ax.pcolormesh(x_x, y_y, density)

    return ax


def _fast_kde_2d(x, y, gridsize=(128, 128), circular=True):
    """
    2D fft-based Gaussian kernel density estimate (KDE).

    The code was adapted from https://github.com/mfouesneau/faststats

    Parameters
    ----------
    x : Numpy array or list
    y : Numpy array or list
    gridsize : tuple
        Number of points used to discretize data. Use powers of 2 for fft optimization
    circular: bool
        If True, use circular boundaries. Defaults to False
    Returns
    -------
    grid: A gridded 2D KDE of the input points (x, y)
    xmin: minimum value of x
    xmax: maximum value of x
    ymin: minimum value of y
    ymax: maximum value of y
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    len_x = len(x)
    weights = np.ones(len_x)
    n_x, n_y = gridsize

    d_x = (xmax - xmin) / (n_x - 1)
    d_y = (ymax - ymin) / (n_y - 1)

    xyi = np.vstack((x, y)).T
    xyi -= [xmin, ymin]
    xyi /= [d_x, d_y]
    xyi = np.floor(xyi, xyi).T

    scotts_factor = len_x ** (-1 / 6)
    cov = np.cov(xyi)
    std_devs = np.diag(cov ** 0.5)
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    inv_cov = np.linalg.inv(cov * scotts_factor ** 2)

    x_x = np.arange(kern_nx) - kern_nx / 2
    y_y = np.arange(kern_ny) - kern_ny / 2
    x_x, y_y = np.meshgrid(x_x, y_y)

    kernel = np.vstack((x_x.flatten(), y_y.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel
    kernel = np.exp(-kernel.sum(axis=0) / 2)
    kernel = kernel.reshape((int(kern_ny), int(kern_nx)))

    boundary = "wrap" if circular else "symm"

    grid = coo_matrix((weights, xyi), shape=(n_x, n_y)).toarray()
    grid = convolve2d(grid, kernel, mode="same", boundary=boundary)

    norm_factor = np.linalg.det(2 * np.pi * cov * scotts_factor ** 2)
    norm_factor = len_x * d_x * d_y * norm_factor ** 0.5

    grid /= norm_factor

    return grid, xmin, xmax, ymin, ymax

