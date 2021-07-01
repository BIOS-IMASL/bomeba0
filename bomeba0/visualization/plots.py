"""
A collections of plots
"""
import matplotlib.pyplot as plt
from .kde import plot_kde


def plot_ramachandran(pose, kind='scatter', contour=True, scatter_kwargs=None, ax=None):
    """
    Ramachandran plot

    Parameters
    ----------
    pose : protein object
    kind : str
        Acepted values as `scatter`, `kde` or `sca+kde`. Defaults to scatter
    alpha : float
        opacity level, should be between 0 (transparent) to 1 (opaque), only works with `kind=scatter`
    contour : bool
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE. Defaults to True.
        Only works for `kind=kde`
    ax : axes
        Matplotlib axes. Defaults to None.

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        ax = plt.gca()
    if scatter_kwargs == None:
        scatter_kwargs = {}

    phi = []
    psi = []
    for i in range(len(pose)):
        phi.append(pose.get_phi(i))
        psi.append(pose.get_psi(i))

    if kind not in ['scatter', 'kde', 'sca+kde']:
        raise ValueError(f"kind should be 'scatter', 'hexbin' or 'kde'not {kind}")

    if kind == 'scatter':
        ax.scatter(phi, psi, **scatter_kwargs)
    elif kind == 'kde':
        plot_kde(phi, psi, contour, ax=ax)
    else:
        scatter_kwargs.setdefault("marker", ".")
        scatter_kwargs.setdefault("color", "k")
        plot_kde(phi, psi, contour, ax=ax)
        ax.scatter(phi, psi, **scatter_kwargs)

    ax.set_xlabel(r'$\phi$', fontsize=16)
    ax.set_ylabel(r'$\psi$', fontsize=16, rotation=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)

    return ax
