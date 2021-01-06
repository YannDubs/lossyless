import torch
import matplotlib.pyplot as plt

__all__ = ["plot_density"]


def setup_grid(range_lim=4, n_pts=1000, device=torch.device("cpu")):
    """
    Return a tensor `xy` of 2 dimensional points (x,y) that span an entire grid [-range_lim,range_lim] 
    with `n_pts` discretizations. `xx` and `xy` can be used for pcolormesh.
    """
    x = torch.linspace(-range_lim, range_lim, n_pts, device=device)
    xx, yy = torch.meshgrid(x, x)
    xy = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, xy


def plot_density(p, n_pts=1000, range_lim=10, figsize=(7, 7), title=None):
    _, ax = plt.subplots(1, 1, figsize=figsize)
    xx, yy, xy = setup_grid(range_lim=range_lim, n_pts=n_pts)
    ax.pcolormesh(
        xx,
        yy,
        torch.exp(p.log_prob(xy)).view(n_pts, n_pts).cpu().data,
        cmap=plt.cm.viridis,
        shading="nearest",
    )

    if title is not None:
        ax.set_title(title)
