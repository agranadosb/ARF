from typing import Optional

from matplotlib import pyplot as plt
from torch import Tensor


def plot_images(images: Tensor, labels: Tensor, n_rows: Optional[int] = None, n_cols: Optional[int] = None,
                size: Optional[tuple] = None):
    """
    Plot a grid of images.

    Parameters
    ----------
    images: torch.Tensor
        Array of images to plot.
    labels: torch.Tensor
        Array of labels corresponding to the images.
    n_rows: int = 1
        Number of rows in the grid.
    n_cols: int = 3
        Number of columns in the grid.
    size: tuple = (12, 12)
        Size of the figure.

    Returns
    -------
    None
    """
    if n_rows is None:
        n_rows = 1
    if n_cols is None:
        n_cols = 3
    if size is None:
        size = (12, 12)
    if isinstance(size, int):
        size = (size, size)
    
    if size[0] <= 0 and size[1] <= 0:
        raise ValueError("Size must be positive.")
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("Number of rows and columns must be positive.")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=size)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(labels[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()