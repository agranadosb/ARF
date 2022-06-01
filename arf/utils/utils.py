from typing import Union, List, Iterable

import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import Sequential
from torchvision.transforms import Normalize, Resize

from arf.data import XRayChestDataset


def plot_images(images: Tensor, labels: Tensor, n_rows: int = 1, n_cols: int = 3, size: Union[int, tuple] = 12):
    """Plot a grid of images.

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
    size: Union[int, tuple] = 12
        Size of the figure.
    """
    if isinstance(size, int):
        size = (size, size)
    total_images = n_rows * n_cols

    if size[0] <= 0 and size[1] <= 0:
        raise ValueError("Size must be positive.")
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("Number of rows and columns must be positive.")
    if len(images) < total_images or len(labels) < total_images:
        raise ValueError("Number of rows and columns must be less than the number of images.")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=size)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(labels[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def labels_to_string(labels: Iterable[Tensor], dataset: XRayChestDataset) -> List[str]:
    """Convert a tensor of labels that could be in index, string or one hot
    format to a list of strings.

    Parameters
    ----------
    labels: Iterable[Tensor]
        Array of labels.
    dataset: XRayChestDataset
        Dataset where the labels were created.

    Returns
    -------
    List[str] : List of strings corresponding to the labels.
    """
    return list(map(lambda x: dataset.label_to_string(x), labels))


def normalize_transformation(image_dimensions: int) -> Sequential:
    """Normalize a tensor.

    Parameters
    ----------
    image_dimensions: int
        Width or height of the image.

    Returns
    -------
    Sequential : Sequential transformations for normalize an image.
    """
    return Sequential(
        Normalize([0., 0., 0.], [255., 255., 255.]),
        Resize([image_dimensions, image_dimensions])
    )


def train_transformation(image_dimensions: int) -> Sequential:
    """Train a tensor.

    Parameters
    ----------
    image_dimensions: int
        Width or height of the image.

    Returns
    -------
    Sequential : Sequential transformations for train an image.
    """
    return Sequential(
        Normalize([0., 0., 0.], [255., 255., 255.]),
        Resize([image_dimensions, image_dimensions])
    )


def target_to_one_hot(target: Tensor) -> Tensor:
    """Convert the output of a model to one hot format.
    
    Parameters
    ----------
    target: Tensor
        Output of the model.
        
    Returns
    -------
    Tensor : One hot representation of the target.
    """
    return torch.nn.functional.one_hot(torch.argmax(target, dim=1))  # noqa


def show_results(images: Tensor, labels: Tensor, targets: Tensor, dataset: XRayChestDataset, n_rows: int = 1,
                 n_cols: int = 3, size: Union[int, tuple] = 12):
    """Plot the images, the labels and the targets.
    
    Parameters
    ----------
    images: torch.Tensor
        Array of images to plot.
    labels: torch.Tensor
        Array of labels corresponding to the images.
    targets: torch.Tensor
        Array of targets corresponding to the images.
    dataset: XRayChestDataset
        Dataset where the labels were created.
    n_rows: int = 1
        Number of rows in the grid.
    n_cols: int = 3
        Number of columns in the grid.
    size: Union[int, tuple] = 12
        Size of the figure.
        
    """
    if isinstance(size, int):
        size = (size, size)
    total_images = n_rows * n_cols
    
    if size[0] <= 0 and size[1] <= 0:
        raise ValueError("Size must be positive.")
    
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("Number of rows and columns must be positive.")
    
    if len(images) < total_images or len(labels) < total_images or len(targets) < total_images:
        raise ValueError("Number of rows and columns must be less than the number of images.")
    
    targets = target_to_one_hot(targets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=size)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].cpu().permute(1, 2, 0))
        label = labels_to_string([labels[i].cpu()], dataset)[0]
        target = labels_to_string([targets[i].cpu()], dataset)[0]
        ax.set_title(f"I: {i} - L: {label} - T: {target}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
