import os
from typing import List, Tuple, Union

from torch import Tensor
from torch import jit
from torch.nn import Sequential
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode


class XRayChestDataset(Dataset):
    r"""A class for loading the X-Ray Chest dataset. This class gets a folder
    with the data split in classes and returns a Dataset with the
    functionality of a pytorch Dataset with the images on the folder.
    
    Args:
        folder (str): The folder with the data divided in folders where each
        folder represents a class.
        images_transformations (torchvision.transforms.Compose): The transforms to be applied to the images.
        channels_first (bool): If True, the images will be returned with the channels first.
    """
    
    def __init__(self, folder: str, *, images_transformations: Sequential = None, channels_first: bool = False) -> None:
        self.folder = folder
        self.samples: List[str] = []
        self.labels: List[str] = []
        self._len = 0
        self.channels_first = channels_first
        self.transforms = None
        if images_transformations is not None:
            self.transforms = jit.script(images_transformations)

    def _init_dataset(self) -> None:
        r"""Initializes the dataset creating a list with samples and a list
        with associated labels. The samples are filenames without full path"""
        for class_folder in sorted(os.listdir(self.folder)):
            samples_list = sorted(os.listdir(
                os.path.join(self.folder, class_folder)
            ))
            self.samples.extend(samples_list)
            self.labels.extend([class_folder for _ in samples_list])
        
        self._len = len(self.samples)
    
    def __len__(self) -> int:
        r"""Returns the length of the dataset.
        
        Returns:
            int: The length of the dataset.
        """
        return self._len
    
    def __getitem__(self, idx: int) -> Union[List[Tuple[Tensor, str]], Tuple[Tensor, str]]:
        r"""Returns the image at the index idx.
        
        Args:
            idx (int): The index of the image.
        
        Returns:
            List[Tuple[Tensor, str]]: A list with the images and the labels or
            tuple with the image and the label if only one sample is selected.
        """
        filenames = self.samples[idx]
        labels = self.labels[idx]
        
        if isinstance(filenames, str):
            filenames = [filenames]
        if isinstance(labels, str):
            labels = [labels]
        
        assert len(filenames) == len(labels)
        
        samples = []
        for filename, label in zip(filenames, labels):
            tensor = read_image(os.path.join(self.folder, label, filename), ImageReadMode.RGB).float()
            channels = transforms.functional.get_image_num_channels(tensor)
    
            if self.transforms is not None:
                tensor = self.transforms(tensor)

            if channels == tensor.shape[0] and not self.channels_first:
                tensor = tensor.permute(1, 2, 0)
    
            samples.append((tensor, label))
        
        if len(samples) == 1:
            return samples[0]
        return samples
