import os
from typing import List, Union, Dict, Optional

import torch
from torch import Tensor
from torch import jit
from torch.nn import Sequential
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from arf.constants import TEXT_LABEL, LABEL_TYPE_OPTIONS, ONE_HOT_LABEL, PAIR_TYPE, INDEX_LABEL


class XRayChestDataset(Dataset):
    r"""A class for loading the X-Ray Chest dataset. This class gets a folder
    with the data split in classes and returns a Dataset with the
    functionality of a pytorch Dataset with the images on the folder. This
    class allows to set the sample label as one hot, index or text (text by
    default).
    
    Parameters
    ----------
    folder : str
        The folder with the data divided in folders where each folder
        represents a class.
    label_type : str = text
        The type of label to be returned. This type should be one of the following:
        - 'one_hot': The label will be a one-hot vector.
        - 'index': The label will be an integer.
        - 'text': The label will be a string.
    channels_first : bool, optional = False
        If True, the images will be returned with the channels first.
    images_transformations: torch.nn.Sequential, optional = None
        The transforms to be applied to the images.
    init: bool, optional = False
        If True, the dataset will be initialized.
    """
    
    def __init__(self, folder: str, *, label_type: str = TEXT_LABEL, channels_first: Optional[bool] = None,
                 images_transformations: Optional[Sequential] = None,
                 init: Optional[bool] = None) -> None:
        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)
        
        if not os.path.exists(folder):
            raise FileNotFoundError(f'The folder {folder} does not exist')
        
        if label_type is None:
            label_type = TEXT_LABEL
        if label_type not in LABEL_TYPE_OPTIONS:
            raise TypeError(f'label_type must be one of {LABEL_TYPE_OPTIONS}')
        
        self.folder = folder
        self.samples: List[str] = []
        self.labels: List[str] = []
        self.label_mapping: Dict[int, str] = dict()
        self._len = 0
        self.channels_first = channels_first
        self.label_type = label_type
        self.transforms = None
        if images_transformations is not None:
            self.transforms = jit.script(images_transformations)
        
        if init:
            self._init_dataset()
    
    def _init_dataset(self) -> None:
        r"""Initializes the dataset creating a list with samples and a list
        with associated labels. The samples are filenames without full path"""
        index_class = 0
        labels = []
        for class_folder in sorted(os.listdir(self.folder)):
            samples_list = sorted(os.listdir(
                os.path.join(self.folder, class_folder)
            ))
            self.label_mapping[index_class] = class_folder
            self.samples.extend(samples_list)
            labels.extend([index_class for _ in samples_list])
            index_class += 1
        
        self.labels = Tensor(labels).to(torch.int64)
        if self.label_type == ONE_HOT_LABEL:
            self.labels = one_hot(Tensor(labels).to(torch.int64), len(self.label_mapping))
        
        self._len = len(self.samples)
    
    def __len__(self) -> int:
        r"""Returns the length of the dataset.
        
        Returns
        -------
            int: The length of the dataset.
        """
        return self._len
    
    def __getitem__(self, idx: int) -> Union[List[PAIR_TYPE], PAIR_TYPE]:
        r"""Returns the image at the index idx.

        Parameters
        ----------
        idx : int
            The index of the image to be returned.

        Returns
        -------
        Union[List[`arf.constants.PAIR_TYPE`], `arf.constants.PAIR_TYPE`]:
        A list with the images and the labels or tuple with the image and
        the label if only one sample is selected.
        """
        filenames = self.samples[idx]
        true_labels: Union[str, Tensor] = self.labels[idx]
        
        if isinstance(filenames, str):
            filenames = [filenames]
        
        is_label_native = isinstance(true_labels, str) or true_labels.shape == torch.Size([])
        is_label_one_hot_with_one_item = (
                self.label_type == ONE_HOT_LABEL and true_labels.shape == torch.Size([len(self.label_mapping)])
        )
        labels = true_labels
        if is_label_native or is_label_one_hot_with_one_item:
            labels = [true_labels]
        
        assert len(filenames) == len(labels)
        
        samples = []
        for filename, label in zip(filenames, labels):
            if self.label_type == ONE_HOT_LABEL:
                index_text_label = label.nonzero().item()
            else:
                index_text_label = label.item()
            
            text_label = self.label_mapping[index_text_label]
            
            image_path = os.path.join(self.folder, text_label, filename)
            tensor = read_image(image_path, ImageReadMode.RGB).float()
            channels = transforms.functional.get_image_num_channels(tensor)
            
            if self.transforms is not None:
                tensor = self.transforms(tensor)
            
            if channels == tensor.shape[0] and not self.channels_first:
                tensor = tensor.permute(1, 2, 0)
            
            samples_label = label
            if self.label_type == TEXT_LABEL:
                samples_label = text_label
            elif self.label_type == INDEX_LABEL:
                samples_label = index_text_label
            
            samples.append((tensor, samples_label))
        
        if len(samples) == 1:
            return samples[0]
        return samples
