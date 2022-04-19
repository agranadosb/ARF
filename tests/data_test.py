import os.path
from unittest import TestCase

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms

from arf.data import XRayChestDataset


class DataTest(TestCase):
    def setUp(self):
        self.folder = 'tests/static/data/'
        self.dataset = XRayChestDataset(self.folder)
        super().setUp()
    
    def test_create_dataset_correctly(self):
        correct_folder = os.path.join(os.getcwd(), 'tests/static/data/')

        self.assertEqual(self.dataset.folder, correct_folder)
        self.assertListEqual(self.dataset.samples, [])
        self.assertListEqual(self.dataset.labels, [])
        self.assertEqual(self.dataset._len, 0)
    
    def test_init_dataset_correctly(self):
        correct_samples = ["test1.png", "test2.png", "test1.png", "test2.png"]
        correct_labels = Tensor([0, 0, 1, 1])
        correct_length = 4
        
        self.dataset._init_dataset()
        
        self.assertListEqual(self.dataset.samples, correct_samples)
        self.assertTrue(torch.all(self.dataset.labels.eq(correct_labels)))
        self.assertEqual(self.dataset._len, correct_length)

    def test_dataset_incorrect_folder(self):
        folder = 'notExists/static/data/'

        with self.assertRaises(FileNotFoundError):
            XRayChestDataset(folder)

    def test_len_dataset_correctly(self):
        correct_length = 4
        
        self.dataset._init_dataset()
        
        self.assertEqual(self.dataset._len, correct_length)
    
    def test_getitem_dataset_correctly(self):
        correct_tensor = torch.tensor(np.ones((400, 400, 3), dtype=np.float16) * 255)
        correct_label = "class1"
        
        self.dataset._init_dataset()
        item, label = self.dataset[0]
        
        self.assertEqual(item.shape, correct_tensor.shape)
        self.assertEqual(item.type(), 'torch.FloatTensor')
        self.assertTrue(torch.all(item.eq(correct_tensor)))
        self.assertEqual(label, correct_label)
    
    def test_getitem_dataset_out_of_range(self):
        self.dataset._init_dataset()
        with self.assertRaises(IndexError):
            self.dataset[4]
    
    def test_getitem_two_items(self):
        correct_numpy = np.ones((400, 400, 3), dtype=np.float16) * 255
        correct_tensor1 = torch.tensor(correct_numpy.copy())
        correct_tensor2 = torch.tensor(correct_numpy.copy())
        correct_label1 = "class1"
        correct_label2 = "class2"
        
        self.dataset._init_dataset()
        items = self.dataset[1:3]
        
        self.assertEqual(len(items), 2)
        self.assertTrue(torch.all(items[0][0].eq(correct_tensor1)))
        self.assertTrue(torch.all(items[1][0].eq(correct_tensor2)))
        self.assertEqual(items[0][1], correct_label1)
        self.assertEqual(items[1][1], correct_label2)
    
    def test_getitem_all_items(self):
        correct_numpy = np.ones((400, 400, 3), dtype=np.float16) * 255
        tensors = [torch.tensor(correct_numpy.copy()) for _ in range(4)]
        labels = ["class1", "class1", "class2", "class2"]
        
        self.dataset._init_dataset()
        items = self.dataset[:]
        
        self.assertEqual(len(items), 4)
        for index, (tensor, label) in enumerate(items):
            self.assertTrue(torch.all(tensor.eq(tensors[index])))
            self.assertEqual(label, labels[index])
    
    def test_set_transforms_parameter_incorrectly(self):
        correct_transforms = torch.nn.Sequential(
            transforms.Normalize([0., 0., 0.], [255., 255., 255.])
        )
        
        with self.assertRaises(TypeError):
            XRayChestDataset(self.folder, correct_transforms)
    
    def test_transforms(self):
        correct_transforms = torch.nn.Sequential(
            transforms.Normalize([0., 0., 0.], [255., 255., 255.])
        )
        correct_numpy = np.ones((400, 400, 3), dtype=np.float16)
        dataset = XRayChestDataset(self.folder, images_transformations=correct_transforms)
        
        dataset._init_dataset()
        item, label = dataset[0]
        
        self.assertTrue(torch.all(item.eq(torch.tensor(correct_numpy))))
    
    def test_channels_first(self):
        correct_numpy = np.ones((3, 400, 400), dtype=np.float16)
        dataset = XRayChestDataset(self.folder, channels_first=True)
        
        dataset._init_dataset()
        item, label = dataset[0]
        
        self.assertEqual(item.shape, correct_numpy.shape)

    def test_label_type_incorrectly(self):
        with self.assertRaises(TypeError):
            XRayChestDataset(self.folder, label_type="incorrect")

    def test_label_type_one_hot(self):
        correct_label = torch.tensor([1, 0])
        dataset = XRayChestDataset(self.folder, label_type="one hot")
        correct_class = "class1"

        dataset._init_dataset()
        _, label = dataset[0]
        item_class = dataset.label_mapping[correct_label.nonzero().item()]

        self.assertTrue(torch.all(label.eq(torch.tensor(correct_label))))
        self.assertEqual(item_class, correct_class)

    def test_getitem_two_items_one_hot(self):
        dataset = XRayChestDataset(self.folder, label_type="one hot")
        correct_label1 = torch.tensor([1, 0])
        correct_label2 = torch.tensor([0, 1])
        correct_class1 = "class1"
        correct_class2 = "class2"

        dataset._init_dataset()
        items = dataset[1:3]
        item_class1 = dataset.label_mapping[items[0][1].nonzero().item()]
        item_class2 = dataset.label_mapping[items[1][1].nonzero().item()]

        self.assertTrue(torch.all(items[0][1].eq(torch.tensor(correct_label1))))
        self.assertTrue(torch.all(items[1][1].eq(torch.tensor(correct_label2))))
        self.assertEqual(item_class1, correct_class1)
        self.assertEqual(item_class2, correct_class2)

    def test_getitem_all_items_one_hot(self):
        dataset = XRayChestDataset(self.folder, label_type="one hot")
        correct_labels = [
            torch.tensor([1, 0]), torch.tensor([1, 0]), torch.tensor([0, 1]), torch.tensor([0, 1])
        ]
    
        dataset._init_dataset()
        items = dataset[:]

        self.assertEqual(len(items), 4)
        for index, (_, label) in enumerate(items):
            self.assertTrue(torch.all(label.eq(correct_labels[index])))

    def test_label_type_index(self):
        correct_label = 0
        dataset = XRayChestDataset(self.folder, label_type="index")
        correct_class = "class1"
    
        dataset._init_dataset()
        _, label = dataset[0]
        item_class = dataset.label_mapping[0]
    
        self.assertEqual(label, correct_label)
        self.assertEqual(item_class, correct_class)

    def test_getitem_two_items_index(self):
        dataset = XRayChestDataset(self.folder, label_type="index")
        correct_label1 = 0
        correct_label2 = 1
        correct_class1 = "class1"
        correct_class2 = "class2"
    
        dataset._init_dataset()
        items = dataset[1:3]
        item_class1 = dataset.label_mapping[items[0][1]]
        item_class2 = dataset.label_mapping[items[1][1]]
    
        self.assertEqual(items[0][1], correct_label1)
        self.assertEqual(items[1][1], correct_label2)
        self.assertEqual(item_class1, correct_class1)
        self.assertEqual(item_class2, correct_class2)

    def test_getitem_all_items_index(self):
        dataset = XRayChestDataset(self.folder, label_type="index")
        correct_labels = [0, 0, 1, 1]
    
        dataset._init_dataset()
        items = dataset[:]
    
        self.assertEqual(len(items), 4)
        for index, (_, label) in enumerate(items):
            self.assertEqual(label, correct_labels[index])
