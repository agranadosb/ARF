from unittest import TestCase

import numpy as np
import torch
from torchvision import transforms

from arf.data import XRayChestDataset


class DataTest(TestCase):
    def setUp(self):
        self.folder = 'tests/static/data/'
        self.dataset = XRayChestDataset(self.folder)
        super().setUp()
    
    def test_create_dataset_correctly(self):
        correct_folder = 'tests/static/data/'
        
        self.assertEqual(self.dataset.folder, correct_folder)
        self.assertListEqual(self.dataset.samples, [])
        self.assertListEqual(self.dataset.labels, [])
        self.assertEqual(self.dataset._len, 0)
    
    def test_init_dataset_correctly(self):
        correct_samples = ["test1.png", "test2.png", "test1.png", "test2.png"]
        correct_labels = ["class1", "class1", "class2", "class2"]
        correct_length = 4
        
        self.dataset._init_dataset()
        
        self.assertListEqual(self.dataset.samples, correct_samples)
        self.assertListEqual(self.dataset.labels, correct_labels)
        self.assertEqual(self.dataset._len, correct_length)
    
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
