from unittest import TestCase
from unittest.mock import patch

from torch import ones

from arf.constants import INDEX_LABEL, ONE_HOT_LABEL
from arf.data import XRayChestDataset
from arf.utils import plot_images, labels_to_string


@patch('arf.utils.plt.show', lambda: None)
class TestUtils(TestCase):
    def setUp(self) -> None:
        self.images = ones(3, 512, 512, 3)
        self.labels = ones(3)
        self.folder = 'tests/static/data/'
        super().setUp()

    def test_plot_images_correctly(self):
        plot_images(images=self.images, labels=self.labels)

    def test_plot_images_bad_size_int(self):
        with self.assertRaises(ValueError):
            plot_images(images=self.images, labels=self.labels, size=0)

    def test_plot_images_bad_size_tuple(self):
        with self.assertRaises(ValueError):
            plot_images(images=self.images, labels=self.labels, size=(0, 0))

    def test_plot_images_bad_rows(self):
        with self.assertRaises(ValueError):
            plot_images(images=self.images, labels=self.labels, n_rows=0)

    def test_plot_images_bad_columns(self):
        with self.assertRaises(ValueError):
            plot_images(images=self.images, labels=self.labels, n_cols=0)

    def test_plot_images_less_images(self):
        with self.assertRaises(ValueError):
            plot_images(images=self.images[:1], labels=self.labels)

    def test_plot_images_less_labels(self):
        with self.assertRaises(ValueError):
            plot_images(images=self.images, labels=self.labels[:1])

    def test_labels_to_string_text_correctly(self):
        dataset = XRayChestDataset(self.folder, init=True)
        labels = [label for _, label in dataset[:]]
        correct_labels = ["class1", "class1", "class2", "class2"]

        result = labels_to_string(labels, dataset)
        
        self.assertListEqual(result, correct_labels)

    def test_labels_to_string_index_correctly(self):
        dataset = XRayChestDataset(self.folder, init=True, label_type=INDEX_LABEL)
        labels = [label for _, label in dataset[:]]
        correct_labels = ["class1", "class1", "class2", "class2"]
    
        result = labels_to_string(labels, dataset)
    
        self.assertListEqual(result, correct_labels)

    def test_labels_to_string_one_hot_correctly(self):
        dataset = XRayChestDataset(self.folder, init=True, label_type=ONE_HOT_LABEL)
        labels = [label for _, label in dataset[:]]
        correct_labels = ["class1", "class1", "class2", "class2"]
    
        result = labels_to_string(labels, dataset)
    
        self.assertListEqual(result, correct_labels)
