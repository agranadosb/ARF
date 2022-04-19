from unittest import TestCase
from unittest.mock import patch

from torch import ones

from arf.utils import plot_images


@patch('arf.utils.plt.show', lambda: None)
class TestUtils(TestCase):
    def setUp(self) -> None:
        self.images = ones(3, 512, 512, 3)
        self.labels = ones(3)
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
