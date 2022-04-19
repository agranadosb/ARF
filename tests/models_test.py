from unittest import TestCase

from torch.nn import ReLU

from arf.models import ConvBlock, ResidualBlock, ResNet


class BaseTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_conv_block_relu(self):
        conv_block = ConvBlock(in_channels=3, out_channels=3)
        self.assertEqual(conv_block.conv.in_channels, 3)
        self.assertEqual(conv_block.conv.out_channels, 3)
        self.assertEqual(conv_block.conv.kernel_size, (3, 3))
        self.assertEqual(conv_block.conv.stride, (1, 1))
        self.assertEqual(conv_block.conv.padding, (1, 1))
        self.assertTrue(isinstance(conv_block.relu, ReLU))

    def test_conv_block_no_relu(self):
        conv_block = ConvBlock(in_channels=3, out_channels=3, relu=False)
        self.assertFalse(isinstance(conv_block.relu, ReLU))


class ResNetTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_resnet_same_channels_block(self):
        residual = ResidualBlock(in_channels=3, out_channels=3)
        
        self.assertIsNone(residual.downsample)

    def test_resnet_different_channels_block(self):
        residual = ResidualBlock(in_channels=3, out_channels=1)
    
        self.assertIsNotNone(residual.downsample)

    def test_resnet_same_channels_block_stride_more_than_one(self):
        residual = ResidualBlock(in_channels=3, out_channels=3, stride=2)
    
        self.assertIsNotNone(residual.downsample)

    def test_resnet_different_channels_block_stride_more_than_one(self):
        residual = ResidualBlock(in_channels=3, out_channels=3, stride=2)
    
        self.assertIsNotNone(residual.downsample)

    def test_resnet_architecture(self):
        resnet = ResNet([(1, 3, 64)], num_classes=2)

        self.assertEqual(resnet.classes, 2)
        self.assertEqual(resnet.input_dimensions, (512, 512))
        self.assertEqual(resnet.input_channels, 3)
        self.assertEqual(resnet.layers[0].conv.out_channels, 64)
        self.assertEqual(resnet.layers[0].conv.kernel_size, (7, 7))
        self.assertEqual(resnet.layers[0].conv.stride, (2, 2))
        self.assertEqual(resnet.layers[0].conv.padding, (3, 3))
        self.assertEqual(resnet.layers[1].stride, 2)
        self.assertEqual(resnet.layers[1].kernel_size, 3)
        self.assertEqual(resnet.layers[1].padding, 1)
        self.assertEqual(resnet.layers[-6].conv2.conv.out_channels, 64)
        self.assertEqual(resnet.layers[-6].conv2.conv.kernel_size, (3, 3))

    def test_resnet_architecture_strides(self):
        resnet = ResNet([(2, 3, 64), (2, 3, 128)], num_classes=2)
        # two default layers + 2 layers of a block, so next block starts at 4
        stride_2_index_layer = 2 + 2

        self.assertEqual(resnet.layers[-6].conv2.conv.out_channels, 128)
        self.assertEqual(resnet.layers[-6].conv2.conv.kernel_size, (3, 3))
        self.assertEqual(resnet.layers[2].conv1.conv.out_channels, 64)
        self.assertEqual(resnet.layers[2].conv1.conv.kernel_size, (3, 3))
        self.assertEqual(resnet.layers[2].conv1.conv.stride, (1, 1))
        self.assertEqual(resnet.layers[stride_2_index_layer].conv1.conv.out_channels, 128)
        self.assertEqual(resnet.layers[stride_2_index_layer].conv1.conv.kernel_size, (3, 3))
        self.assertEqual(resnet.layers[stride_2_index_layer].conv1.conv.stride, (2, 2))

    def test_resnet_architecture_strides_many_layers(self):
        resnet = ResNet([(4, 3, 64), (4, 3, 128), (1, 3, 128)], num_classes=2)

        first_stride_2_index_layer = 2 + 4
        second_stride_2_index_layer = 2 + 4 + 4

        self.assertEqual(resnet.layers[first_stride_2_index_layer].conv1.conv.stride, (2, 2))
        self.assertEqual(resnet.layers[second_stride_2_index_layer].conv1.conv.stride, (2, 2))

    def test_resnet_dense_size(self):
        resnet = ResNet([(1, 3, 64), (1, 3, 128), (1, 3, 128)], num_classes=2)
        # 2 stride 2 for first layer, 1 stride 2 per block and 1 stride 1 for last layer.
        dense_size = 128 * (512 // (2 ** (2 + 3))) ** 2

        self.assertEqual(resnet.layers[-3].in_features, dense_size)

    def test_resnet_dense_size_different_input_dimensions_tuple(self):
        resnet = ResNet([(1, 3, 64), (1, 3, 128), (1, 3, 128)], num_classes=2, input_dimensions=(256, 256))
        # 2 stride 2 for first layer, 1 stride 2 per block and 1 stride 1 for last layer.
        dense_size = 128 * (256 // (2 ** (2 + 3))) ** 2

        self.assertEqual(resnet.layers[-3].in_features, dense_size)

    def test_resnet_dense_size_different_input_dimensions_value(self):
        resnet = ResNet([(1, 3, 64), (1, 3, 128), (1, 3, 128)], num_classes=2, input_dimensions=256)
        # 2 stride 2 for first layer, 1 stride 2 per block and 1 stride 1 for last layer.
        dense_size = 128 * (256 // (2 ** (2 + 3))) ** 2

        self.assertEqual(resnet.layers[-3].in_features, dense_size)

    def test_resnet_validate_input_dimensions_incorrect_length(self):
        with self.assertRaises(ValueError):
            ResNet([(1, 3, 64), (1, 3, 128), (1, 3, 128)], num_classes=2, input_dimensions=(256, 256, 256))

    def test_resnet_validate_input_dimensions_incorrect_values_zero(self):
        with self.assertRaises(ValueError):
            ResNet([(1, 3, 64), (1, 3, 128), (1, 3, 128)], num_classes=2, input_dimensions=(0, 0))

    def test_resnet_validate_input_dimensions_incorrect_value_zero(self):
        with self.assertRaises(ValueError):
            ResNet([(1, 3, 64), (1, 3, 128), (1, 3, 128)], num_classes=2, input_dimensions=0)

    def test_resnet_input_dimensions_incorrect_different_values(self):
        with self.assertRaises(ValueError):
            ResNet([(1, 3, 64), (1, 3, 128), (1, 3, 128)], num_classes=2, input_dimensions=(512, 256))

    def test_resnet_input_channels_custom(self):
        resnet = ResNet([(1, 3, 64), (1, 3, 128), (1, 3, 128)], num_classes=2, input_channels=64)

        self.assertEqual(resnet.layers[2].conv1.conv.in_channels, 64)

    def test_resnet_input_classes_custom(self):
        resnet = ResNet([(1, 3, 64), (1, 3, 128), (1, 3, 128)], num_classes=20)

        self.assertEqual(resnet.layers[-3].out_features, 20)

    def test_resnet_validate_blocks_bad_length(self):
        with self.assertRaises(ValueError):
            ResNet([(1, 3)], num_classes=2)  # noqa
    
    def test_resnet_validate_blocks_bad_repetitions_value(self):
        with self.assertRaises(ValueError):
            ResNet([(0, 3, 64)], num_classes=2)

    def test_resnet_validate_blocks_bad_kernel_size_value(self):
        with self.assertRaises(ValueError):
            ResNet([(1, 0, 64)], num_classes=2)

    def test_resnet_validate_blocks_bad_output_channels_value(self):
        with self.assertRaises(ValueError):
            ResNet([(1, 3, 0)], num_classes=2)
