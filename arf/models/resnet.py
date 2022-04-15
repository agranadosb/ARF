from typing import Tuple, List, Union

from torch import Tensor, add
from torch.nn import Module, Conv2d, MaxPool2d, AvgPool2d, Flatten, Linear, ReLU, Softmax

from arf.models.base import ConvBlock


class ResidualBlock(Module):
    """This class implements a residual block.
    args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolutional kernel
        stride (int): stride of the convolution
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, 1)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, 1, 1, relu=False)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the residual block.
        args:
            x: input tensor

        returns:
            x: output tensor
        """
        out = self.conv2(self.conv1(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return add(out, x)


class ResNet(Module):
    """This class implements a resnet network. It receives a list of blocks
    that define how are the parameters of the convolutional layers and how many
    repetitions of the block will be applied
    
    args:
        blocks (ResidualBlock): List of blocks of the network. Each block is a
            tuple with the number of repetitions of the block, output channels and
            kernel size. For example, a block that will be repeated 3 times with
            output channels of 32 and kernel size of 3 will be defined as:
             - `(3, 32, 3)`
        num_classes (int): number of classes to be predicted
        input_channels (int): number of input channels
        input_dimensions (int): number of input dimensions
    """
    
    def __init__(self, blocks: List[Tuple[int, int, int]], *, num_classes: int = 10, input_channels: int = 3,
                 input_dimensions: Union[int, tuple] = 512) -> None:
        super().__init__()
        if isinstance(input_dimensions, int):
            input_dimensions = (input_dimensions, input_dimensions)
        
        if len(input_dimensions) != 2:
            raise ValueError("The input dimensions must be a tuple of two dimensions")
        if input_dimensions[0] != input_dimensions[1] or len(input_dimensions) != 2:
            raise ValueError("The input dimensions must be square")
        if any(dim < 1 for dim in input_dimensions):
            raise ValueError("The input dimensions must be greater than zero")

        self.input_dimensions = input_dimensions
        self.blocks = blocks
        self.classes = num_classes
        self.input_channels = input_channels
        # 2 stride 2 for first layer, 1 stride 2 per block and 1 stride 1 for last layer.
        dense_size = self.input_dimensions[0] // (2 ** (2 + len(self.blocks) + 1))
        
        layers = [
            ConvBlock(input_channels, 64, kernel_size=7, stride=2, padding=3),
            MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        stride = 1
        for number_blocks, kernel_size, out_channels in blocks:
            for index in range(number_blocks):
                layers.append(ResidualBlock(input_channels, out_channels, kernel_size, stride=stride))
                input_channels = out_channels
                stride = 1
            stride = 2
        
        layers.extend([
            AvgPool2d(kernel_size=2, stride=1),
            Flatten(),
            Linear(in_features=dense_size, out_features=num_classes),
            ReLU(),
            Softmax()
        ])
        
        self.layers = layers

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass of the network.
            args:
                x: input tensor

            returns:
                x: output tensor
            """
            for layer in self.layers:
                x = layer(x)
            return x
