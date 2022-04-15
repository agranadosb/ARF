from typing import Union, Callable

from torch import Tensor
from torch.nn import Module, Conv2d, BatchNorm2d, ReLU


class ConvBlock(Module):
    """This class implements a convolutional block. It is composed of a
    convolutional layer, a batch normalization layer, and a ReLU activation
    function.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolutional kernel
        stride (int): stride of the convolutional kernel
        padding (int): padding of the convolutional kernel
        relu (bool): whether to use a ReLU activation function
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: Union[str, int] = 1,
                 relu: bool = True):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = BatchNorm2d(out_channels)
        self.relu: Callable = lambda x: x
        if relu:
            self.relu: Callable = ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.relu(self.bn(self.conv(x)))
