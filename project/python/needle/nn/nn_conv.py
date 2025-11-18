"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding = kernel_size // 2
        w_shape = (kernel_size, kernel_size, in_channels, out_channels)
        W = init.kaiming_uniform(
            in_channels, out_channels, shape=w_shape, nonlinearity="relu", device=device, dtype=dtype
        )
        self.weight = Parameter(W)
        if bias:
            denom = float(in_channels * (kernel_size ** 2))
            bound = 1.0 / (denom ** 0.5)
            b = init.rand(out_channels, low=-bound, high=bound, device=device, dtype=dtype)
            self.bias = Parameter(b)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_nhwc = ops.transpose(x, (0, 2, 3, 1))
        y_nhwc = ops.conv(
            x_nhwc,
            self.weight,
            stride=self.stride,
            padding=self.padding,
        )

        # Only add bias if it exists
        if self.bias is not None:
            bias_b = ops.broadcast_to(
                ops.reshape(self.bias, (1, 1, 1, self.out_channels)),
                y_nhwc.shape,
            )
            y_nhwc = y_nhwc + bias_b

        # NHWC -> NCHW back
        return ops.transpose(y_nhwc, (0, 3, 1, 2))
        ### END YOUR SOLUTION