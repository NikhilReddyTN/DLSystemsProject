from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            axis = None
        elif isinstance(self.axes, (tuple, list)):
            assert len(self.axes) == 1, "Only single-axis reduction supported"
            axis = self.axes[0]
        else:
            axis = self.axes
        Z_max = Z.max(axis=axis, keepdims=True) 
        if axis is None:
            tgt = tuple(1 for _ in Z.shape)
        else:
            tgt = tuple(1 if i == (axis if axis >= 0 else axis + Z.ndim) else s
                        for i, s in enumerate(Z.shape))
        Z_max_b = array_api.broadcast_to(array_api.reshape(Z_max, tgt), Z.shape)
        shifted = Z - Z_max_b
        s = array_api.sum(array_api.exp(shifted), axis=axis) 
        out = array_api.log(s)
        if axis is None:
            keep_shape = out.shape 
        else:
            ax = axis if axis >= 0 else axis + Z.ndim
            keep_shape = tuple(Z.shape[i] for i in range(Z.ndim) if i != ax)

        return out + array_api.reshape(Z_max, keep_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (Z,) = node.inputs
        if self.axes is None:
            axis = None
        elif isinstance(self.axes, (tuple, list)):
            axis = self.axes[0]
        else:
            axis = self.axes
        lse = node
        if axis is None:
            tgt = tuple(1 for _ in Z.shape)
        else:
            ax = axis if axis >= 0 else axis + len(Z.shape)
            tgt = tuple(1 if i == ax else Z.shape[i] for i in range(len(Z.shape)))
        lse_b = broadcast_to(reshape(lse, tgt), Z.shape)
        og = broadcast_to(reshape(out_grad, tgt), Z.shape)
        sm = exp(Z - lse_b)
        return (og * sm,)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)