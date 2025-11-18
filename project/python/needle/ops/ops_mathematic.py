"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b=node.inputs
        return out_grad * b * power(a,b-1), out_grad * power(a,b) * log(a)
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a=node.inputs[0]
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b=node.inputs
        return out_grad/b, -out_grad*a/power_scalar(b, 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        nd = len(a.shape)
        if self.axes is None:
            perm = list(range(nd))
            perm[-1], perm[-2] = perm[-2], perm[-1]
        else:
            if len(self.axes) == 2:
                i, j = self.axes
                perm = list(range(nd))
                if i < 0: i += nd
                if j < 0: j += nd
                perm[i], perm[j] = perm[j], perm[i]
            else:
                assert len(self.axes) == nd, "full permutation must match tensor rank"
                perm = list(self.axes)
        return a.permute(tuple(perm))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        nd = len(out_grad.shape)
        if self.axes is None:
            inv_axes = (-2, -1) 
            return (transpose(out_grad, inv_axes),)
        if len(self.axes) == 2:
            return (transpose(out_grad, self.axes),)
        p = list(self.axes)
        p = [ax if ax >= 0 else ax + nd for ax in p]
        inv = [0] * nd
        for i, pi in enumerate(p):
            inv[pi] = i
        return (transpose(out_grad, tuple(inv)),)
        ### END YOUR SOLUTION

def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a=node.inputs[0]
        return reshape(out_grad,a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a=node.inputs[0]
        grad=out_grad

        new_dims = len(self.shape) - len(a.shape)
        for i in range(new_dims):
            grad = summation(grad, axes=(0,))
      
        for i, (input_dim, output_dim) in enumerate(zip(a.shape, self.shape)):
            if input_dim == 1 and output_dim > 1:
                grad = summation(grad, axes=(i,))
                grad = reshape(grad, grad.shape[:i] + (1,) + grad.shape[i:])
        
        return grad
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a,self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        in_shape = a.shape
        if self.axes is None:
            tgt = tuple(1 for _ in in_shape)
        else:
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            axset = set(axes)
            tgt = tuple(1 if i in axset else s for i, s in enumerate(in_shape))
        grad = reshape(out_grad, tgt)
        return (broadcast_to(grad, in_shape),)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        dA = matmul(out_grad, transpose(b, (1, 0)))
        dB = matmul(transpose(a, (1, 0)), out_grad)
        return dA, dB
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a=node.inputs[0]
        return out_grad/a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a=node.inputs[0]
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a,0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        mask_nd = (a.realize_cached_data() > 0)
        mask = Tensor(mask_nd, device=a.device)   # constant tensor, no grad
        return (out_grad * mask,)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        t = tanh(a)
        return (out_grad * add_scalar(mul_scalar(multiply(t, t), -1.0), 1.0),)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        arrays = [x for x in args]
        base = arrays[0]
        base_shape = base.shape
        n = len(arrays)

        axis = self.axis if self.axis >= 0 else self.axis + (len(base_shape) + 1)
        out_shape = base_shape[:axis] + (n,) + base_shape[axis:]
        out = array_api.empty(out_shape, device=base.device)
        view_shape = base_shape[:axis] + (1,) + base_shape[axis:]
        sl = [slice(None)] * len(out_shape)
        for i, arr in enumerate(arrays):
            sl[axis] = slice(i, i + 1)
            out[tuple(sl)] = arr.compact().reshape(view_shape)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        axis = self.axis if self.axis >= 0 else self.axis + A.ndim
        n = A.shape[axis]
        parts = []
        sl = [slice(None)] * A.ndim
        for i in range(n):
            sl[axis] = slice(i, i + 1)
            part = A[tuple(sl)]
            squeezed = part.compact().reshape(part.shape[:axis] + part.shape[axis+1:])
            parts.append(squeezed)
        return tuple(parts)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (stack([out_grad[i] for i in range(len(out_grad))], self.axis),)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(a.shape)))
        elif not isinstance(axes, tuple):
            axes = (axes,)
        return array_api.flip(a, axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        axes = self.axes
        if axes is None:
            (a,) = node.inputs
            axes = tuple(range(len(a.shape)))
        elif not isinstance(axes, tuple):
            axes = (axes,)
        return (flip(out_grad, axes),)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        d = self.dilation
        if d == 0:
            return a
        ndim = len(a.shape)
        raw_axes = self.axes
        if raw_axes is None:
            axes = tuple(range(ndim))
        else:
            if not isinstance(raw_axes, tuple):
                raw_axes = (raw_axes,)
            norm_axes = []
            for ax in raw_axes:
                if -ndim <= ax < ndim:
                    if ax < 0:
                        ax += ndim
                    norm_axes.append(ax)
            axes = tuple(norm_axes)
        if not axes:
            return a
        old_shape = a.shape
        new_shape = list(old_shape)
        for ax in axes:
            new_shape[ax] = old_shape[ax] * (d + 1)

        out = array_api.full(tuple(new_shape), 0.0, device=a.device)
        sl = []
        for i in range(ndim):
            if i in axes:
                sl.append(slice(0, new_shape[i], d + 1))
            else:
                sl.append(slice(0, new_shape[i], 1))
        out[tuple(sl)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (undilate(out_grad, self.axes, self.dilation),)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        d = self.dilation
        if d == 0:
            return a
        ndim = len(a.shape)
        axes = self.axes
        if axes is None:
            axes = tuple(range(ndim))
        elif not isinstance(axes, tuple):
            axes = (axes,)
        axes = tuple(ax if ax >= 0 else ax + ndim for ax in axes)
        in_shape = a.shape
        sl_in = []
        for i in range(ndim):
            if i in axes:
                sl_in.append(slice(0, in_shape[i], d + 1))
            else:
                sl_in.append(slice(0, in_shape[i], 1))

        return a[tuple(sl_in)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (dilate(out_grad, self.axes, self.dilation),)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        stride = self.stride
        pad = self.padding
        N, H, W, C_in = A.shape
        K, K2, C_in_w, C_out = B.shape
        assert K == K2
        assert C_in == C_in_w
        if pad > 0:
            pad_axes = ((0, 0), (pad, pad), (pad, pad), (0, 0))
            A_padded = A.pad(pad_axes)
        else:
            A_padded = A

        Np, Hp, Wp, Cp = A_padded.shape
        assert Cp == C_in
        out_H = (Hp - K) // stride + 1
        out_W = (Wp - K) // stride + 1
        Ns, Hs, Ws, Cs = A_padded.strides  
        A_im2col = A_padded.as_strided(
            shape=(N, out_H, out_W, K, K, C_in),
            strides=(Ns, Hs * stride, Ws * stride, Hs, Ws, Cs),
        )
        inner_dim = K * K * C_in
        A_mat = A_im2col.compact().reshape((N * out_H * out_W, inner_dim))
        W_mat = B.compact().reshape((inner_dim, C_out))
        out = A_mat @ W_mat
        return out.reshape((N, out_H, out_W, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        s, p = self.stride, self.padding
        K = B.shape[0]
        og = out_grad
        if s > 1:
            og = dilate(og, axes=(1, 2), dilation=s - 1)
        W_flip = flip(B, axes=(0, 1))             
        W_T = transpose(W_flip, (0, 1, 3, 2)) 

        dA = conv(og, W_T, stride=1, padding=K - 1 - p)

        og_for_w = out_grad
        if s > 1:
            og_for_w = dilate(og_for_w, axes=(1, 2), dilation=s - 1)
        X_prime = transpose(A, (3, 1, 2, 0))
        G_prime = transpose(og_for_w, (1, 2, 0, 3))
        dB_tmp = conv(X_prime, G_prime, stride=1, padding=p)
        dB = transpose(dB_tmp, (1, 2, 0, 3))
        return dA, dB
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


