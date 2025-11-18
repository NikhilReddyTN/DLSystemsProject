"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
          init.kaiming_uniform(
            fan_in=in_features,
            fan_out=out_features,
            device=device,
            dtype=dtype
          )
        )
        if bias is True:
          self.bias=Parameter(
            init.kaiming_uniform(
              fan_in=out_features,
              fan_out=1,
              device=device,
              dtype=dtype
            ).reshape((1,out_features))
          )
        else:
          self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias is not None:
          out = out + self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch = X.shape[0]
        features = int(np.prod(X.shape[1:]))
        return X.reshape((batch, features))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
          x=module.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        z_y = (logits * y_one_hot).sum(axes=(1,))
        return (log_sum_exp - z_y).sum() / logits.shape[0]
        ### END YOUR SOLUTION

class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N = x.shape[0]
        if self.training:
          batch_mean = x.sum(axes=(0,)) / N
          mean_b = batch_mean.reshape((1, self.dim)).broadcast_to(x.shape)
          x_centered = x - mean_b
          batch_var = (x_centered ** 2).sum(axes=(0,)) / N
          rm = self.running_mean.cached_data
          rv = self.running_var.cached_data
          bm = batch_mean.cached_data
          bv = batch_var.cached_data
          m = self.momentum
          self.running_mean.cached_data = (1.0 - m) * rm + m * bm
          self.running_var.cached_data  = (1.0 - m) * rv + m * bv
          var_b = batch_var.reshape((1, self.dim)).broadcast_to(x.shape)
        else:
          mean_b = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
          var_b  = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)
          x_centered = x - mean_b

        x_hat = x_centered / ((var_b + self.eps) ** 0.5)
        gamma = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        beta  = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return gamma * x_hat + beta
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))

class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        mean = x.sum(axes=(1,)) / self.dim
        mean = mean.reshape((batch_size, 1)).broadcast_to(x.shape)
        x_centered = x - mean
        var = (x_centered ** 2).sum(axes=(1,)) / self.dim 
        var = var.reshape((batch_size, 1)).broadcast_to(x.shape)
        std = (var + self.eps) ** 0.5
        x_norm = x_centered / std
        weight_broadcast = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias_broadcast = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return weight_broadcast * x_norm + bias_broadcast
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training or self.p == 0.0:
          return x
        keep_prob = 1.0 - self.p
        mask = init.randb(*x.shape, p=keep_prob, device=x.device, dtype=x.dtype)
        return (x * mask) / keep_prob
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
