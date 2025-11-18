"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
        self.t = 0 

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        beta = self.momentum
        for p in self.params:
          if p.grad is None:
            continue
          g = p.grad.cached_data
          if self.weight_decay != 0.0:
            g = g + self.weight_decay * p.cached_data  # L2
          if p not in self.u:
            self.u[p] = (1.0 - beta) * g
          else:
            self.u[p] = beta * self.u[p] + (1.0 - beta) * g
          p.cached_data = p.cached_data - self.lr * self.u[p]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps

        for p in self.params:
          if p.grad is None:
              continue

          g = p.grad.cached_data
          if self.weight_decay != 0.0:
              g = g + self.weight_decay * p.cached_data  # L2 on parameters

          m_prev = self.m.get(p)
          v_prev = self.v.get(p)

          if m_prev is None:
              m = (1.0 - b1) * g
              v = (1.0 - b2) * (g * g)
          else:
              m = b1 * m_prev + (1.0 - b1) * g
              v = b2 * v_prev + (1.0 - b2) * (g * g)

          self.m[p] = m
          self.v[p] = v

          m_hat = m / (1.0 - (b1 ** self.t))
          v_hat = v / (1.0 - (b2 ** self.t))
          denom = (v_hat ** 0.5) + eps              # or: array_api.sqrt(v_hat) + eps
          step = self.lr * (m_hat / denom)
          p.cached_data = p.cached_data - step
        ### END YOUR SOLUTION
