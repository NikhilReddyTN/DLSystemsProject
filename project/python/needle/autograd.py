"""Core data structures."""
import os
import needle
from .backend_numpy import Device, all_devices
from typing import List, Optional, NamedTuple, Tuple, Union, Dict
from collections import namedtuple
import numpy

from needle import init

# needle version
LAZY_MODE = False
TENSOR_COUNTER = 0
ENABLE_EW_FUSION = os.environ.get("NEEDLE_EW_FUSION", "1") != "0"

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api
NDArray = numpy.ndarray

from .backend_selection import array_api, NDArray, default_device, cpu

EW_FUSION_ACTIVE = ENABLE_EW_FUSION and hasattr(array_api, "fused_elementwise")

# Keep op codes in sync with backend implementations.
FUSED_OP_ADD_SCALAR = 1
FUSED_OP_MUL_SCALAR = 2
FUSED_OP_DIV_SCALAR = 3
FUSED_OP_POWER_SCALAR = 4
FUSED_OP_NEGATE = 5
FUSED_OP_EXP = 6
FUSED_OP_LOG = 7
FUSED_OP_TANH = 8
FUSED_OP_RELU = 9

_FUSION_REGISTRY = None

class Op:
    """Operator definition."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """Op class specialized to output tensors, will be alternate subclasses for other structures"""

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad
        self._num_users = 0
        for inp in inputs:
            if hasattr(inp, "_num_users"):
                inp._num_users += 1

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


### Not needed in HW1
class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "needle.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else default_device()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        skip_realize = (
            EW_FUSION_ACTIVE and tensor.requires_grad and _is_fusible_op(op)
        )
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            if not skip_realize:
                tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        # numpy array always sits on cpu
        if array_api is numpy:
            return cpu()
        return data.device

    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return numpy.array(data)
        return data.numpy()

    def realize_cached_data(self):
        if self.cached_data is not None:
            return self.cached_data
        if EW_FUSION_ACTIVE and _attempt_elementwise_fusion(self):
            return self.cached_data
        return super().realize_cached_data()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)

    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def __neg__(self):
        return needle.ops.Negate()(self)

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)




    __radd__ = __add__
    __rmul__ = __mul__

def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    for node in reverse_topo_order:
      if not node.requires_grad:
        continue

      v_i= sum_node_list(node_to_output_grads_list[node])
      node.grad=v_i

      if not node.is_leaf():
        input_grads = node.op.gradient_as_tuple(v_i, node)
        for k,in_node in enumerate(node.inputs):
          if in_node.requires_grad:
            if in_node not in node_to_output_grads_list:
              node_to_output_grads_list[in_node] = []
            node_to_output_grads_list[in_node].append(input_grads[k])
    ### END YOUR SOLUTION


def _get_fusion_registry():
    global _FUSION_REGISTRY
    if _FUSION_REGISTRY is None:
        import needle.ops as ops

        _FUSION_REGISTRY = {
            ops.AddScalar: (FUSED_OP_ADD_SCALAR, lambda op: float(op.scalar)),
            ops.MulScalar: (FUSED_OP_MUL_SCALAR, lambda op: float(op.scalar)),
            ops.DivScalar: (FUSED_OP_DIV_SCALAR, lambda op: float(op.scalar)),
            ops.PowerScalar: (FUSED_OP_POWER_SCALAR, lambda op: float(op.scalar)),
            ops.Negate: (FUSED_OP_NEGATE, lambda op: 0.0),
            ops.Exp: (FUSED_OP_EXP, lambda op: 0.0),
            ops.Log: (FUSED_OP_LOG, lambda op: 0.0),
            ops.Tanh: (FUSED_OP_TANH, lambda op: 0.0),
            ops.ReLU: (FUSED_OP_RELU, lambda op: 0.0),
        }
    return _FUSION_REGISTRY


def _is_fusible_op(op: Optional[Op]) -> bool:
    if op is None:
        return False
    registry = _get_fusion_registry()
    return op.__class__ in registry


def _fusion_metadata(op: Op) -> Tuple[int, float]:
    code, extractor = _get_fusion_registry()[op.__class__]
    return code, extractor(op)


def _attempt_elementwise_fusion(node: "Tensor") -> bool:
    chain_info = _collect_fusible_chain(node)
    if chain_info is None:
        return False
    base, chain = chain_info
    base_data = base.realize_cached_data()
    if base_data is None:
        return False
    if not hasattr(base_data, "is_compact"):
        return False
    base_array = base_data if base_data.is_compact() else base_data.compact()
    device = base_array.device
    outputs = []
    codes: List[int] = []
    params: List[float] = []
    for current in chain:
        out_arr = array_api.empty(base_array.shape, device=device)
        current.cached_data = out_arr
        code, param = _fusion_metadata(current.op)
        outputs.append(out_arr)
        codes.append(code)
        params.append(param)
    array_api.fused_elementwise(base_array, outputs, codes, params)
    return True


def _collect_fusible_chain(node: "Tensor") -> Optional[Tuple["Tensor", List["Tensor"]]]:
    if not _is_fusible_node(node):
        return None
    chain: List[Tensor] = []
    current: Tensor = node
    base: Optional[Tensor] = None
    while True:
        chain.append(current)
        parent = current.inputs[0]
        if not isinstance(parent, Tensor):
            base = None
            break
        if not _can_fuse_parent(parent):
            base = parent
            break
        current = parent
    if base is None or not isinstance(base, Tensor) or len(chain) < 2:
        return None
    chain.reverse()
    return base, chain


def _is_fusible_node(node: "Tensor") -> bool:
    return (
        isinstance(node, Tensor)
        and node.cached_data is None
        and node.op is not None
        and len(node.inputs) == 1
        and _is_fusible_op(node.op)
    )


def _can_fuse_parent(parent: "Tensor") -> bool:
    return (
        isinstance(parent, Tensor)
        and parent.cached_data is None
        and parent.op is not None
        and getattr(parent, "_num_users", 0) == 1
        and len(parent.inputs) == 1
        and _is_fusible_op(parent.op)
    )


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    visited = set()
    topo_order = []
    for node in node_list:
      topo_sort_dfs(node,visited,topo_order)
    return topo_order
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    if node in visited:
      return
    
    visited.add(node)
    for n in node.inputs:
      topo_sort_dfs(n,visited,topo_order)

    topo_order.append(node)
    ### END YOUR SOLUTION


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)
