"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (ops.tanh(x * 0.5) + 1.0) * 0.5
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        bound = 1.0 / hidden_size**0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound,
                                        device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound,
                                        device=device, dtype=dtype))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound,
                                              device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound,
                                              device=device, dtype=dtype))
        else:
            self.bias_ih = None
            self.bias_hh = None
        if nonlinearity == "tanh":
            self._act = ops.tanh
        elif nonlinearity == "relu":
            self._act = ops.relu
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=self.dtype)

        pre = X @ self.W_ih + h @ self.W_hh

        if self.bias_ih is not None:
            B = X.shape[0]
            b_ih = ops.broadcast_to(self.bias_ih, (B, self.hidden_size))
            b_hh = ops.broadcast_to(self.bias_hh, (B, self.hidden_size))
            pre = pre + b_ih + b_hh

        return self._act(pre)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        cells = []
        for layer in range(num_layers):
            in_features = input_size if layer == 0 else hidden_size
            cell = RNNCell(
                in_features, hidden_size, bias=bias, nonlinearity=nonlinearity,
                device=device, dtype=dtype
            )
            setattr(self, f"rnn_cell_{layer}", cell)
            cells.append(cell)
        self.rnn_cells: List[RNNCell] = cells
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len = X.shape[0]
        bs = X.shape[1]

        if h0 is None:
            # IMPORTANT: pass dims separately, not a single tuple
            hs = [init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
                  for _ in range(self.num_layers)]
        else:
            h0_slices = ops.split(h0, axis=0)  # list of (B, H)
            hs = [h0_slices[i] for i in range(self.num_layers)]

        outputs = []
        x_ts = ops.split(X, axis=0)  # list of (B, I)
        for t in range(seq_len):
            x_t = x_ts[t]
            for layer in range(self.num_layers):
                cell = self.rnn_cells[layer]
                x_t = cell(x_t, hs[layer])
                hs[layer] = x_t
            outputs.append(x_t)

        output = ops.stack(outputs, axis=0)   # (seq_len, B, H)
        h_n = ops.stack(hs, axis=0)           # (num_layers, B, H)
        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        bound = 1.0 / (hidden_size ** 0.5)
        self.W_ih = Parameter(
            init.rand(input_size, 4 * hidden_size, low=-bound, high=bound,
                      device=device, dtype=dtype)
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound,
                      device=device, dtype=dtype)
        )
        if bias:
            self.bias_ih = Parameter(
                init.rand(4 * hidden_size, low=-bound, high=bound,
                          device=device, dtype=dtype)
            )
            self.bias_hh = Parameter(
                init.rand(4 * hidden_size, low=-bound, high=bound,
                          device=device, dtype=dtype)
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        self._sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        B = X.shape[0]
        H = self.hidden_size
        if h is None:
            h_t_1 = init.zeros(B, H, device=self.device, dtype=self.dtype)
            c_t_1 = init.zeros(B, H, device=self.device, dtype=self.dtype)
        else:
            h_t_1, c_t_1 = h
        pre = X @ self.W_ih + h_t_1 @ self.W_hh
        if self.bias_ih is not None:
            b_ih = ops.broadcast_to(self.bias_ih, (B, 4 * H))
            b_hh = ops.broadcast_to(self.bias_hh, (B, 4 * H))
            pre = pre + b_ih + b_hh
        pre_3d = ops.reshape(pre, (B, 4, H))
        chunks = ops.split(pre_3d, axis=1)  
        i_lin = ops.reshape(chunks[0], (B, H))
        f_lin = ops.reshape(chunks[1], (B, H))
        g_lin = ops.reshape(chunks[2], (B, H))
        o_lin = ops.reshape(chunks[3], (B, H))
        i = self._sigmoid(i_lin)
        f = self._sigmoid(f_lin)
        g = ops.tanh(g_lin)
        o = self._sigmoid(o_lin)
        c_t = f * c_t_1 + i * g
        h_t = o * ops.tanh(c_t)
        return h_t, c_t
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        cells = []
        for layer in range(num_layers):
            in_features = input_size if layer == 0 else hidden_size
            cell = LSTMCell(in_features, hidden_size, bias=bias,
                            device=device, dtype=dtype)
            setattr(self, f"lstm_cell_{layer}", cell)
            cells.append(cell)
        self.lstm_cells: List[LSTMCell] = cells
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len = X.shape[0]
        B = X.shape[1]
        H = self.hidden_size
        if h is None:
            hs = [init.zeros(B, H, device=self.device, dtype=self.dtype)
                  for _ in range(self.num_layers)]
            cs = [init.zeros(B, H, device=self.device, dtype=self.dtype)
                  for _ in range(self.num_layers)]
        else:
            h0, c0 = h
            h_slices = ops.split(h0, axis=0)  # list of (B, H)
            c_slices = ops.split(c0, axis=0)  # list of (B, H)
            hs = [h_slices[i] for i in range(self.num_layers)]
            cs = [c_slices[i] for i in range(self.num_layers)]
        outputs = []
        x_ts = ops.split(X, axis=0)  # list of length seq_len, each (B, I)
        for t in range(seq_len):
            x_t = x_ts[t]
            for layer in range(self.num_layers):
                cell = self.lstm_cells[layer]
                h_new, c_new = cell(x_t, (hs[layer], cs[layer]))
                hs[layer], cs[layer] = h_new, c_new
                x_t = h_new  # feed to next layer
            outputs.append(x_t)
        output = ops.stack(outputs, axis=0)      # (seq_len, B, H)
        h_n = ops.stack(hs, axis=0)              # (num_layers, B, H)
        c_n = ops.stack(cs, axis=0)              # (num_layers, B, H)
        return output, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        n = seq_len * bs
        x_flat = ops.reshape(x, (n,))
        idx_np = x_flat.cached_data.numpy().astype(np.int64)
        V = self.num_embeddings
        oh_np = np.zeros((n, V), dtype=np.float32)
        oh_np[np.arange(n), idx_np] = 1.0
        oh = Tensor(oh_np, device=self.device, dtype=self.dtype)   # no grad
        emb_flat = oh @ self.weight                                # (n, E)
        return ops.reshape(emb_flat, (seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION