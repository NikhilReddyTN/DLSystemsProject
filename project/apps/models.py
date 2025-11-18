import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class ConvBN(nn.Module):
    def __init__(self, cin, cout, k, s, device=None, dtype="float32"):
        super().__init__()
        self.conv = nn.Conv(cin, cout, k, stride=s, bias=True, device=device, dtype=dtype)
        self.bn   = nn.BatchNorm2d(cout, device=device, dtype=dtype)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.c1 = ConvBN(3,   16, 7, 4, device=device, dtype=dtype)
        self.c2 = ConvBN(16,  32, 3, 2, device=device, dtype=dtype)
        self.res1 = nn.Residual(
            nn.Sequential(
                ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
                ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
            )
        )
        self.c3 = ConvBN(32,  64, 3, 2, device=device, dtype=dtype)
        self.c4 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype)
        self.res2 = nn.Residual(
            nn.Sequential(
                ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
                ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
            )
        )
        self.flatten = nn.Flatten()                   # avoids (-1) reshape
        self.fc1     = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(128, 10,  device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.c1(x)
        x = self.c2(x)
        x = self.res1(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.res2(x)
        x = self.flatten(x)      # N,128 (since spatial is 1x1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.seq_model_type = seq_model
        self.device = device
        self.dtype = dtype

        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model.lower() == 'rnn':
            self.seq = nn.RNN(embedding_size, hidden_size, num_layers=num_layers,
                              bias=True, device=device, dtype=dtype)
        elif seq_model.lower() == 'lstm':
            self.seq = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers,
                               bias=True, device=device, dtype=dtype)
        else:
            raise ValueError("seq_model must be 'rnn' or 'lstm'")
        self.fc = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        emb = self.embedding(x)  # (S, B, E)
        if h is None:
            seq_out, h_new = self.seq(emb, None)
        else:
            seq_out, h_new = self.seq(emb, h)  # (S, B, H), h_new
        S, B, H = seq_out.shape
        seq_out_flat = ndl.ops.reshape(seq_out, (S * B, H))          # (S*B, H)
        logits = self.fc(seq_out_flat)                               # (S*B, V)
        return logits, h_new
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
