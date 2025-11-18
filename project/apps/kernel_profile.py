import argparse
import sys
from typing import Iterable, Tuple

import numpy as np

sys.path.append("python")

import needle as ndl
import needle.nn as nn


def _get_device(name: str):
    name = name.lower()
    if name == "cuda":
        dev = ndl.cuda()
        if dev.enabled():
            return dev
        raise RuntimeError("CUDA backend requested but not available.")
    if name == "cpu_numpy":
        return ndl.cpu_numpy()
    return ndl.cpu()


def _iterate_batches(N: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        yield start, end


def _run_epoch(model, optimizer, loss_fn, data, labels, batch_size, device, preprocess=None):
    model.train()
    N = data.shape[0]
    for start, end in _iterate_batches(N, batch_size):
        batch = data[start:end]
        target = labels[start:end]
        if preprocess is not None:
            batch = preprocess(batch)
        x = ndl.Tensor(batch, device=device)
        y = ndl.Tensor(target, device=device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.reset_grad()
        loss.backward()
        optimizer.step()


def profile_mlp(args, device):
    np.random.seed(0)
    samples = 2048
    in_dim = 128
    num_classes = 10
    data = np.random.randn(samples, in_dim).astype("float32")
    labels = np.random.randint(0, num_classes, size=(samples,)).astype("float32")
    model = nn.Sequential(
        nn.Linear(in_dim, 256, device=device),
        nn.ReLU(),
        nn.Linear(256, 128, device=device),
        nn.ReLU(),
        nn.Linear(128, num_classes, device=device),
    )
    opt = ndl.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SoftmaxLoss()
    for epoch in range(args.epochs):
        ndl.reset_kernel_profiler()
        _run_epoch(model, opt, loss_fn, data, labels, args.batch_size, device)
        total = ndl.get_total_kernel_count(device.name)
        print(f"[MLP] Epoch {epoch + 1}: {total} kernel launches")


def profile_cnn(args, device):
    np.random.seed(1)
    samples = 1024
    num_classes = 10
    data = np.random.randn(samples, 3, 32, 32).astype("float32")
    labels = np.random.randint(0, num_classes, size=(samples,)).astype("float32")
    model = nn.Sequential(
        nn.Conv(3, 16, 3, device=device),
        nn.ReLU(),
        nn.Conv(16, 32, 3, device=device),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 32 * 32, 128, device=device),
        nn.ReLU(),
        nn.Linear(128, num_classes, device=device),
    )
    opt = ndl.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.SoftmaxLoss()

    def preprocess(batch):
        return batch

    for epoch in range(args.epochs):
        ndl.reset_kernel_profiler()
        _run_epoch(model, opt, loss_fn, data, labels, args.batch_size, device, preprocess=preprocess)
        total = ndl.get_total_kernel_count(device.name)
        print(f"[CNN] Epoch {epoch + 1}: {total} kernel launches")


def profile_rnn(args, device):
    np.random.seed(2)
    samples = 512
    seq_len = 20
    input_dim = 16
    num_classes = 8
    data = np.random.randn(samples, seq_len, input_dim).astype("float32")
    labels = np.random.randint(0, num_classes, size=(samples,)).astype("float32")

    class SimpleRNN(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.rnn = nn.RNN(input_dim, 64, num_layers=1, device=device)
            self.classifier = nn.Linear(64, num_classes, device=device)

        def forward(self, x):
            out, _ = self.rnn(x)
            last = out[-1]
            return self.classifier(last)

    def preprocess(batch):
        return batch.transpose(1, 0, 2)

    model = SimpleRNN(device)
    opt = ndl.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SoftmaxLoss()
    for epoch in range(args.epochs):
        ndl.reset_kernel_profiler()
        _run_epoch(model, opt, loss_fn, data, labels, args.batch_size, device, preprocess=preprocess)
        total = ndl.get_total_kernel_count(device.name)
        print(f"[RNN] Epoch {epoch + 1}: {total} kernel launches")


def main():
    parser = argparse.ArgumentParser(description="Profile kernel launches per epoch.")
    parser.add_argument("--device", default="cpu", help="cpu, cpu_numpy, or cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--models", nargs="+", default=["mlp", "cnn", "rnn"], choices=["mlp", "cnn", "rnn"]
    )
    args = parser.parse_args()
    device = _get_device(args.device)
    ndl.enable_kernel_profiler()
    try:
        if "mlp" in args.models:
            profile_mlp(args, device)
        if "cnn" in args.models:
            profile_cnn(args, device)
        if "rnn" in args.models:
            profile_rnn(args, device)
    finally:
        ndl.disable_kernel_profiler()


if __name__ == "__main__":
    main()
