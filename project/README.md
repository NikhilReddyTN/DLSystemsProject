# Homework 4
Public repository and stub/testing code for Homework 4 of 10-714.

## Elementwise Chain Fusion (EW-Fuse)
This repository now supports automatic fusion of unary elementwise operator chains. When supported by the backend (Needle NDArray CPU/CUDA backends), runs of operations such as `add_scalar`, `mul_scalar`, `exp`, `tanh`, or `relu` that form a single-use chain are executed with a single custom kernel launch. Fusion is enabled by default and can be toggled by setting the environment variable `NEEDLE_EW_FUSION=0` before running any Needle program.

## Kernel launch profiling
Kernel launches can be tracked via the profiler utilities exported from `needle`. Enable counting with `ndl.enable_kernel_profiler()`, reset counts with `ndl.reset_kernel_profiler()`, and query totals via `ndl.get_total_kernel_count(device_name)`. A helper script is available for profiling synthetic MLP/CNN/RNN training workloads:

```bash
python apps/kernel_profile.py --device cuda --epochs 2 --models mlp cnn
```

This prints the number of backend kernel launches per epoch for the selected models. Set `--device cpu` (default) if CUDA is unavailable.
