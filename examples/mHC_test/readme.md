# Unofficial Rough Implementation of mHC Kernel Fusion

This directory contains unofficial, experimental implementations of operators related to mHC (micro Head Cache) kernel fusion. The implementations are based on TileLang and include several example scripts for validating functional correctness and performing simple benchmarking.

## Directory Description

- `sinkhorn_knopp.py` — Sinkhorn–Knopp normalization kernel (forward pass) implemented in TileLang, including a reference implementation, unit tests, and a benchmark script.
- `sinkhorn_knopp_bwd.py` — Corresponding backward/gradient implementation (for verifying backpropagation correctness).
- `small_coefficients.py` — TileLang kernel (forward pass) that fuses operations related to small-coefficients, along with a reference implementation and benchmark.
- `small_coefficients_bwd.py` — Backward implementation for small-coefficients.

## Quick Start

First, ensure the Python environment in the repository root meets the project dependencies (recommended to use the repository-provided `requirements*.txt` or the development container). At minimum, you need:

- Python 3.8+
- `torch` (CUDA version depending on the target device)
- `tilelang` installed and importable (within the repository or in the environment)

Run the example scripts from the repository root directory:

```bash
python tilelang/examples/mHC_test/sinkhorn_knopp.py
python tilelang/examples/mHC_test/small_coefficients.py
```

These scripts will:
- Perform a numerical comparison between the reference (pure PyTorch) implementation and the TileLang JIT kernel (using `torch.testing.assert_close`).
- Run a benchmark and print the latency (ms) after passing the correctness check.

If you only want to run a quick small-scale verification, you can modify the arguments of the `main(...)` call at the end of each script to a smaller `num_tokens`.

## Developer Notes

- The example implementations rely on TileLang's JIT/`@tl.jit` capability and use `tilelang.profiler.do_bench` for simple benchmarking. If you need to debug or view the generated kernel source code, you can uncomment the print statements for `get_kernel_source()` in the examples.

## Performance Summary

- Test Device: NVIDIA H20, Test Image: nvcr.io/nvidia/pytorch:25.11-py3

| Num Tokens | Kernel Type | Forward Latency (ms) | Backward Latency (ms) | Reference Forward (ms) | Reference Backward (ms) | Forward Speedup | Backward Speedup |
|------------|-------------|----------------------|------------------------|------------------------|-------------------------|----------------|------------------|
| 1          | small_coefficients | 0.004733 | 0.006605 | 0.050119 | 0.383099 | 10.59× | 58.01× |
| 16         |   | 0.004909 | 0.026274 | 0.048304 | 0.393257 | 9.84× | 14.97× |
| 1024       |   | 0.005451 | 0.047929 | 0.048580 | 0.398574 | 8.91× | 8.31× |
| 16384      |   | 0.012375 | 0.064353 | 0.048225 | 0.407946 | 3.90× | 6.34× |
| 65536      |   | 0.033858 | 0.159759 | 0.064264 | 0.421223 | 1.90× | 2.64× |
| 131072     |   | 0.056596 | 0.253524 | 0.090621 | 0.418222 | 1.60× | 1.65× |
|------------|-------------|----------------------|------------------------|------------------------|-------------------------|----------------|------------------|
| 1          | sinkhorn_knopp | 0.038322 | 0.053058 | 0.342459 | 1.471663 | 8.94× | 27.73× |
| 16         |   | 0.038441 | 0.053142 | 0.341821 | 1.481363 | 8.89× | 27.87× |
| 1024       |   | 0.033662 | 0.046940 | 0.340513 | 1.475025 | 10.12× | 31.43× |
| 16384      |   | 0.038712 | 0.061251 | 0.342180 | 1.466891 | 8.84× | 23.95× |
| 65536      |   | 0.060909 | 0.105609 | 0.607559 | 1.697532 | 9.98× | 16.08× |
| 131072     |   | 0.106048 | 0.191265 | 1.013797 | 2.987684 | 9.56× | 15.62× |

## Contribution & License

This implementation is unofficial experimental code. Improvements and feedback are welcome. Please follow the contribution guidelines and license information in the repository root directory.

---

File location: [tilelang/examples/mHC_test/readme.md](tilelang/examples/mHC_test/readme.md)