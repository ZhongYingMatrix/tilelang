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

 Kernel Type: `small_coefficients`

| Num Tokens | Forward Latency (ms) | Reference Forward (ms) | Forward Speedup | Backward Latency (ms) | Reference Backward (ms) | Backward Speedup |
|------------|----------------------|------------------------|------------------|------------------------|--------------------------|-------------------|
| 1          | 0.0047               | 0.0518                 | 10.92×           | 0.0081                 | 0.3656                   | 45.00×            |
| 16         | 0.0049               | 0.0506                 | 10.32×           | 0.0194                 | 0.3763                   | 19.42×            |
| 1k         | 0.0051               | 0.0508                 | 9.92×            | 0.0416                 | 0.3813                   | 9.15×             |
| 16k        | 0.0064               | 0.0504                 | 7.83×            | 0.0455                 | 0.3883                   | 8.54×             |
| 64k        | 0.0094               | 0.0658                 | 6.98×            | 0.1149                 | 0.4000                   | 3.48×             |
| 128k       | 0.0133               | 0.0945                 | 7.12×            | 0.2130                 | 0.4007                   | 1.88×             |

 Kernel Type: `sinkhorn_knopp`
| Num Tokens | Forward Latency (ms) | Reference Forward (ms) | Forward Speedup | Backward Latency (ms) | Reference Backward (ms) | Backward Speedup |
|------------|----------------------|------------------------|------------------|------------------------|--------------------------|-------------------|
| 1          | 0.0389               | 0.3856                 | 9.91×            | 0.0532                 | 1.4966                   | 28.13×            |
| 16         | 0.0392               | 0.3916                 | 9.99×            | 0.0533                 | 1.5136                   | 28.39×            |
| 1k         | 0.0342               | 0.3870                 | 11.31×           | 0.0469                 | 1.5135                   | 32.27×            |
| 16k        | 0.0388               | 0.3911                 | 10.07×           | 0.0604                 | 1.4919                   | 24.70×            |
| 64k        | 0.0609               | 0.6019                 | 9.88×            | 0.1053                 | 1.7050                   | 16.19×            |
| 128k       | 0.1072               | 1.0111                 | 9.43×            | 0.1905                 | 2.9724                   | 15.60×            |



## Contribution & License

This implementation is unofficial experimental code. Improvements and feedback are welcome. Please follow the contribution guidelines and license information in the repository root directory.

---

File location: [tilelang/examples/mHC_test/readme.md](tilelang/examples/mHC_test/readme.md)