# Unofficial Rough Implementation of mHC Kernel Fusion

This directory contains unofficial, experimental implementations of operators related to mHC kernel fusion. The implementations are based on TileLang and include several example scripts for validating functional correctness and performing simple benchmarking.

## Directory Description

- `sinkhorn_knopp.py` — Sinkhorn–Knopp normalization kernel (forward pass) implemented in TileLang, including a reference implementation, unit tests, and a benchmark script.
- `sinkhorn_knopp_bwd.py` — Corresponding backward/gradient implementation (for verifying backpropagation correctness).
- `small_coefficients.py` — TileLang kernel (forward pass) that fuses operations related to small-coefficients, along with a reference implementation and benchmark.
- `small_coefficients_bwd.py` — Backward implementation for small-coefficients.
- `gemm_expsum.py` — TileLang kernel (forward pass) that fuses operations related to gemm_expsum, along with a reference implementation and benchmark.
- `gemm_expsum_bwd.py` — Backward implementation for gemm_expsum.

## Quick Start

First, ensure the Python environment in the repository root meets the project dependencies (recommended to use the repository-provided `requirements*.txt` or the development container). At minimum, you need:

- Python 3.8+
- `torch` (CUDA version depending on the target device)
- `tilelang` installed and importable (within the repository or in the environment)

Run the example scripts from the repository root directory:

```bash
python tilelang/examples/mHC_test/sinkhorn_knopp.py
python tilelang/examples/mHC_test/small_coefficients.py
...
```

These scripts will:
- Perform a numerical comparison between the reference (pure PyTorch) implementation and the TileLang JIT kernel (using `torch.testing.assert_close`).
- Run a benchmark and print the latency (ms) after passing the correctness check.

If you only want to run a quick small-scale verification, you can modify the arguments of the `main(...)` call at the end of each script to a smaller `num_tokens`.

## Developer Notes

- The example implementations rely on TileLang's JIT/`@tl.jit` capability and use `tilelang.profiler.do_bench` for simple benchmarking. If you need to debug or view the generated kernel source code, you can uncomment the print statements for `get_kernel_source()` in the examples.

## Performance Summary

- Test Device: NVIDIA H20, Test Image: nvcr.io/nvidia/pytorch:25.11-py3

 Kernel Type: `gemm_expsum`

| Num Tokens | Forward Latency (ms) | Reference Forward (ms) | Forward Speedup | Backward Latency (ms) | Reference Backward (ms) | Backward Speedup |
|------------|----------------------|------------------------|------------------|-----------------------|-------------------------|-------------------|
| 1          | 0.0228               | 0.0288                 | 1.26×            | 0.0345                | 0.0979                  | 2.84×             |
| 16         | 0.0231               | 0.0375                 | 1.62×            | 0.0351                | 0.0990                  | 2.82×             |
| 1k         | 0.1267               | 0.3382                 | 2.67×            | 0.2660                | 0.5591                  | 2.10×             |
| 16k        | 1.7670               | 4.7822                 | 2.71×            | 4.1307                | 8.2372                  | 1.99×             |
| 64k        | 7.0311               | 18.3406                | 2.61×            | 16.5006               | 32.8566                 | 1.99×             |
| 128k       | 14.0225              | 36.6704                | 2.62×            | 33.2023               | 65.9813                 | 1.99×             |

 Kernel Type: `small_coefficients`

| Num Tokens | Forward Latency (ms) | Reference Forward (ms) | Forward Speedup | Backward Latency (ms) | Reference Backward (ms) | Backward Speedup |
|------------|----------------------|------------------------|------------------|------------------------|--------------------------|-------------------|
| 1          | 0.0046               | 0.0270                 | 5.86×            | 0.0085                 | 0.3444                   | 40.52×            |
| 16         | 0.0046               | 0.0274                 | 5.93×            | 0.0199                 | 0.3523                   | 17.70×            |
| 1k         | 0.0049               | 0.0337                 | 6.83×            | 0.0412                 | 0.3606                   | 8.76×             |
| 16k        | 0.0063               | 0.0373                 | 5.92×            | 0.0451                 | 0.3671                   | 8.14×             |
| 64k        | 0.0092               | 0.0529                 | 5.77×            | 0.1150                 | 0.3867                   | 3.36×             |
| 128k       | 0.0131               | 0.0735                 | 5.61×            | 0.2112                 | 0.3808                   | 1.80×             |

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