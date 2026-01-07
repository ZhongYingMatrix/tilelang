import tilelang as tl
import tilelang.language as T
import torch


r"""
    \hres{l}                                                                      &: \text{float32}           &&= \text{Sinkhorn-Knopp}\left(\tlhres{l}\right)   \label{eq:fuse:10}

"""

def ref_compute(tlhres, n=4, num_iters=20):
    # tlhres: (num_tokens, n * n) fp32
    num_tokens = tlhres.shape[0]
    matrix = tlhres.view(num_tokens, n, n)
    for _ in range(num_iters):
        matrix = matrix / matrix.sum(dim=2, keepdim=True)
        matrix = matrix / matrix.sum(dim=1, keepdim=True)
    new_tlhres = matrix.view(num_tokens, n * n)
    return new_tlhres

@tl.jit(out_idx=[-1])
def sinkhorn_knopp(
    n: int,
    num_iters: int = 20,
    blk_m: int = 64,
    # threads: int = 256,
):
    num_tokens = T.symbolic("num_tokens")
    dtype = T.float32

    @T.prim_func
    def sinkhorn_knopp_kernel(
        input_flat: T.Tensor((num_tokens, n * n), dtype),
        output_flat: T.Tensor((num_tokens, n * n), dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), threads=blk_m) as bx:
            tx = T.get_thread_binding()
            sample_id = bx * blk_m + tx
            # T.print(sample_id, msg="sample_id:")

            # Allocate per-thread local matrix
            matrix = T.alloc_local((n, n), dtype=dtype)
            row_sum = T.alloc_local((n,), dtype=dtype)
            col_sum = T.alloc_local((n,), dtype=dtype)
            acc = T.alloc_var(dtype)

            # Load: (n, n) from flat input
            for j in T.serial(n):
                for k in T.serial(n):
                    if sample_id < num_tokens:
                        matrix[j, k] = input_flat[sample_id, j * n + k]
                    else:
                        matrix[j, k] = T.float32(0.0)

            # Sinkhorn iterations
            for it in T.serial(num_iters):
                # Row normalization: sum over columns (k)
                for j in T.serial(n):
                    acc = T.float32(0.0)
                    for k in T.serial(n):
                        acc += matrix[j, k]
                    row_sum[j] = acc

                for j in T.serial(n):
                    for k in T.serial(n):
                        matrix[j, k] = T.if_then_else(
                            row_sum[j] > T.float32(1e-12),
                            matrix[j, k] / row_sum[j],
                            T.float32(0.0)
                        )

                # Column normalization: sum over rows (j)
                for k in T.serial(n):
                    acc = T.float32(0.0)
                    for j in T.serial(n):
                        acc += matrix[j, k]
                    col_sum[k] = acc

                for j in T.serial(n):
                    for k in T.serial(n):
                        matrix[j, k] = T.if_then_else(
                            col_sum[k] > T.float32(1e-12),
                            matrix[j, k] / col_sum[k],
                            T.float32(0.0)
                        )

            # Store back
            for j in T.serial(n):
                for k in T.serial(n):
                    if sample_id < num_tokens:
                        output_flat[sample_id, j * n + k] = matrix[j, k]

    return sinkhorn_knopp_kernel

def main(num_tokens=16 * 1024, n=4, num_iters=20):
    print(f"Running Sinkhorn-Knopp test with num_tokens={num_tokens}, n={n}, num_iters={num_iters}")
    jit_kernel = sinkhorn_knopp(n=n, num_iters=num_iters)
    # print(jit_kernel.get_kernel_source())

    tlhres_in = torch.randn(num_tokens, n * n, device="cuda", dtype=torch.float32).abs()
    # breakpoint()
    # tlhres_out = torch.zeros_like(tlhres_in)

    tlhres_out = jit_kernel(tlhres_in)

    tlhres_ref = ref_compute(tlhres_in, n=n, num_iters=num_iters)

    # breakpoint()
    
    torch.testing.assert_close(tlhres_out, tlhres_ref, atol=1e-5, rtol=1e-5)

    print("Sinkhorn-Knopp TileLang kernel passed the correctness test.")
    
    from tilelang.profiler import do_bench
    print("Benchmarking TileLang Sinkhorn-Knopp kernel...")
    latency = do_bench(
        lambda: jit_kernel(tlhres_in),
        warmup=10,
        rep=100,
    )
    ref_latency = do_bench(
        lambda: ref_compute(tlhres_in, n=n, num_iters=num_iters),
        warmup=10,
        rep=100,
    )
    print(f"TileLang Sinkhorn-Knopp latency: {latency} ms")
    print(f"Reference Sinkhorn-Knopp latency: {ref_latency} ms")
    
if __name__ == "__main__":
    main(1)
    main(16)
    main(1024)
    main(16 * 1024)
    main(64 * 1024)
    main(128 * 1024)
