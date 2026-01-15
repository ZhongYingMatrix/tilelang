import tilelang as tl
import tilelang.language as T
import torch

r"""
\mathcal{F}_{\mathrm{post,res}}\coloneq \hres{l}\mathbf{x}_l+\mathcal{H}_{l}^{\mathrm{post}\, \top}\mathcal{F}(\cdot,\cdot)
"""

def ref_compute(F, H_post, x, H_res):
    # F: (num_tokens, C) bf16
    # H_post: (num_tokens, n) float32
    # x: (num_tokens, n * C) bf16
    # H_res: (num_tokens, n * n) float32
    C = F.shape[-1]
    n = H_post.shape[-1]
    F = F.float()
    x = x.float().view(-1, n, C)
    H_res = H_res.view(-1, n, n)
    res = torch.einsum("bnk,bkc->bnc", H_res, x) + torch.einsum("bn,bc->bnc", H_post, F)
    return res.view(-1, n * C).bfloat16()

@tl.jit(
    pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,}
)
def residual_merge(n=4, C=7168, BLOCK_M=64, BLOCK_C=64, THREADS_PER_BLOCK=128):
    num_tokens = T.symbolic("num_tokens")
    x_dtype = T.bfloat16
    dtype = T.float32
    assert n == 4, "Nested Serial doesnt seem to work, hardcoded 4 for now."
    
    @T.prim_func
    def residual_merge_kernel(
        F: T.Tensor((num_tokens, C), x_dtype),
        H_post: T.Tensor((num_tokens, n), dtype),
        x: T.Tensor((num_tokens, n * C), x_dtype),
        H_res: T.Tensor((num_tokens, n * n), dtype),
        res: T.Tensor((num_tokens, n * C), x_dtype)):
        with T.Kernel(T.ceildiv(num_tokens, BLOCK_M), T.ceildiv(C, BLOCK_C), threads=THREADS_PER_BLOCK) as (bm, bc):
            # Allocate shared memory
            F_shared = T.alloc_shared((BLOCK_M, BLOCK_C), dtype)  # upcast
            H_post_shared = T.alloc_shared((BLOCK_M, n), dtype)
            x_shared = T.alloc_shared((BLOCK_M, n * BLOCK_C), dtype)  # upcast
            H_res_shared = T.alloc_shared((BLOCK_M, n * n), dtype)
            
            # Copy from global to shared
            for m, c in T.Parallel(BLOCK_M, BLOCK_C):
                F_shared[m, c] = T.cast(F[bm * BLOCK_M + m, bc * BLOCK_C + c], dtype)
                for nn in T.Serial(n):
                    x_shared[m, nn * BLOCK_C + c] = T.cast(x[bm * BLOCK_M + m, nn * C + bc * BLOCK_C + c], dtype)
            for m, nn in T.Parallel(BLOCK_M, n):
                H_post_shared[m, nn] = H_post[bm * BLOCK_M + m, nn]
            for m, nn in T.Parallel(BLOCK_M, n * n):
                H_res_shared[m, nn] = H_res[bm * BLOCK_M + m, nn]
            
            # Compute
            for m, c in T.Parallel(BLOCK_M, BLOCK_C):
                for nn in T.Serial(n):
                    val = 0
                    # for kk in T.Serial(kn):
                    #     val += H_res_shared[m, nn * n + kk] * x_shared[m, kk * BLOCK_C + c]
                    val += H_res_shared[m, nn * n] * x_shared[m, c]
                    val += H_res_shared[m, nn * n + 1] * x_shared[m, 1 * BLOCK_C + c]
                    val += H_res_shared[m, nn * n + 2] * x_shared[m, 2 * BLOCK_C + c]
                    val += H_res_shared[m, nn * n + 3] * x_shared[m, 3 * BLOCK_C + c]
                    val += H_post_shared[m, nn] * F_shared[m, c]
                    res[bm * BLOCK_M + m, nn * C + bc * BLOCK_C + c] = T.cast(val, x_dtype)

    return residual_merge_kernel

def main(num_tokens=1024):
    print(f"{num_tokens=}")
    n = 4
    C = 7168
    F = torch.randn((num_tokens, C), dtype=torch.bfloat16, device="cuda")
    H_post = torch.randn((num_tokens, n), dtype=torch.float32, device="cuda")
    x = torch.randn((num_tokens, n * C), dtype=torch.bfloat16, device="cuda")
    H_res = torch.randn((num_tokens, n * n), dtype=torch.float32, device="cuda")
    ref_res = ref_compute(F, H_post, x, H_res)
    
    kernel = residual_merge(n=n, C=C)
    res = torch.zeros_like(ref_res)
    kernel(F, H_post, x, H_res, res)
    torch.testing.assert_close(res, ref_res, atol=1e-2, rtol=1e-2)
    # print(f"{res=}, {ref_res=}")
    
    from tilelang.profiler import do_bench
    print("Benchmarking TileLang residual merge kernel...")
    latency = do_bench(
        lambda: kernel(F, H_post, x, H_res, res),
        warmup=10,
        rep=100,
    )
    print(f"TileLang residual merge kernel latency: {latency:.4f} ms")
    ref_latency = do_bench(
        lambda: ref_compute(F, H_post, x, H_res),
        warmup=10,
        rep=100,
    )
    print(f"PyTorch residual merge kernel latency: {ref_latency:.4f} ms")

if __name__ == "__main__":
    main(1)
    main(16)
    main(1024)
    main(16 * 1024)
    main(64 * 1024)
    main(128 * 1024)