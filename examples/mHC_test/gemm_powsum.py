
import tilelang as tl
import tilelang.language as T
import torch

r"""
    \left[{\ttlhpre{l}}, {\ttlhpost{l}}, {\ttlhres{l}}\right]   &: \text{float32}           &&= \vec{\mathbf{x}}_l\phi_l                                                  \label{eq:fuse:5}\\
    r                                                                               &: \text{float32}           &&= \left\|\vec{\mathbf{x}}_l\right\|_2 / \sqrt{nC}                                                       \label{eq:fuse:6}\\
"""

def ref_compute(x, phi):
    # x: (num_tokens, n * C) bf16
    # phi: (n * C, n**2 + 2 * n) fp32
    x = x.float()
    H = x @ phi
    r = (x.pow(2).sum(dim=-1)/x.shape[-1]).sqrt()
    # r = (x.pow(2).sum(dim=-1)/x.shape[-1])
    return H, r

@tl.jit(
    # out_idx=[-2, -1],
    pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,}
)
def fused_gemm_powsum_split_k(n, C, dtype, THREADS_PER_BLOCK, BLOCK_M, BLOCK_K, split_k):
    num_tokens = T.symbolic("num_tokens")
    BLOCK_N = 32  # total_dim = n * n + 2 * n  -> 24
    x_dtype = T.bfloat16
    # x_dtype = T.float32
    dtype = T.float32
    
    SplitK = T.ceildiv(n * C, split_k)
    
    @T.prim_func
    def fused_gemm_powsum_kernel(
        x: T.Tensor((num_tokens, n * C), x_dtype),
        phi: T.Tensor((n * C, n * n + 2 * n), dtype),
        H: T.Tensor((num_tokens, n * n + 2 * n), dtype),
        r: T.Tensor((num_tokens,), dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, BLOCK_M), split_k, threads=THREADS_PER_BLOCK) as (bx, bk):
            # Shared memory for x and phi and r
            x_shared = T.alloc_shared((BLOCK_M, BLOCK_K), x_dtype)
            phi_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            r_shared = T.alloc_shared([BLOCK_M], dtype)

            # Fragment memory for x, H
            x_frag = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)  # upcast
            r_frag = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
            H_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
            
            T.clear(x_frag)
            T.clear(r_frag)
            T.clear(H_frag)
            
            # Loop over K dim
            # for k in T.Pipelined(T.ceildiv(n * C, BLOCK_K), num_stages=3):
            for k in T.Pipelined(T.ceildiv(SplitK, BLOCK_K), num_stages=3):
                # Copy from global to shared
                T.copy(x[bx * BLOCK_M, bk * SplitK + k * BLOCK_K], x_shared)
                T.copy(phi[bk * SplitK + k * BLOCK_K, 0], phi_shared)
                for m, kk in T.Parallel(BLOCK_M, BLOCK_K):
                    val = T.cast(x_shared[m, kk], dtype)
                    x_frag[m, kk] = val
                    r_frag[m, kk] += val * val
                
                # Perform mma
                T.gemm(x_frag, phi_shared, H_frag)
            
            T.reduce_sum(r_frag, r_shared, dim=1)
            
            # Copy from fragment to global
            # T.copy(H_frag, H[bx * BLOCK_M, 0])
            for m, nn in T.Parallel(BLOCK_M, BLOCK_N):
                # H[bx * BLOCK_M + m, nn] = H_frag[m, nn]
                T.atomic_add(H[bx * BLOCK_M + m, nn], H_frag[m, nn])
            for m in T.Parallel(BLOCK_M):
                # r[bx * BLOCK_M + m] = T.sqrt(r_shared[m] / (n * C))
                T.atomic_add(r[bx * BLOCK_M + m], r_shared[m] / (n * C))
    
    return fused_gemm_powsum_kernel

@tl.jit(
    pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,}
)
def post_sqrt(THREADS_PER_BLOCK = 128, BLOCK_M = 128):
    num_tokens = T.symbolic("num_tokens")
    dtype = T.float32
    
    @T.prim_func
    def post_sqrt_kernel(r: T.Tensor((num_tokens,), dtype)):
         with T.Kernel(T.ceildiv(num_tokens, BLOCK_M), threads=THREADS_PER_BLOCK) as bx:
             for m in T.Parallel(BLOCK_M):
                 r[bx * BLOCK_M + m] = T.sqrt(r[bx * BLOCK_M + m])
    return post_sqrt_kernel

def fused_gemm_powsum(n=4, C=7168, dtype=T.float32, THREADS_PER_BLOCK = 128, BLOCK_M = 128, BLOCK_K = 32, split_k = 64):
    k0 = fused_gemm_powsum_split_k(n, C, dtype, THREADS_PER_BLOCK, BLOCK_M, BLOCK_K, split_k)
    k1 = post_sqrt(THREADS_PER_BLOCK, BLOCK_M)
    
    def fused_gemm_powsum_fn(x, phi, H, r):
        k0(x, phi, H, r)
        k1(r)
        
    return fused_gemm_powsum_fn

def main(num_tokens=1024):
    print(f"{num_tokens=}")
    n = 4
    C = 7168
    x = torch.randn((num_tokens, n * C), dtype=torch.bfloat16, device="cuda")
    phi = torch.randn((n * C, n * n + 2 * n), dtype=torch.float32, device="cuda")
    ref_H, ref_r = ref_compute(x, phi)
    fused_kernel = fused_gemm_powsum(n=n, C=C)
    # print(fused_kernel.get_kernel_source())
    H = torch.zeros((num_tokens, n * n + 2 * n), dtype=torch.float32, device="cuda")
    r = torch.zeros((num_tokens,), dtype=torch.float32, device="cuda")
    fused_kernel(x, phi, H, r)
    # breakpoint()
    torch.testing.assert_close(H, ref_H, atol=5e-1, rtol=5e-1)
    torch.testing.assert_close(r, ref_r, atol=1e-5, rtol=1e-5)
    
    
    from tilelang.profiler import do_bench
    print("Benchmarking TileLang gemm powsum kernel...")
    latency = do_bench(
        lambda: fused_kernel(x, phi, H, r),
        warmup=10,
        rep=100,
    )
    ref_latency = do_bench(
        lambda: ref_compute(x, phi),
        warmup=10,
        rep=100,
    )
    print(f"TileLang gemm powsum latency: {latency} ms")
    print(f"Reference gemm powsum latency: {ref_latency} ms")
    
if __name__ == "__main__":
    main(num_tokens=1)
    main(num_tokens=16)
    main(num_tokens=1024)
    main(num_tokens=16 * 1024)
    main(num_tokens=64 * 1024)
    main(num_tokens=128 * 1024)