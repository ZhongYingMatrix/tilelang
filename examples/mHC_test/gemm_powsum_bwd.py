import tilelang as tl
import tilelang.language as T
import torch

from gemm_powsum import ref_compute

@tl.jit(
    pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,}
)
def fused_gemm_powsum_bwd(n=4, C=7168, dtype=T.float32, THREADS_PER_BLOCK = 128, BLOCK_M = 128, BLOCK_N = 128):
    num_tokens = T.symbolic("num_tokens")
    BLOCK_K = 32  # total_dim = n * n + 2 * n  -> 24
    x_dtype = T.bfloat16
    dtype = T.float32
    
    @T.prim_func
    def fused_gemm_powsum_bwd_kernel(
        x: T.Tensor((num_tokens, n * C), x_dtype),
        phi: T.Tensor((n * C, n * n + 2 * n), dtype),
        # H: T.Tensor((num_tokens, n * n + 2 * n), dtype),
        r: T.Tensor((num_tokens,), dtype),
        grad_H: T.Tensor((num_tokens, n * n + 2 * n), dtype),
        grad_r: T.Tensor((num_tokens,), dtype),
        grad_x: T.Tensor((num_tokens, n * C), x_dtype),
        grad_phi: T.Tensor((n * C, n * n + 2 * n), dtype),
    ):
        # with T.Kernel(T.ceildiv(num_tokens, BLOCK_M), T.ceildiv(n * C, BLOCK_N), threads=THREADS_PER_BLOCK) as (bm, bn):
        # with T.Kernel(T.ceildiv(num_tokens, BLOCK_M), threads=THREADS_PER_BLOCK) as bm:
        with T.Kernel(T.ceildiv(n * C, BLOCK_N), threads=THREADS_PER_BLOCK) as bn:
            # Shared memory for x, phi, H, r, grad_H, grad_r, grad_x, grad_phi
            x_shared = T.alloc_shared((BLOCK_M, BLOCK_N), x_dtype)
            phi_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            # H_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            r_shared = T.alloc_shared([BLOCK_M], dtype)
            grad_H_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            grad_r_shared = T.alloc_shared([BLOCK_M], dtype)
            # grad_x_shared = T.alloc_shared((BLOCK_M, BLOCK_N), x_dtype)
            # grad_phi_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            
            # Fragment memory for x_cast, grad_x and grad_phi
            x_cast = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
            grad_x_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)  # needs to cast back x_dtype
            grad_phi_frag = T.alloc_fragment((BLOCK_N, BLOCK_K), dtype)
            
            # Clear grad phi out of loop
            T.clear(grad_phi_frag)
            
            # for bn in T.Pipelined(T.ceildiv(n * C, BLOCK_N), num_stages=3):
            for bm in T.Pipelined(T.ceildiv(num_tokens, BLOCK_M), num_stages=3):
                T.clear(grad_x_frag)

                # Copy global to shared
                T.copy(x[bm * BLOCK_M, bn * BLOCK_N], x_shared)
                T.copy(phi[bn * BLOCK_N, 0], phi_shared)
                # T.copy(H[bm * BLOCK_M, 0], H_shared)
                T.copy(r[bm * BLOCK_M], r_shared)
                T.copy(grad_H[bm * BLOCK_M, 0], grad_H_shared)
                T.copy(grad_r[bm * BLOCK_M], grad_r_shared)
                
                # Cast
                for m, nn in T.Parallel(BLOCK_M, BLOCK_N):
                    x_cast[m, nn] = T.cast(x_shared[m, nn], dtype)
                
                # Perform mma
                T.gemm(grad_H_shared, phi_shared, grad_x_frag, transpose_B=True)
                T.gemm(x_cast, grad_H_shared, grad_phi_frag, transpose_A=True)
                
                for m, nn in T.Parallel(BLOCK_M, BLOCK_N):
                    grad_x_frag[m, nn] += x_shared[m, nn] / (n * C) / r_shared[m] * grad_r_shared[m]

                # Copy grad x to global
                # T.copy(grad_x_frag, grad_x[bm * BLOCK_M, bn * BLOCK_N])
                for m, nn in T.Parallel(BLOCK_M, BLOCK_N):
                    grad_x[bm * BLOCK_M + m, bn * BLOCK_N + nn] = T.cast(grad_x_frag[m, nn], x_dtype)
            # Copy grad fhi
            T.copy(grad_phi_frag, grad_phi[bn * BLOCK_N, 0])

    return fused_gemm_powsum_bwd_kernel


def main(num_tokens=1024):
    print(f"{num_tokens=}")
    n = 4
    C = 7168
    x = torch.randn((num_tokens, n * C), dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    phi = torch.randn((n * C, n * n + 2 * n), dtype=torch.float32, device="cuda").requires_grad_(True)
    ref_H, ref_r = ref_compute(x, phi)
    
    grad_H = torch.randn_like(ref_H)
    grad_r = torch.randn_like(ref_r)
    
    grad_x_ref, grad_phi_ref = torch.autograd.grad(
        [ref_H, ref_r],
        [x, phi],
        grad_outputs=[grad_H, grad_r],
        retain_graph=True
    )
    
    bwd_kernel = fused_gemm_powsum_bwd(n=n, C=C)
    # print(bwd_kernel.get_kernel_source())
    # H = torch.zeros((num_tokens, n * n + 2 * n), dtype=torch.float32, device="cuda")
    r = torch.clone(ref_r)
    grad_x = torch.zeros_like(x)
    grad_phi = torch.zeros_like(phi)
    bwd_kernel(x, phi, r, grad_H, grad_r, grad_x, grad_phi)
    # breakpoint()
    try:
        torch.testing.assert_close(grad_x, grad_x_ref, atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(grad_phi, grad_phi_ref, atol=5e-1, rtol=5e-1)
    except Exception as e:
        print(e)
        breakpoint()
    
    
    from tilelang.profiler import do_bench
    print("Benchmarking TileLang gemm powsum bwd kernel...")
    latency = do_bench(
        lambda: bwd_kernel(x, phi, r, grad_H, grad_r, grad_x, grad_phi),
        warmup=10,
        rep=100,
    )
    ref_latency = do_bench(
        lambda: torch.autograd.grad(
            [ref_H, ref_r],
            [x, phi],
            grad_outputs=[grad_H, grad_r],
            retain_graph=True
        ),
        warmup=10,
        rep=100,
    )
    print(f"TileLang gemm powsum bwd latency: {latency} ms")
    print(f"Reference gemm powsum bwd latency: {ref_latency} ms")
    
if __name__ == "__main__":
    main(num_tokens=1)
    main(num_tokens=16)
    main(num_tokens=1024)
    main(num_tokens=16 * 1024)
    main(num_tokens=64 * 1024)
    main(num_tokens=128 * 1024)