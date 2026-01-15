import tilelang as tl
import tilelang.language as T
import torch

from residual_merge import ref_compute

@tl.jit(
    pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,}
)
def residual_merge_bwd(n=4, C=7168, BLOCK_C=64, THREADS_PER_BLOCK=256):
    num_tokens = T.symbolic("num_tokens")
    x_dtype = T.bfloat16
    dtype = T.float32
    assert THREADS_PER_BLOCK >= n * BLOCK_C, "Reduce sum seems buggy in this case."
    
    @T.prim_func
    def residual_merge_bwd_kernel(
        F: T.Tensor((num_tokens, C), x_dtype),
        H_post: T.Tensor((num_tokens, n), dtype),
        x: T.Tensor((num_tokens, n * C), x_dtype),
        H_res: T.Tensor((num_tokens, n * n), dtype),
        grad_res: T.Tensor((num_tokens, n * C), x_dtype),
        grad_F: T.Tensor((num_tokens, C), x_dtype),
        grad_H_post: T.Tensor((num_tokens, n), dtype),
        grad_x: T.Tensor((num_tokens, n * C), x_dtype),
        grad_H_res: T.Tensor((num_tokens, n * n), dtype),
    ):
        with T.Kernel(num_tokens, T.ceildiv(C, BLOCK_C), threads=THREADS_PER_BLOCK) as (bm, bc):
            # Allocate memory
            grad_res_shared = T.alloc_shared((n, BLOCK_C), x_dtype)
            H_post_shared = T.alloc_shared((n,), dtype)
            F_shared = T.alloc_shared((BLOCK_C,), dtype)
            H_res_shared = T.alloc_shared((n * n,), dtype)
            x_shared = T.alloc_shared((n, BLOCK_C), x_dtype)
            
            grad_F_shared = T.alloc_shared((BLOCK_C,), dtype)  # upcast
            grad_x_shared = T.alloc_shared((n, BLOCK_C), dtype)  # upcast
            
            # grad_H_post, grad_H_res sum along C dimension
            grad_H_post_local = T.alloc_shared((n, BLOCK_C), dtype)
            grad_H_res_local = T.alloc_shared((n * n, BLOCK_C), dtype)
            T.clear(grad_H_post_local)
            T.clear(grad_H_res_local)
            
            grad_H_post_sum = T.alloc_shared((n,), dtype)
            grad_H_res_sum = T.alloc_shared((n * n,), dtype)
            
            T.clear(grad_H_post_sum)
            
            for nn in T.Parallel(n):
                H_post_shared[nn] = H_post[bm, nn]
            for nn in T.Parallel(n * n):
                H_res_shared[nn] = H_res[bm, nn]

            for nn, c in T.Parallel(n, BLOCK_C):
                grad_res_shared[nn, c] = grad_res[bm, nn * C + bc * BLOCK_C + c]
                x_shared[nn, c] = x[bm, nn * C + bc * BLOCK_C + c]
                grad_x_shared[nn, c] = 0
            for c in T.Parallel(BLOCK_C):
                F_shared[c] = F[bm, bc * BLOCK_C + c]
                grad_F_shared[c] = 0
            
            # Compute
            for c in T.Parallel(BLOCK_C):
                for nn in T.Serial(n):
                    grad_F_shared[c] += T.cast(grad_res_shared[nn, c], dtype) * H_post_shared[nn]

            for nn, c in T.Parallel(n, BLOCK_C):
                for k in T.Serial(n):
                    grad_x_shared[nn, c] += T.cast(grad_res_shared[k, c], dtype) * H_res_shared[k * n + nn]
                
                G = T.cast(grad_res_shared[nn, c], dtype)
                grad_H_post_local[nn, c] = G * F_shared[c]
                for k in T.Serial(n):
                    grad_H_res_local[nn * n + k, c] += G * x_shared[k, c]
            
            T.reduce_sum(grad_H_post_local, grad_H_post_sum)
            T.reduce_sum(grad_H_res_local, grad_H_res_sum)
            
            # Copy back grad_x, grad_F
            for nn, c in T.Parallel(n, BLOCK_C):
                grad_x[bm, nn * C + bc * BLOCK_C + c] = T.cast(grad_x_shared[nn, c], x_dtype)
            for c in T.Parallel(BLOCK_C):
                grad_F[bm, bc * BLOCK_C + c] = T.cast(grad_F_shared[c], x_dtype)
            # Copy back grad_H_post, grad_H_res
            for nn in T.Parallel(n):
                T.atomic_add(grad_H_post[bm, nn], grad_H_post_sum[nn])
            for nn in T.Parallel(n * n):
                T.atomic_add(grad_H_res[bm, nn], grad_H_res_sum[nn])

    return residual_merge_bwd_kernel

def main(num_tokens=1024):
    print(f"{num_tokens=}")
    n = 4
    C = 7168
    F = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    H_post = torch.randn(num_tokens, n, dtype=torch.float32, device="cuda").requires_grad_(True)
    x = torch.randn(num_tokens, n * C, dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    H_res = torch.randn(num_tokens, n * n, dtype=torch.float32, device="cuda").requires_grad_(True)
    
    ref_res = ref_compute(F, H_post, x, H_res)
    grad_res = torch.randn_like(ref_res)
    ref_res.backward(grad_res, retain_graph=True)
    grad_F_ref = F.grad
    grad_H_post_ref = H_post.grad
    grad_x_ref = x.grad
    grad_H_res_ref = H_res.grad
    
    kernel = residual_merge_bwd(n=n, C=C)
    # print(kernel.get_kernel_source())
    grad_F = torch.zeros_like(F)
    grad_H_post = torch.zeros_like(H_post)
    grad_x = torch.zeros_like(x)
    grad_H_res = torch.zeros_like(H_res)
    kernel(F, H_post, x, H_res, grad_res, grad_F, grad_H_post, grad_x, grad_H_res)
    
    torch.testing.assert_close(grad_F, grad_F_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(grad_H_post, grad_H_post_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(grad_x, grad_x_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(grad_H_res, grad_H_res_ref, atol=1e-2, rtol=1e-2)
    
    from tilelang.profiler import do_bench
    print("Benchmarking TileLang residual merge backward kernel...")
    latency = do_bench(
        lambda: kernel(F, H_post, x, H_res, grad_res, grad_F, grad_H_post, grad_x, grad_H_res),
        warmup=10,
        rep=100,
    )
    print(f"TileLang residual merge backward kernel latency: {latency:.4f} ms")
    ref_latency = do_bench(
        lambda: ref_res.backward(grad_res, retain_graph=True),
        warmup=10,
        rep=100,
    )
    print(f"PyTorch residual merge backward kernel latency: {ref_latency:.4f} ms")
    
if __name__ == "__main__":
    main(1)
    main(16)
    main(1024)
    main(16 * 1024)
    main(64 * 1024)
    main(128 * 1024)
