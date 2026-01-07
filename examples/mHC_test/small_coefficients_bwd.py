import tilelang as tl
import tilelang.language as T
import torch

from small_coefficients import ref_compute

@tl.jit(pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,})
def fused_small_coefficients_bwd(n=4, dtype=T.float32):
    num_tokens = T.symbolic("num_tokens")
    total_dim = n * n + 2 * n  # 24
    TOKENS_PER_BLOCK = 32

    @T.prim_func
    def fused_small_coefficients_bwd_kernel(
        H: T.Tensor((num_tokens, total_dim), dtype),
        alpha_l_pre: T.Tensor((1,), dtype),
        alpha_l_post: T.Tensor((1,), dtype),
        alpha_l_res: T.Tensor((1,), dtype),
        b_l: T.Tensor((total_dim,), dtype),
        r: T.Tensor((num_tokens,), dtype),
        grad_output: T.Tensor((num_tokens, total_dim), dtype),
        # outputs
        grad_H: T.Tensor((num_tokens, total_dim), dtype),
        grad_alpha_pre: T.Tensor((1,), dtype),
        grad_alpha_post: T.Tensor((1,), dtype),
        grad_alpha_res: T.Tensor((1,), dtype),
        grad_b_l: T.Tensor((total_dim,), dtype),
        grad_r: T.Tensor((num_tokens,), dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, TOKENS_PER_BLOCK), threads=TOKENS_PER_BLOCK) as bx:
            tx = T.get_thread_binding()
            token_id = bx * TOKENS_PER_BLOCK + tx

            grad_alpha_pre_shared = T.alloc_shared((1,), dtype)
            grad_alpha_post_shared = T.alloc_shared((1,), dtype)
            grad_alpha_res_shared = T.alloc_shared((1,), dtype)
            T.fill(grad_alpha_pre_shared, T.cast(0, dtype))
            T.fill(grad_alpha_post_shared, T.cast(0, dtype))
            T.fill(grad_alpha_res_shared, T.cast(0, dtype))
            grad_b_l_shared = T.alloc_shared((total_dim,), dtype)
            T.fill(grad_b_l_shared, T.cast(0, dtype))

            if token_id < num_tokens:
                a_pre = alpha_l_pre[0]
                a_post = alpha_l_post[0]
                a_res = alpha_l_res[0]
                inv_r = T.cast(1.0, dtype) / r[token_id]
                inv_r2 = inv_r * inv_r
                
                z = T.alloc_var(dtype)
                sig = T.alloc_var(dtype)
                dz = T.alloc_var(dtype)

                # Thread-local accumulators for scalar gradients
                grad_alpha_pre_local = T.alloc_var(dtype, init=0)
                grad_alpha_post_local = T.alloc_var(dtype, init=0)
                grad_alpha_res_local = T.alloc_var(dtype, init=0)
                grad_r_local = T.alloc_var(dtype, init=0)

                # Local accumulator for grad_b_l (per-dim)
                grad_b_l_local = T.alloc_local([total_dim], dtype=dtype)
                T.fill(grad_b_l_local, T.cast(0, dtype))

                # Process all dimensions for this token
                for dim_id in T.serial(total_dim):  # can also unroll, but serial is fine
                    go = grad_output[token_id, dim_id]
                    x = H[token_id, dim_id]
                    bias = b_l[dim_id]

                    if dim_id < n:
                        z = inv_r * a_pre * x + bias
                        sig = T.sigmoid(z)
                        dz = go * sig * (1.0 - sig)
                        grad_H[token_id, dim_id] = dz * inv_r * a_pre
                        grad_alpha_pre_local += dz * inv_r * x
                        grad_b_l_local[dim_id] += dz
                        grad_r_local += -dz * a_pre * x * inv_r2
                    elif dim_id < 2 * n:
                        z = inv_r * a_post * x + bias
                        sig = T.sigmoid(z)
                        dz = go * T.cast(2.0, dtype) * sig * (1.0 - sig)
                        grad_H[token_id, dim_id] = dz * inv_r * a_post
                        grad_alpha_post_local += dz * inv_r * x
                        grad_b_l_local[dim_id] += dz
                        grad_r_local += -dz * a_post * x * inv_r2
                    else:
                        grad_H[token_id, dim_id] = go * inv_r * a_res
                        grad_alpha_res_local += go * inv_r * x
                        grad_b_l_local[dim_id] += go
                        grad_r_local += -go * a_res * x * inv_r2

                grad_r[token_id] = grad_r_local  # per-token grad_r
                T.atomic_add(grad_alpha_pre_shared[0], grad_alpha_pre_local)
                T.atomic_add(grad_alpha_post_shared[0], grad_alpha_post_local)
                T.atomic_add(grad_alpha_res_shared[0], grad_alpha_res_local)
                for i in T.serial(total_dim):
                    T.atomic_add(grad_b_l_shared[i], grad_b_l_local[i])
            
            T.sync_threads()
            if tx == 0:
                T.atomic_add(grad_alpha_pre[0], grad_alpha_pre_shared[0])
                T.atomic_add(grad_alpha_post[0], grad_alpha_post_shared[0])
                T.atomic_add(grad_alpha_res[0], grad_alpha_res_shared[0])
                for i in T.serial(total_dim):
                    T.atomic_add(grad_b_l[i], grad_b_l_shared[i])

    return fused_small_coefficients_bwd_kernel

def test_bwd(num_tokens = 16 * 1024):
    print(f"Running backward test with num_tokens={num_tokens}")
    n = 4
    H = torch.randn((num_tokens, n * n + 2 * n), dtype=torch.float32).cuda()
    alpha_l_pre = torch.randn((1,), dtype=torch.float32).cuda()
    alpha_l_post = torch.randn((1,), dtype=torch.float32).cuda()
    alpha_l_res = torch.randn((1,), dtype=torch.float32).cuda()
    b_l = torch.randn((n * n + 2 * n,), dtype=torch.float32).cuda()
    r = torch.randn((num_tokens,), dtype=torch.float32).cuda()

    H.requires_grad_(True)
    alpha_l_pre.requires_grad_(True)
    alpha_l_post.requires_grad_(True)
    alpha_l_res.requires_grad_(True)
    b_l.requires_grad_(True)
    r.requires_grad_(True)

    out = ref_compute(H, alpha_l_pre, alpha_l_post, alpha_l_res, b_l, r, n=n)
    grad_output = torch.randn_like(out) / 100.0  # small gradients to avoid numerical issues

    out.backward(grad_output, retain_graph=True)

    grad_H_ref = H.grad
    grad_alpha_pre_ref = alpha_l_pre.grad
    grad_alpha_post_ref = alpha_l_post.grad
    grad_alpha_res_ref = alpha_l_res.grad
    grad_b_l_ref = b_l.grad
    grad_r_ref = r.grad

    # TileLang backward
    jit_bwd = fused_small_coefficients_bwd(n=n)
    # print(jit_bwd.get_kernel_source())
    (grad_H_tl,
     grad_alpha_pre_tl,
     grad_alpha_post_tl,
     grad_alpha_res_tl,
     grad_b_l_tl,
     grad_r_tl) = (
        torch.zeros_like(H),
        torch.zeros_like(alpha_l_pre),
        torch.zeros_like(alpha_l_post),
        torch.zeros_like(alpha_l_res),
        torch.zeros_like(b_l),
        torch.zeros_like(r),
    )
    jit_bwd(
        H,
        alpha_l_pre,
        alpha_l_post,
        alpha_l_res,
        b_l,
        r,
        grad_output,
        grad_H_tl,
        grad_alpha_pre_tl,
        grad_alpha_post_tl,
        grad_alpha_res_tl,
        grad_b_l_tl,
        grad_r_tl,
    )

    torch.testing.assert_close(grad_H_tl, grad_H_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_alpha_pre_tl, grad_alpha_pre_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grad_alpha_post_tl, grad_alpha_post_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grad_alpha_res_tl, grad_alpha_res_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grad_b_l_tl, grad_b_l_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grad_r_tl, grad_r_ref, atol=1e-4, rtol=1e-4)
    print("Backward kernel passed!")
    
    from tilelang.profiler import do_bench
    def run_bwd():
        jit_bwd(
            H,
            alpha_l_pre,
            alpha_l_post,
            alpha_l_res,
            b_l,
            r,
            grad_output,
            grad_H_tl,
            grad_alpha_pre_tl,
            grad_alpha_post_tl,
            grad_alpha_res_tl,
            grad_b_l_tl,
            grad_r_tl,
        )
    bwd_latency = do_bench(run_bwd, warmup=10, rep=100)
    print(f"Backward latency: {bwd_latency} ms")
    def run_ref():
        out.backward(grad_output, retain_graph=True)
    ref_latency = do_bench(run_ref, warmup=10, rep=100)
    print(f"Ref backward latency: {ref_latency} ms")
    
if __name__ == "__main__":
    test_bwd(1)
    test_bwd(16)
    test_bwd(1024)
    test_bwd(16 * 1024)
    test_bwd(64 * 1024)
    test_bwd(128 * 1024)