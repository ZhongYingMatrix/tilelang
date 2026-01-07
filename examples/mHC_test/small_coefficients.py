
import tilelang as tl
import tilelang.language as T
import torch

r"""
    \left[\tlhpre{l}, \tlhpost{l}, \tlhres{l}\right]         &: \text{float32}           &&= 1/r \left[\alpha_l^\mathrm{pre}{\ttlhpre{l}}, \alpha_l^\mathrm{post}{\ttlhpost{l}}, \alpha_l^\mathrm{res}{\ttlhres{l}}\right] + \mathbf{b}_l \label{eq:fuse:7}\\
    \hpre{l}                                                                      &: \text{float32}           &&= \sigma\left(\tlhpre{l}\right)                                   \label{eq:fuse:8}\\
    \hpost{l}                                                                      &: \text{float32}           &&= 2\sigma\left(\tlhpost{l}\right)                                  \label{eq:fuse:9}\\
"""
# n = 4 typically

def ref_compute(H, alpha_l_pre, alpha_l_post, alpha_l_res, b_l, r, n=4):
    # H: (num_tokens, n**2 + 2 * n) fp32
    # alpha_l_pre, alpha_l_post, alpha_l_res: scalars, fp32
    # b_l: (n**2 + 2 * n,) fp32
    # r: (num_tokens,), fp32

    tlhpre = (alpha_l_pre * H[:, : n]) / r.unsqueeze(1) + b_l[: n]
    tlhpost = (alpha_l_post * H[:, n : 2 * n]) / r.unsqueeze(1) + b_l[n : 2 * n]
    tlhres = (alpha_l_res * H[:, 2 * n :]) / r.unsqueeze(1) + b_l[2 * n :]
    hpre = torch.sigmoid(tlhpre)
    hpost = 2 * torch.sigmoid(tlhpost)

    new_H = torch.empty_like(H)
    new_H[:, : n] = hpre
    new_H[:, n : 2 * n] = hpost
    new_H[:, 2 * n :] = tlhres
    return new_H

@tl.jit(pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,})
def fused_small_coefficients(n=4, dtype=T.float32):
    num_tokens = T.symbolic("num_tokens")
    total_dim = n * n + 2 * n  # 24
    # THREADS_PER_BLOCK = 256
    TOKENS_PER_BLOCK = 32

    @T.prim_func
    def fused_small_coefficients_kernel(
        H: T.Tensor((num_tokens, total_dim), dtype),
        alpha_l_pre: T.Tensor((1,), dtype),
        alpha_l_post: T.Tensor((1,), dtype),
        alpha_l_res: T.Tensor((1,), dtype),
        b_l: T.Tensor((total_dim,), dtype),
        r: T.Tensor((num_tokens,), dtype),
        H_out: T.Tensor((num_tokens, total_dim), dtype),
    ):
        # Grid: ceil(num_tokens / TOKENS_PER_BLOCK)
        with T.Kernel(T.ceildiv(num_tokens, TOKENS_PER_BLOCK), threads=TOKENS_PER_BLOCK) as bx:
            tx = T.get_thread_binding()
            token_id = bx * TOKENS_PER_BLOCK + tx

            if token_id < num_tokens:
                a_pre, a_post, a_res = alpha_l_pre[0], alpha_l_post[0], alpha_l_res[0]
                scale = 1.0 / r[token_id]
                for dim_id in T.unroll(n):
                    H_out[token_id, dim_id] = T.sigmoid(a_pre * scale * H[token_id, dim_id] + b_l[dim_id])
                for dim_id in T.unroll(n, 2 * n):
                    H_out[token_id, dim_id] = T.sigmoid(a_post * scale * H[token_id, dim_id] + b_l[dim_id]) * T.cast(2.0, dtype)
                for dim_id in T.unroll(2 * n, total_dim):
                    H_out[token_id, dim_id] = a_res * scale * H[token_id, dim_id] + b_l[dim_id]

    return fused_small_coefficients_kernel

def main(num_tokens = 16 * 1024):
    print(f"Running test with num_tokens={num_tokens}")
    n = 4
    H = torch.randn((num_tokens, n * n + 2 * n), dtype=torch.float32).cuda()
    alpha_l_pre = torch.randn((1,), dtype=torch.float32).cuda()
    alpha_l_post = torch.randn((1,), dtype=torch.float32).cuda()
    alpha_l_res = torch.randn((1,), dtype=torch.float32).cuda()
    b_l = torch.randn((n * n + 2 * n,), dtype=torch.float32).cuda()
    r = torch.randn((num_tokens,), dtype=torch.float32).cuda()
    ref_out = ref_compute(H, alpha_l_pre, alpha_l_post, alpha_l_res, b_l, r, n=n)

    fused_kernel = fused_small_coefficients(n=n)
    # print(fused_kernel.get_kernel_source())
    fused_out = torch.empty_like(H)
    fused_kernel(
        H,
        alpha_l_pre,
        alpha_l_post,
        alpha_l_res,
        b_l,
        r,
        fused_out,
    )

    torch.testing.assert_close(fused_out, ref_out, atol=1e-6, rtol=1e-6)
    print("Test passed!")
    
    from tilelang.profiler import do_bench
    def run_fused():
        return fused_kernel(
            H,
            alpha_l_pre,
            alpha_l_post,
            alpha_l_res,
            b_l,
            r,
            fused_out,
        )
    fused_latency = do_bench(run_fused, warmup=10, rep=100)
    print(f"Fused latency: {fused_latency} ms")
    def run_ref():
        return ref_compute(H, alpha_l_pre, alpha_l_post, alpha_l_res, b_l, r, n=n)
    ref_latency = do_bench(run_ref, warmup=10, rep=100)
    print(f"Ref latency: {ref_latency} ms")
    

if __name__ == "__main__":
    main(1)
    main(16)
    main(1024)
    main(16 * 1024)
    main(64 * 1024)
    main(128 * 1024)