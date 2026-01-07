import tilelang as tl
import tilelang.language as T
import torch

from sinkhorn_knopp import ref_compute

@tl.jit(out_idx=[-1])
def sinkhorn_knopp_bwd(
    n: int,
    num_iters: int = 20,
    blk_m: int = 64,
):
    num_tokens = T.symbolic("num_tokens")
    dtype = T.float32

    @T.prim_func
    def sinkhorn_knopp_bwd_kernel(
        input_flat: T.Tensor((num_tokens, n * n), dtype),      # A: forward input
        grad_output_flat: T.Tensor((num_tokens, n * n), dtype), # dL/dP
        grad_input_flat: T.Tensor((num_tokens, n * n), dtype),  # dL/dA (output)
    ):
        with T.Kernel(T.ceildiv(num_tokens, blk_m), threads=blk_m) as bx:
            tx = T.get_thread_binding()
            sample_id = bx * blk_m + tx

            # Local matrices
            X = T.alloc_local((n, n), dtype=dtype)   # current matrix during forward replay
            dX = T.alloc_local((n, n), dtype=dtype)  # gradient w.r.t. X (starts as dL/dP)

            # History of row_sum and col_sum for each iteration
            row_sum_hist = T.alloc_local((num_iters, n), dtype=dtype)
            col_sum_hist = T.alloc_local((num_iters, n), dtype=dtype)

            eps = T.float32(1e-12)
            acc = T.alloc_var(dtype)

            # --- Load input and initialize ---
            for j in T.serial(n):
                for k in T.serial(n):
                    flat_idx = j * n + k
                    if sample_id < num_tokens:
                        X[j, k] = T.exp(input_flat[sample_id, flat_idx])
                        dX[j, k] = grad_output_flat[sample_id, flat_idx]
                    else:
                        X[j, k] = T.float32(0.0)
                        dX[j, k] = T.float32(0.0)

            # --- Forward replay: record row_sum and col_sum ---
            for it in T.serial(num_iters):
                # Row normalization: sum over columns (dim=2)
                for j in T.serial(n):
                    acc = T.float32(0.0)
                    for k in T.serial(n):
                        acc += X[j, k]
                    row_sum_hist[it, j] = acc

                for j in T.serial(n):
                    rs = row_sum_hist[it, j]
                    rs_inv = T.if_then_else(rs > eps, T.float32(1.0) / rs, T.float32(0.0))
                    for k in T.serial(n):
                        X[j, k] = X[j, k] * rs_inv

                # Column normalization: sum over rows (dim=1)
                for k in T.serial(n):
                    acc = T.float32(0.0)
                    for j in T.serial(n):
                        acc += X[j, k]
                    col_sum_hist[it, k] = acc

                for k in T.serial(n):
                    cs = col_sum_hist[it, k]
                    cs_inv = T.if_then_else(cs > eps, T.float32(1.0) / cs, T.float32(0.0))
                    for j in T.serial(n):
                        X[j, k] = X[j, k] * cs_inv

            # --- Backward pass: reverse iterations ---
            for it_rev in T.serial(num_iters):
                it = num_iters - 1 - it_rev

                # ---- Undo column normalization (reverse of second step) ----
                # dX = (dX - mean(dX * Y, dim=1)) / col_sum
                # But we don't have Y stored; however, X currently IS Y (after both normalizations)
                # So we use current X as Y_col

                # Compute (dX * X).sum(dim=1, keepdim=True) → per column
                col_mean = T.alloc_local((n,), dtype=dtype)
                for k in T.serial(n):
                    acc = T.float32(0.0)
                    for j in T.serial(n):
                        acc += dX[j, k] * X[j, k]
                    col_mean[k] = acc

                # Update dX: subtract column mean, then divide by col_sum
                for k in T.serial(n):
                    cs = col_sum_hist[it, k]
                    for j in T.serial(n):
                        corrected_grad = dX[j, k] - col_mean[k]
                        dX[j, k] = T.if_then_else(cs > eps, corrected_grad / cs, T.float32(0.0))

                # Now "undo" the column scaling: X before column norm = X * col_sum
                # But for next backward step (row norm), we need X as it was after row norm only
                # So reconstruct X_row = X * col_sum_hist[it]
                for k in T.serial(n):
                    cs = col_sum_hist[it, k]
                    for j in T.serial(n):
                        X[j, k] = X[j, k] * cs  # now X is state after row norm, before col norm

                # ---- Undo row normalization (reverse of first step) ----
                # dX = (dX - mean(dX * X, dim=2)) / row_sum

                row_mean = T.alloc_local((n,), dtype=dtype)
                for j in T.serial(n):
                    acc = T.float32(0.0)
                    for k in T.serial(n):
                        acc += dX[j, k] * X[j, k]
                    row_mean[j] = acc

                for j in T.serial(n):
                    rs = row_sum_hist[it, j]
                    for k in T.serial(n):
                        corrected_grad = dX[j, k] - row_mean[j]
                        dX[j, k] = T.if_then_else(rs > eps, corrected_grad / rs, T.float32(0.0))

                # Reconstruct X to original state before this iteration (for potential debug, not strictly needed)
                for j in T.serial(n):
                    rs = row_sum_hist[it, j]
                    for k in T.serial(n):
                        X[j, k] = X[j, k] * rs

            # --- Store final gradient ---
            for j in T.serial(n):
                for k in T.serial(n):
                    flat_idx = j * n + k
                    if sample_id < num_tokens:
                        grad_input_flat[sample_id, flat_idx] = dX[j, k] * X[j, k]

    return sinkhorn_knopp_bwd_kernel


def test_bwd(num_tokens = 128):
    print(f"Testing Sinkhorn-Knopp backward with num_tokens={num_tokens}")
    torch.manual_seed(0)
    n = 4
    num_iters = 20
    

    # Inputs
    A = torch.randn(num_tokens, n * n, device="cuda", dtype=torch.float32).abs().requires_grad_(True)
    grad_P = torch.randn_like(A)

    # Reference backward
    A_ref = A.clone().detach().requires_grad_(True)
    P_ref = ref_compute(A_ref, n=n, num_iters=num_iters)
    grad_A_ref, = torch.autograd.grad(P_ref, A_ref, grad_outputs=grad_P)

    # TileLang backward
    jit_bwd = sinkhorn_knopp_bwd(n=n, num_iters=num_iters)
    # print(jit_bwd.get_kernel_source())
    grad_A_tl = jit_bwd(A, grad_P)  # A is saved input, grad_P is upstream

    torch.testing.assert_close(grad_A_tl, grad_A_ref, atol=1e-5, rtol=1e-5)
    print("Backward kernel passed!")
    
    from tilelang.profiler import do_bench
    def run_bwd():
        return jit_bwd(A, grad_P)
    bwd_latency = do_bench(run_bwd, warmup=10, rep=100)
    print(f"Backward latency: {bwd_latency} ms")
    A_ref = A.clone().detach().requires_grad_(True)
    P_ref = ref_compute(A_ref, n=n, num_iters=num_iters)
    def ref_bwd():
        P_ref.backward(grad_P, retain_graph=True)
        return grad_A_ref
    ref_latency = do_bench(ref_bwd, warmup=10, rep=100)
    print(f"Reference backward latency: {ref_latency} ms")
    
    
if __name__ == "__main__":
    test_bwd(1)
    test_bwd(16)
    test_bwd(1024)
    test_bwd(16 * 1024)
    test_bwd(64 * 1024)
    test_bwd(128 * 1024)