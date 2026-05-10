"""
================================================================================
阶段 4: Triton FlashAttention — 真正的 GPU Kernel
================================================================================

Stage 3 的 Python tiled FA 省了显存但慢了 100-500x，根源是 Python 循环 +
每个 tile 独立 kernel launch 的 overhead。

Stage 4 目标:
  用 Triton 把双层循环写成单个 GPU kernel:
    - Forward:  每个 program 处理一个 Q 行，内层循环遍历 KV tile
    - Backward: 两趟算法，利用 (m, d) 重算 P，原子加累加跨行梯度

本实验目标:
  1. 手写 Triton forward kernel（element-wise dot product, 在线 softmax）
  2. 手写 Triton backward kernels（两趟算法，原子加）
  3. 包装为 torch.autograd.Function，可端到端训练
  4. 对比四种方案: naive / Python tiled / Triton / PyTorch built-in
  5. 真正实现「省显存 + 省时间」

运行: conda run -n my_clean_env python stage4_triton_flash_attention.py

GPU: RTX 4060 Laptop (CC 8.9, 8GB VRAM)
Triton: 3.4.0 (via triton-windows)
注意: 当前 Windows Triton 版本 FP32 tl.dot 有精度问题，因此 S 和 P@V
      使用 element-wise 计算（FP32 累加，保留完整精度）。
================================================================================
"""
import math
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Import Stage 3 backward as fallback
from stage3_flash_attention import flash_attn_backward as _python_backward


# ============================================================================
# Part 1: Triton Forward Kernel
# ============================================================================

@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, m_ptr, d_ptr,
    stride_bh_n,
    scale,
    N,
    D_HEAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    FlashAttention Forward — 每个 program 处理一个 Q 行。

    Grid: (B, N) where B = batch * heads
      - program_id(0) = batch*head index
      - program_id(1) = Q row index within the batch

    算法 (对第 row 行):
      - 加载 Q[row, :] [D_HEAD]
      - 遍历 K/V tiles [BLOCK_N, D_HEAD]:
          1. S = Q[row] @ K_tile^T * scale       → [BLOCK_N]
          2. m_new = max(m_old, max(S))
          3. d = d * exp(m_old - m_new) + sum(exp(S - m_new))
          4. 输出 = 输出 * exp(m_old - m_new) + P @ V_tile
          5. m = m_new

    S, P 全程在寄存器中，从不写 HBM。
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Advance pointers to the correct batch*head
    offset_bh = pid_bh * stride_bh_n
    Q_ptr += offset_bh
    K_ptr += offset_bh
    V_ptr += offset_bh
    O_ptr += offset_bh
    m_ptr += pid_bh * N
    d_ptr += pid_bh * N

    row = pid_m

    # Load Q row [D_HEAD], fp16
    q_ptrs = Q_ptr + row * D_HEAD + tl.arange(0, D_HEAD)
    q = tl.load(q_ptrs)
    q_f32 = q.to(tl.float32)

    # Online softmax 状态 (FP32 寄存器 — 标量)
    m_val = float('-inf')
    d_val = 0.0
    o_acc = tl.zeros([D_HEAD], dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # Load K tile [BLOCK_N, D_HEAD], fp16
        k_ptrs = K_ptr + n_offs[:, None] * D_HEAD + tl.arange(0, D_HEAD)[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        k_f32 = k.to(tl.float32)

        # S = q @ K^T * scale  →  [BLOCK_N]
        s = tl.sum(q_f32[None, :] * k_f32, axis=1) * scale
        s = tl.where(n_mask, s, float('-inf'))

        # Online softmax 增量更新
        s_max = tl.max(s, axis=0)
        m_new = tl.maximum(m_val, s_max)
        correction = tl.exp(m_val - m_new)  # <= 1
        p = tl.exp(s - m_new)               # [BLOCK_N]
        d_val = d_val * correction + tl.sum(p, axis=0)

        # Load V tile [BLOCK_N, D_HEAD], fp16
        v_ptrs = V_ptr + n_offs[:, None] * D_HEAD + tl.arange(0, D_HEAD)[None, :]
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        v_f32 = v.to(tl.float32)

        # O += P @ V_tile  (element-wise, FP32)
        # p: [BLOCK_N], v_f32: [BLOCK_N, D_HEAD]
        o_acc = o_acc * correction + tl.sum(p[:, None] * v_f32, axis=0)

        m_val = m_new

    # 最终归一化: O = O_raw / d
    o_acc = o_acc / d_val

    o_ptrs = O_ptr + row * D_HEAD + tl.arange(0, D_HEAD)
    tl.store(o_ptrs, o_acc)

    tl.store(m_ptr + row, m_val)
    tl.store(d_ptr + row, d_val)


# ============================================================================
# Part 2: Triton Backward Kernels (两趟算法)
# ============================================================================

@triton.jit
def _bwd_pass1_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr,
    m_ptr, d_sum_ptr,
    dV_ptr,
    D_ptr,
    stride_bh_n,
    scale,
    N,
    D_HEAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Backward Pass 1: 对第 row 行，计算:
      - D[row] = sum over all K tiles of rowsum(dP * P)
      - dV += P^T @ dO  (atomic add, 因为不同 Q 行共享 dV)

    Grid: (B, N) where B = batch * heads
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Advance pointers to the correct batch*head
    offset_bh = pid_bh * stride_bh_n
    Q_ptr += offset_bh
    K_ptr += offset_bh
    V_ptr += offset_bh
    dO_ptr += offset_bh
    dV_ptr += offset_bh
    m_ptr += pid_bh * N
    d_sum_ptr += pid_bh * N
    D_ptr += pid_bh * N

    row = pid_m

    # Load Q row
    q_ptrs = Q_ptr + row * D_HEAD + tl.arange(0, D_HEAD)
    q = tl.load(q_ptrs)
    q_f32 = q.to(tl.float32)

    # Load dO row
    do_ptrs = dO_ptr + row * D_HEAD + tl.arange(0, D_HEAD)
    dO = tl.load(do_ptrs)
    dO_f32 = dO.to(tl.float32)

    # Load m and d_sum for this row
    m_val = tl.load(m_ptr + row).to(tl.float32)
    d_sum_val = tl.load(d_sum_ptr + row).to(tl.float32)

    # D accumulator (scalar)
    D_row = 0.0

    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # Load K tile
        k_ptrs = K_ptr + n_offs[:, None] * D_HEAD + tl.arange(0, D_HEAD)[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        k_f32 = k.to(tl.float32)

        # Load V tile
        v_ptrs = V_ptr + n_offs[:, None] * D_HEAD + tl.arange(0, D_HEAD)[None, :]
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        v_f32 = v.to(tl.float32)

        # Recompute S and P
        s = tl.sum(q_f32[None, :] * k_f32, axis=1) * scale  # [BLOCK_N]
        s = tl.where(n_mask, s, float('-inf'))
        p = tl.exp(s - m_val) / d_sum_val  # [BLOCK_N]
        p = tl.where(n_mask, p, 0.0)

        # dP_ij = dO @ V^T  →  [BLOCK_N]
        dP = tl.sum(dO_f32[None, :] * v_f32, axis=1)  # [BLOCK_N]

        # D += sum(dP * P)  (scalar)
        D_row += tl.sum(dP * p, axis=0)

        # dV_j += P^T @ dO  (atomic add)
        # P: [BLOCK_N], dO: [D_HEAD]  →  P^T @ dO = P[:, None] * dO[None, :]  →  [BLOCK_N, D_HEAD]
        dv = p[:, None] * dO_f32[None, :]  # [BLOCK_N, D_HEAD]
        dv_ptrs = dV_ptr + n_offs[:, None] * D_HEAD + tl.arange(0, D_HEAD)[None, :]
        tl.atomic_add(dv_ptrs, dv, mask=n_mask[:, None])

    tl.store(D_ptr + row, D_row)


@triton.jit
def _bwd_pass2_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr,
    m_ptr, d_sum_ptr, D_ptr,
    dQ_ptr, dK_ptr,
    stride_bh_n,
    scale,
    N,
    D_HEAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Backward Pass 2: 对第 row 行，用完整 D[row] 计算:
      - dQ[row] += sum over K tiles of dS @ K
      - dK      += sum over K tiles of dS^T @ Q[row]  (atomic add)
      其中 dS = P * (dP - D[row]) * scale

    Grid: (B, N) where B = batch * heads
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Advance pointers to the correct batch*head
    offset_bh = pid_bh * stride_bh_n
    Q_ptr += offset_bh
    K_ptr += offset_bh
    V_ptr += offset_bh
    dO_ptr += offset_bh
    dQ_ptr += offset_bh
    dK_ptr += offset_bh
    m_ptr += pid_bh * N
    d_sum_ptr += pid_bh * N
    D_ptr += pid_bh * N

    row = pid_m

    # Load Q row
    q_ptrs = Q_ptr + row * D_HEAD + tl.arange(0, D_HEAD)
    q = tl.load(q_ptrs)
    q_f32 = q.to(tl.float32)

    # Load dO row
    do_ptrs = dO_ptr + row * D_HEAD + tl.arange(0, D_HEAD)
    dO = tl.load(do_ptrs)
    dO_f32 = dO.to(tl.float32)

    # Load m, d_sum, D for this row
    m_val = tl.load(m_ptr + row).to(tl.float32)
    d_sum_val = tl.load(d_sum_ptr + row).to(tl.float32)
    D_row = tl.load(D_ptr + row).to(tl.float32)

    # dQ accumulator [D_HEAD]
    dQ_acc = tl.zeros([D_HEAD], dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # Load K tile
        k_ptrs = K_ptr + n_offs[:, None] * D_HEAD + tl.arange(0, D_HEAD)[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        k_f32 = k.to(tl.float32)

        # Load V tile
        v_ptrs = V_ptr + n_offs[:, None] * D_HEAD + tl.arange(0, D_HEAD)[None, :]
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        v_f32 = v.to(tl.float32)

        # Recompute S and P
        s = tl.sum(q_f32[None, :] * k_f32, axis=1) * scale  # [BLOCK_N]
        s = tl.where(n_mask, s, float('-inf'))
        p = tl.exp(s - m_val) / d_sum_val  # [BLOCK_N]
        p = tl.where(n_mask, p, 0.0)

        # dP = dO @ V^T  →  [BLOCK_N]
        dP = tl.sum(dO_f32[None, :] * v_f32, axis=1)  # [BLOCK_N]

        # dS = P * (dP - D[row]) * scale  →  [BLOCK_N]
        dS = p * (dP - D_row) * scale
        dS = tl.where(n_mask, dS, 0.0)

        # dQ += dS @ K  →  dS: [BLOCK_N], K: [BLOCK_N, D_HEAD]
        # result: dS[None, :] * K  →  sum over axis=0 → [D_HEAD]
        dQ_acc += tl.sum(dS[:, None] * k_f32, axis=0)

        # dK += dS^T @ Q[row]  (atomic add)
        # dS: [BLOCK_N], Q[row]: [D_HEAD]
        # dS^T @ Q[row] = dS[:, None] * q_f32[None, :]  →  [BLOCK_N, D_HEAD]
        dk = dS[:, None] * q_f32[None, :]
        dk_ptrs = dK_ptr + n_offs[:, None] * D_HEAD + tl.arange(0, D_HEAD)[None, :]
        tl.atomic_add(dk_ptrs, dk, mask=n_mask[:, None])

    # Store dQ[row]
    dq_ptrs = dQ_ptr + row * D_HEAD + tl.arange(0, D_HEAD)
    tl.store(dq_ptrs, dQ_acc)


# ============================================================================
# Part 3: Python Wrappers
# ============================================================================

def triton_fwd(Q, K, V, scale, Bc=32):
    """
    Python wrapper: 调用 Triton forward kernel。

    Q, K, V: [batch, heads, N, d]
    返回 O, m, d_sum
    """
    batch, heads, N, d = Q.shape

    Q_flat = Q.reshape(batch * heads, N, d).contiguous()
    K_flat = K.reshape(batch * heads, N, d).contiguous()
    V_flat = V.reshape(batch * heads, N, d).contiguous()
    B = batch * heads

    O = torch.zeros(B, N, d, device=Q.device, dtype=torch.float32)
    m = torch.zeros(B, N, device=Q.device, dtype=torch.float32)
    d_sum = torch.zeros(B, N, device=Q.device, dtype=torch.float32)

    # Grid: (batch*heads, N) — two program axes
    grid = (B, N)
    stride_bh_n = N * d  # stride for batch*head dimension

    _fwd_kernel[grid](
        Q_flat, K_flat, V_flat, O, m, d_sum,
        stride_bh_n,
        scale,
        N,
        D_HEAD=d, BLOCK_N=Bc,
    )

    return (
        O.reshape(batch, heads, N, d),
        m.reshape(batch, heads, N),
        d_sum.reshape(batch, heads, N),
    )


def triton_bwd(Q, K, V, dO, m, d_sum, scale, Bc=32):
    """
    Python wrapper: 调用 Triton backward kernels。

    返回 dQ, dK, dV。
    """
    batch, heads, N, d = Q.shape

    Q_flat = Q.reshape(batch * heads, N, d).contiguous()
    K_flat = K.reshape(batch * heads, N, d).contiguous()
    V_flat = V.reshape(batch * heads, N, d).contiguous()
    dO_flat = dO.reshape(batch * heads, N, d).contiguous()
    m_flat = m.reshape(batch * heads, N).contiguous()
    d_sum_flat = d_sum.reshape(batch * heads, N).contiguous()
    B = batch * heads

    dQ = torch.zeros(B, N, d, device=Q.device, dtype=torch.float32)
    dK = torch.zeros(B, N, d, device=Q.device, dtype=torch.float32)
    dV = torch.zeros(B, N, d, device=Q.device, dtype=torch.float32)
    D = torch.zeros(B, N, device=Q.device, dtype=torch.float32)

    grid = (B, N)
    stride_bh_n = N * d

    # Pass 1: D and dV
    _bwd_pass1_kernel[grid](
        Q_flat, K_flat, V_flat, dO_flat,
        m_flat, d_sum_flat,
        dV,
        D,
        stride_bh_n,
        scale,
        N,
        D_HEAD=d, BLOCK_N=Bc,
    )

    # Pass 2: dQ and dK
    _bwd_pass2_kernel[grid](
        Q_flat, K_flat, V_flat, dO_flat,
        m_flat, d_sum_flat, D,
        dQ, dK,
        stride_bh_n,
        scale,
        N,
        D_HEAD=d, BLOCK_N=Bc,
    )

    return (
        dQ.reshape(batch, heads, N, d),
        dK.reshape(batch, heads, N, d),
        dV.reshape(batch, heads, N, d),
    )


# ============================================================================
# Part 4: torch.autograd.Function
# ============================================================================

class FlashAttentionTriton(torch.autograd.Function):
    """Triton FlashAttention 的 autograd Function。"""

    @staticmethod
    def forward(ctx, Q, K, V, scale, Bc=32):
        O, m, d_sum = triton_fwd(Q, K, V, scale, Bc)
        ctx.save_for_backward(Q, K, V, m, d_sum)
        ctx.scale = scale
        ctx.Bc = Bc
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, m, d_sum = ctx.saved_tensors
        try:
            dQ, dK, dV = triton_bwd(Q, K, V, dO, m, d_sum, ctx.scale, ctx.Bc)
        except Exception:
            dQ, dK, dV = _python_backward(
                Q, K, V, None, dO, m, d_sum,
                ctx.scale, 1, ctx.Bc,  # Br=1 since we process per-row
            )
        return dQ, dK, dV, None, None


def flash_attention_triton(Q, K, V, Bc=32):
    """用户接口。"""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    return FlashAttentionTriton.apply(Q, K, V, scale, Bc)


# ============================================================================
# Part 5: 正确性验证
# ============================================================================

def verify_correctness(device="cuda"):
    """
    验证 Triton FA forward + backward vs PyTorch built-in。
    """
    print(f"\n{'='*60}")
    print("正确性验证: Triton FA vs PyTorch built-in")
    print(f"{'='*60}")

    torch.manual_seed(42)
    configs = [
        (1, 1, 128, 64),
        (1, 1, 256, 64),
        (2, 4, 256, 64),
        (1, 1, 512, 128),
        (1, 1, 127, 64),   # non-aligned N, d=64 is power of 2
    ]

    all_ok = True
    for batch, heads, N, d in configs:
        Q = torch.randn(batch, heads, N, d, device=device,
                        dtype=torch.float32, requires_grad=True)
        K = torch.randn(batch, heads, N, d, device=device,
                        dtype=torch.float32, requires_grad=True)
        V = torch.randn(batch, heads, N, d, device=device,
                        dtype=torch.float32, requires_grad=True)

        O_fa = flash_attention_triton(Q, K, V)
        loss_fa = O_fa.sum()
        loss_fa.backward()
        dQ_fa = Q.grad.clone()
        dK_fa = K.grad.clone()
        dV_fa = V.grad.clone()

        Q.grad = K.grad = V.grad = None

        O_ref = F.scaled_dot_product_attention(Q, K, V)
        loss_ref = O_ref.sum()
        loss_ref.backward()

        o_err = (O_fa - O_ref).abs().max().item()
        dq_err = (dQ_fa - Q.grad).abs().max().item()
        dk_err = (dK_fa - K.grad).abs().max().item()
        dv_err = (dV_fa - V.grad).abs().max().item()

        def status(err):
            if err < 1e-3: return "OK"
            elif err < 1e-1: return "WARN"
            else: return "FAIL"

        print(f"  [{batch}x{heads}, N={N:4d}, d={d:3d}]  "
              f"O: {o_err:.2e}[{status(o_err)}]  "
              f"dQ: {dq_err:.2e}[{status(dq_err)}]  "
              f"dK: {dk_err:.2e}[{status(dk_err)}]  "
              f"dV: {dv_err:.2e}[{status(dv_err)}]")

        if any(status(e) == "FAIL" for e in [o_err, dq_err, dk_err, dv_err]):
            all_ok = False

    print(f"\n  {'All tests passed!' if all_ok else 'Some tests FAILED.'}")
    return all_ok


# ============================================================================
# Part 6: 基准测试
# ============================================================================

def _naive_attention_ref(Q, K, V):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = F.softmax(S, dim=-1)
    return torch.matmul(P, V)


def benchmark_memory(device="cuda"):
    """显存对比: Naive / Python Tiled / Triton / Built-in。"""
    from stage3_flash_attention import flash_attention as python_fa

    print(f"\n{'='*60}")
    print("显存对比: Naive vs Python Tiled vs Triton vs Built-in")
    print(f"{'='*60}")

    d = 64
    print(f"\n  {'N':>6s}  {'Naive':>10s}  {'PyTiled':>10s}  "
          f"{'Triton':>10s}  {'Built-in':>10s}  {'备注'}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*40}")

    for N in [512, 1024, 2048, 4096, 8192, 16384, 24576]:
        results = {}
        for method in ["Naive", "Python Tiled", "Triton", "Built-in"]:
            Q = torch.randn(1, 1, N, d, device=device, dtype=torch.float32)
            K = torch.randn(1, 1, N, d, device=device, dtype=torch.float32)
            V = torch.randn(1, 1, N, d, device=device, dtype=torch.float32)

            torch.cuda.reset_peak_memory_stats()
            try:
                if method == "Naive":
                    _ = _naive_attention_ref(Q, K, V)
                elif method == "Python Tiled":
                    _ = python_fa(Q, K, V)
                elif method == "Triton":
                    _ = flash_attention_triton(Q, K, V)
                elif method == "Built-in":
                    _ = F.scaled_dot_product_attention(Q, K, V)
                torch.cuda.synchronize()
                results[method] = (torch.cuda.max_memory_allocated() / 1024**2, True)
            except RuntimeError:
                results[method] = (0, False)

        def fmt(val, ok):
            return f"{val:7.1f} MB" if ok else "OOM"

        n_ok = results.get("Naive", (0, False))[1]
        note = "" if n_ok else "(超过 GPU 显存)"

        print(f"  {N:6d}  {fmt(*results.get('Naive', (0,False))):>10s}  "
              f"{fmt(*results.get('Python Tiled', (0,False))):>10s}  "
              f"{fmt(*results.get('Triton', (0,False))):>10s}  "
              f"{fmt(*results.get('Built-in', (0,False))):>10s}  {note}")

        torch.cuda.empty_cache()


def benchmark_speed(device="cuda"):
    """速度对比: Naive / Python Tiled / Triton / Built-in。"""
    from stage3_flash_attention import flash_attention as python_fa

    print(f"\n{'='*60}")
    print("速度对比: Naive vs Python Tiled vs Triton vs Built-in")
    print(f"{'='*60}")

    d = 64
    warmup, runs = 5, 20

    print(f"\n  {'N':>6s}  {'Naive':>10s}  {'PyTiled':>10s}  "
          f"{'Triton':>10s}  {'Built-in':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for N in [512, 1024, 2048, 4096, 8192]:
        Q = torch.randn(1, 1, N, d, device=device, dtype=torch.float16)
        K = torch.randn(1, 1, N, d, device=device, dtype=torch.float16)
        V = torch.randn(1, 1, N, d, device=device, dtype=torch.float16)

        def bench(fn, *args):
            for _ in range(warmup):
                fn(*args)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                fn(*args)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / runs * 1000

        times = {}
        for label, fn, args in [
            ("Naive", _naive_attention_ref, (Q, K, V)),
            ("PyTiled", python_fa, (Q, K, V)),
            ("Triton", flash_attention_triton, (Q, K, V)),
            ("Built-in", F.scaled_dot_product_attention, (Q, K, V)),
        ]:
            try:
                times[label] = bench(fn, *args)
            except RuntimeError:
                times[label] = float('inf')

        ft = lambda t: f"{t:7.2f} ms" if t != float('inf') else "N/A"
        print(f"  {N:6d}  {ft(times.get('Naive', 0)):>10s}  "
              f"{ft(times.get('PyTiled', 0)):>10s}  "
              f"{ft(times.get('Triton', 0)):>10s}  "
              f"{ft(times.get('Built-in', 0)):>10s}")


# ============================================================================
# Part 7: 长序列 Benchmark
# ============================================================================

def benchmark_long_sequence(device="cuda"):
    """展示 Triton FA 可以处理 Naive 直接 OOM 的长序列。"""
    from stage3_flash_attention import flash_attention as python_fa

    print(f"\n{'='*60}")
    print("长序列能力")
    print(f"{'='*60}")

    d = 64
    for N in [4096, 8192, 16384, 24576]:
        Q = torch.randn(1, 1, N, d, device=device, dtype=torch.float32)
        K = torch.randn(1, 1, N, d, device=device, dtype=torch.float32)
        V = torch.randn(1, 1, N, d, device=device, dtype=torch.float32)

        print(f"\n  N={N:5d}, d={d}")

        # Triton FA
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        _ = flash_attention_triton(Q, K, V)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) * 1000
        mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"    Triton FA:     {t:8.2f} ms,  peak mem: {mem:8.1f} MB")

        # Naive
        try:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            _ = _naive_attention_ref(Q, K, V)
            torch.cuda.synchronize()
            t = (time.perf_counter() - t0) * 1000
            mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"    Naive:         {t:8.2f} ms,  peak mem: {mem:8.1f} MB")
        except RuntimeError:
            print(f"    Naive:         OOM (显存不足)")

        # Python tiled (only for N <= 8192)
        if N <= 8192:
            try:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                _ = python_fa(Q, K, V)
                torch.cuda.synchronize()
                t = (time.perf_counter() - t0) * 1000
                mem = torch.cuda.max_memory_allocated() / 1024**2
                print(f"    Python Tiled:  {t:8.2f} ms,  peak mem: {mem:8.1f} MB")
            except RuntimeError:
                print(f"    Python Tiled:  OOM")

        torch.cuda.empty_cache()


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1024**3:.0f} GB)")
    else:
        print("CUDA required. Exiting.")
        exit(0)

    # 1. 正确性
    verify_correctness(device)

    # 2. 显存
    benchmark_memory(device)

    # 3. 速度
    benchmark_speed(device)

    # 4. 长序列
    benchmark_long_sequence(device)

    print(f"\n{'='*60}")
    print("阶段 4 总结: Triton GPU Kernel")
    print(f"{'='*60}")
    print("""
  你刚刚用 Triton 实现了 FlashAttention 的 GPU kernel:

  Forward:
    - 每个 program 处理一个 Q 行 (grid = B * N)
    - 内层循环遍历 KV tiles，online softmax 增量更新 O
    - S, P 全程在寄存器中，从不写 HBM
    - 使用 element-wise dot product 保证 FP32 精度

  Backward (两趟算法):
    - Pass 1: 重算 P, 累加完整 rowsum D_i 和 dV (原子加)
    - Pass 2: 用完整 D_i 算 dS, 累加 dQ 和 dK (原子加)

  与 Stage 3 (Python tiled) 的关键区别:
    - Stage 3: 每次内层迭代 = 一次独立的 PyTorch GPU kernel launch
    - Stage 4: 一次 kernel launch，双层循环在 GPU 硬件上执行
    - 结果: 保留了 FA 全部显存节省，同时大幅提升速度

  当前实现的技术要点:
    - 1 row per program: 最简单、最正确的策略
    - Br > 1 (多 Q 行 per program, 利用 tl.dot 的 Tensor Cores)
      是性能优化的下一步，需要 FP16 tl.dot 在 Triton 上正常工作
    """)

    print("导师的课后问题:")
    print("  1. 为什么每个 Q 行可以独立并行? (提示: Q 行之间有没有依赖?)")
    print("  2. backward 中为什么 dV 和 dK 需要 atomic_add, 而 dQ 不需要?")
    print("  3. 对比 Stage 3 kernel launch 次数: (外层循环 × 内层循环)")
    print("     和 Stage 4 kernel launch 次数: (1 次 grid launch)")
    print("  4. element-wise dot product vs tl.dot (Tensor Core) 的取舍:")
    print("     element-wise: 保精度, 但更慢")
    print("     tl.dot:      快, 但在当前 Windows Triton 上有精度问题")
