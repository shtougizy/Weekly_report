"""
================================================================================
阶段 1: Naive Scaled Dot-Product Attention — 理解瓶颈
================================================================================

FlashAttention 论文的核心洞察: 标准 Attention 不是 compute-bound，
而是 memory-bound。QK^T 这个 N×N 矩阵对长序列的显存带宽是灾难。

本实验目标:
  1. 手写标准 Attention，理解每一步的 shape 变化
  2. 用 PyTorch profiler 看到底哪里在读写 HBM
  3. 计算理论的 memory IO 量，和实际对比
  4. 理解为什么 "memory-bound" 是 FA 要解决的核心问题

运行: python stage1_naive_attention.py

导师提示:
  - 这个文件里每个函数都很短，不要跳读
  - 把每个 print 的输出都看一遍
  - 重点关注 `profile_attention` 的输出
================================================================================
"""
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch.profiler import ProfilerActivity, profile, record_function


# ============================================================================
# Part 1: 手写 Naive Attention
# ============================================================================

def naive_scaled_dot_product_attention(Q, K, V, scale=None):
    """
    标准 Scaled Dot-Product Attention 的显式计算。

    数学:
      Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

    Shape 变换:
      Q: [batch, heads, N, d]
      K: [batch, heads, N, d]
      V: [batch, heads, N, d]

    计算图 (假设 batch=1, heads=1):
      1. S = Q @ K^T               -> [N, N]   ← ← ← 瓶颈在这里！
      2. S = S / sqrt(d)           -> [N, N]   这个 N×N 矩阵要完整写到 HBM
      3. P = softmax(S, dim=-1)    -> [N, N]   再读出做 softmax
      4. O = P @ V                 -> [N, d]   再读出做矩阵乘

    参数:
      scale: 默认 1/sqrt(d)。可传入以支持其他缩放。

    返回:
      output: [batch, heads, N, d]
      intermediates: dict 包含中间结果，用于分析
    """
    batch, heads, N, d = Q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # Step 1: 计算 attention scores — 这是你全部显存焦虑的来源
    # Q @ K^T: [N, d] x [d, N] = [N, N]
    # 当 N=4096 时: 4096×4096×4bytes(FP32) = 64MB
    # 当 N=8192 时: 8192×8192×4bytes = 256MB
    # 当 N=16384 时: 16384×16384×4bytes = 1GB ← 一个矩阵就 1GB
    # 而且 softmax 还要再读一遍！这就是 memory-bound 的根源
    S = torch.matmul(Q, K.transpose(-2, -1))  # [batch, heads, N, N]

    # Step 2: Scale
    S = S * scale

    # Step 3: Softmax along the last dim (over keys)
    P = F.softmax(S, dim=-1)  # [batch, heads, N, N]

    # Step 4: Weighted sum over values
    O = torch.matmul(P, V)  # [batch, heads, N, d]

    return O, {"scores": S, "attention_weights": P}


def pytorch_builtin_attention(Q, K, V):
    """作为对照: PyTorch 内置的 scaled_dot_product_attention。

    当满足条件时 (如特定的序列长度、causal=False)，
    PyTorch 会自动调用 flash attention 或 memory-efficient attention。
    同时直接调用也可以作为正确性基准。
    """
    return F.scaled_dot_product_attention(Q, K, V), {}


# ============================================================================
# Part 2: 显存分析
# ============================================================================

def estimate_peak_memory(N, d, batch=1, heads=1, dtype_bytes=2):
    """
    理论估算 naive attention 的峰值显存占用 (不包括模型参数)。

    中间张量:
      S = Q @ K^T       -> batch * heads * N * N * dtype_bytes
      P = softmax(S)    -> batch * heads * N * N * dtype_bytes (原地计算的理想情况)
      O = P @ V         -> batch * heads * N * d * dtype_bytes

    加上输入:
      Q, K, V           -> 3 * batch * heads * N * d * dtype_bytes

    总峰值 ≈ batch * heads * (2 * N^2 + N*d 中间 + 3*N*d 输入)
           ≈ batch * heads * (2 * N^2)  当 N >> d 时
    """
    base = batch * heads * dtype_bytes
    input_mem = 3 * base * N * d        # Q, K, V
    scores_mem = base * N * N           # S
    attn_mem = base * N * N             # P (softmax 通常需要额外分配)
    output_mem = base * N * d           # O
    total = input_mem + scores_mem + attn_mem + output_mem

    return {
        "QKV": input_mem,
        "scores_S": scores_mem,
        "attn_P": attn_mem,
        "output_O": output_mem,
        "total": total,
        "total_GB": total / (1024**3),
    }


# ============================================================================
# Part 3: Profiling — 看清楚瓶颈
# ============================================================================

def profile_attention(N=2048, d=64, device="cuda"):
    """
    用 PyTorch Profiler 分析 naive attention 的时间线和内存访问。
    """
    Q = torch.randn(1, 1, N, d, device=device, dtype=torch.float16)
    K = torch.randn(1, 1, N, d, device=device, dtype=torch.float16)
    V = torch.randn(1, 1, N, d, device=device, dtype=torch.float16)

    # 预热 GPU
    for _ in range(5):
        _ = naive_scaled_dot_product_attention(Q, K, V)

    torch.cuda.synchronize()

    # Profile naive attention
    print(f"\n{'='*60}")
    print(f"Profiling Naive Attention: N={N}, d={d}, dtype=FP16")
    print(f"{'='*60}")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("naive_attention"):
            O, _ = naive_scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()

    # 打印关键事件
    print("\n--- CUDA Kernel 时间线 (按 GPU time 排序) ---")
    key_events = []
    for event in prof.key_averages():
        cuda_time = getattr(event, 'self_cuda_time_total', 0) or 0
        if cuda_time > 0:
            key_events.append((event, cuda_time))

    key_events.sort(key=lambda x: x[1], reverse=True)

    total_cuda_time = sum(ct for _, ct in key_events)
    for i, (event, cuda_time) in enumerate(key_events[:10]):
        pct = cuda_time / total_cuda_time * 100 if total_cuda_time > 0 else 0
        print(f"  {i+1}. {event.key:<50s} "
              f"GPU: {cuda_time/1000:8.3f} us  "
              f"({pct:5.1f}%)  "
              f"calls: {event.count}")

    # 打印内存统计
    print("\n--- GPU Memory 使用 ---")
    try:
        mem_stats = prof.key_averages()
        total_alloc = sum(
            getattr(e, "cuda_memory_usage", 0) or 0 for e in mem_stats
        )
        print(f"  Total allocated during profile: {total_alloc / 1024 / 1024:.1f} MB")
    except Exception:
        print("  (memory details not available)")

    # 理论估算
    print("\n--- 理论显存估算 ---")
    mem = estimate_peak_memory(N, d, dtype_bytes=2)  # FP16
    print(f"  Q,K,V:     {mem['QKV']/1024/1024:.1f} MB")
    print(f"  S (scores): {mem['scores_S']/1024/1024:.1f} MB  ← ← 这是O(N^2)项")
    print(f"  P (attn):   {mem['attn_P']/1024/1024:.1f} MB")
    print(f"  O (output): {mem['output_O']/1024/1024:.1f} MB")
    print(f"  Peak total: {mem['total_GB']:.3f} GB")
    print()
    print(f"  关键洞察: scores 矩阵是 {N}×{N} = {N*N:,} 个元素")
    print(f"  而 Q/K/V 只有 {N}×{d} = {N*d:,} 个元素")
    print(f"  scores / input ratio = {N*N / (N*d):.0f}x")

    # 打印 profile 文件路径供 Chrome trace 查看
    try:
        prof.export_chrome_trace("stage1_trace.json")
        print(f"\n  Chrome trace saved to: stage1_trace.json")
        print(f"  (在 Chrome 中打开 chrome://tracing 加载此文件)")
    except Exception:
        pass

    return O


# ============================================================================
# Part 4: Benchmark — 不同序列长度下的表现
# ============================================================================

def benchmark_sequence_lengths(device="cuda"):
    """
    对比不同序列长度下 naive attention 的时间和显存。

    预期:
      - 时间应该是 O(N^2) 增长
      - 显存应该很快爆炸 (OOM on long sequences)
    """
    d = 64
    seq_lengths = [512, 1024, 2048, 4096, 8192]

    print(f"\n{'='*60}")
    print(f"Benchmark: Naive Attention vs Sequence Length (d={d})")
    print(f"{'='*60}")
    print(f"{'N':>6s}  {'Time (ms)':>10s}  {'理论峰值(GB)':>12s}  {'备注'}")
    print(f"{'-'*6}  {'-'*10}  {'-'*12}  {'-'*20}")

    for N in seq_lengths:
        mem = estimate_peak_memory(N, d, dtype_bytes=2)

        # 如果理论峰值超过显存，直接跳过
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if mem["total_GB"] > gpu_mem * 0.8:
            print(f"{N:6d}  {'SKIP':>10s}  {mem['total_GB']:11.3f}  (超过 GPU 显存 {gpu_mem:.0f}GB)")
            continue

        try:
            Q = torch.randn(1, 1, N, d, device=device, dtype=torch.float16)
            K = torch.randn(1, 1, N, d, device=device, dtype=torch.float16)
            V = torch.randn(1, 1, N, d, device=device, dtype=torch.float16)

            # Warmup
            for _ in range(3):
                naive_scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()

            # Benchmark
            num_runs = 10
            t0 = time.perf_counter()
            for _ in range(num_runs):
                naive_scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / num_runs

            print(f"{N:6d}  {elapsed*1000:10.3f}  {mem['total_GB']:11.3f}  ")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{N:6d}  {'OOM':>10s}  {mem['total_GB']:11.3f}  (显存不足)")
            else:
                print(f"{N:6d}  {'ERROR':>10s}  {mem['total_GB']:11.3f}  ({str(e)[:40]})")
        except Exception as e:
            print(f"{N:6d}  {'ERROR':>10s}  {mem['total_GB']:11.3f}  ({str(e)[:40]})")


# ============================================================================
# Part 5: 正确性验证
# ============================================================================

def verify_correctness(device="cuda"):
    """验证 naive attention 和 PyTorch 内置实现的结果一致。"""
    N, d = 256, 64
    Q = torch.randn(1, 1, N, d, device=device, dtype=torch.float32)
    K = torch.randn(1, 1, N, d, device=device, dtype=torch.float32)
    V = torch.randn(1, 1, N, d, device=device, dtype=torch.float32)

    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)

    # Naive
    O_naive, _ = naive_scaled_dot_product_attention(Q, K, V)
    loss_naive = O_naive.sum()
    loss_naive.backward()
    dQ_naive = Q.grad.clone()
    dK_naive = K.grad.clone()
    dV_naive = V.grad.clone()

    Q.grad = K.grad = V.grad = None

    # PyTorch builtin (在 FP32 下通常不会触发 flash attention 的数值差异)
    O_ref, _ = pytorch_builtin_attention(Q, K, V)
    loss_ref = O_ref.sum()
    loss_ref.backward()
    dQ_ref = Q.grad.clone()
    dK_ref = K.grad.clone()
    dV_ref = V.grad.clone()

    # 比对
    def check(name, a, b):
        diff = (a - b).abs().max().item()
        rel_diff = diff / (b.abs().max().item() + 1e-8)
        status = "OK" if rel_diff < 1e-4 else "MISMATCH"
        print(f"  {name:12s}  max_abs_diff={diff:.2e}  rel_diff={rel_diff:.2e}  [{status}]")

    print(f"\n{'='*60}")
    print("正确性验证: Naive vs PyTorch built-in (FP32)")
    print(f"{'='*60}")
    check("output O", O_naive, O_ref)
    check("grad dQ", dQ_naive, dQ_ref)
    check("grad dK", dK_naive, dK_ref)
    check("grad dV", dV_naive, dV_ref)


# ============================================================================
# Part 6: 可视化 QK^T 矩阵
# ============================================================================

def visualize_attention_pattern(N=128, device="cuda"):
    """
    可视化 attention scores 矩阵。

    目的: 让你直观感受 N×N 矩阵到底长什么样。
    对于 causal attention, 矩阵是下三角的。
    """
    d = 64
    Q = torch.randn(1, 1, N, d, device=device)
    K = torch.randn(1, 1, N, d, device=device)

    scale = 1.0 / math.sqrt(d)
    S = (Q @ K.transpose(-2, -1)) * scale  # [1, 1, N, N]
    P = F.softmax(S, dim=-1)

    # 打印矩阵的一些统计信息
    print(f"\n{'='*60}")
    print(f"Attention 矩阵分析 (N={N})")
    print(f"{'='*60}")
    print(f"  Shape:    {P.shape}")
    print(f"  Min P:    {P.min().item():.6f}")
    print(f"  Max P:    {P.max().item():.6f}")
    print(f"  Mean P:   {P.mean().item():.6f}")

    # 检查每行的 softmax 和是否为 1
    row_sums = P.sum(dim=-1)
    print(f"  Row sum range: [{row_sums.min().item():.4f}, {row_sums.max().item():.4f}]")
    print(f"  (应该接近 1.0)")

    # 因果 mask 的影响 (如果是 causal)
    print(f"\n  矩阵内容示意 (前 8×8):")
    p_np = P[0, 0, :8, :8].cpu().numpy()
    for row in range(8):
        vals = " ".join(f"{p_np[row, col]:.4f}" for col in range(8))
        print(f"    row {row}: {vals}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")

    # 1. 正确性验证: 确保我们的 naive 实现是对的
    verify_correctness(device)

    # 2. Profiling: 看具体哪些操作在消耗 GPU 时间
    profile_attention(N=2048, d=64, device=device)

    # 3. Benchmark: 不同序列长度下的表现
    benchmark_sequence_lengths(device)

    # 4. 可视化: 看看 attention 矩阵长什么样
    visualize_attention_pattern(N=128, device=device)

    print(f"\n{'='*60}")
    print("阶段 1 总结:")
    print(f"{'='*60}")
    print("""
  标准 Attention 的核心问题是:
    - Q @ K^T 产生 [N, N] 矩阵，内存是 O(N^2)
    - 这个矩阵必须完整写入 HBM (Global Memory)
    - softmax 又要从 HBM 读出、计算、再写回
    - P @ V 再次读出

  FlashAttention 的目标:
    - 把 Q, K, V 切成小块 (tiles)
    - 每个块在 SRAM (shared memory) 内完成 S -> softmax -> O 的全过程
    - 只把最终 O 写回 HBM, 中间的 S 和 P 永远不写入 HBM
    - 所以用到了 "online softmax" — 这是我们阶段 2 要学的内容

  阶段 2 预告: 手写 Online Softmax
    - naive softmax: 3 趟遍历 (max -> exp -> normalize)
    - online softmax: 1 趟遍历，增量更新
    - 这是理解 FlashAttention forward 的关键数学技巧
    """)

    print("\n导师的课后问题 (请思考, 下节课我问你):")
    print("  1. N=4096, d=64 时，scores 矩阵占多少 MB?")
    print("  2. 为什么显存带宽是瓶颈，而不是计算? (提示: FLOPS vs HBM bandwidth)")
    print("  3. 如果 N=32768, 理论上需要多少显存? 你的 GPU 能装下吗?")
