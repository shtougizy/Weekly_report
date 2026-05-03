"""
================================================================================
阶段 2: Online Softmax — FlashAttention 的数学核心
================================================================================

问题: 标准 softmax 需要 3 趟遍历整个向量
  1. 找 max(x)
  2. sum(exp(x - max))
  3. exp(x - max) / sum

FlashAttention 按块加载 K/V，每加载一个新块，旧的 softmax 统计量
需要被"修正"——因为你可能在新块里发现更大的 max，导致之前所有
exp 值需要重新 scale。

Online softmax 解决了这个问题: 一趟遍历，增量更新。

本实验目标:
  1. 手写 naive safe softmax (3 趟遍历)
  2. 手写 online softmax (1 趟遍历)
  3. 验证两者数值一致
  4. 理解 "running max" 和 "running denominator" 的更新公式

关键公式 (这就是 FlashAttention forward 的核心):

  初始: m = -inf, d = 0

  对每个新块 x_block:
    m_new = max(m, max(x_block))
    d     = d * exp(m - m_new)     # 修正旧 sum: 旧 max < 新 max 时需要 rescale
             + sum(exp(x_block - m_new))   # 加上新块的贡献
    m     = m_new

  最终: softmax = exp(x - m) / d

  注意: exp(m - m_new) 当 m < m_new 时是 < 1 的缩放因子。
        当 m == m_new 时是 1。
        m 只会增大，不会减小 → 缩放因子 ≤ 1。

运行: python stage2_online_softmax.py
================================================================================
"""
import math

import torch
import torch.nn.functional as F


# ============================================================================
# Part 1: 三种 Softmax 实现
# ============================================================================

def naive_softmax(x, dim=-1):
    """
    最原始的 softmax——数值不安全 (overflow)。
    当 x 中有较大正值时, exp(x) 会爆 FP16/FP32。
    """
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def safe_softmax(x, dim=-1):
    """
    Safe softmax: 减去 max(x) 再 exp。
    这是教科书写法, 数值稳定。
    但遍历了 3 次: max, exp+sum, divide。
    """
    m = x.max(dim=dim, keepdim=True).values  # 第 1 趟: 找 max
    e = torch.exp(x - m)                      # 第 2 趟: exp
    return e / e.sum(dim=dim, keepdim=True)   # 第 3 趟: 归一化


def online_softmax(x, dim=-1, block_size=None):
    """
    Online softmax: 一趟遍历, 分块增量更新。

    增量更新规则:
      m_i = max(m_{i-1}, max(block_i))
      d_i = d_{i-1} * exp(m_{i-1} - m_i) + sum(exp(block_i - m_i))

    参数:
      block_size: 模拟 SRAM 限制, 每次只处理这么多元素。
                  None 表示一次性处理 (退化为 safe softmax)。
    """
    shape = x.shape
    # 把 dim 轴移到最后一维方便处理
    if dim != -1 and dim != len(shape) - 1:
        x = x.transpose(dim, -1)

    x_flat = x.reshape(-1, x.shape[-1])  # [batch, N]
    N = x_flat.shape[-1]

    if block_size is None or block_size >= N:
        # 退化为标准 safe softmax
        return safe_softmax(x, dim=-1).reshape(shape)

    # --- Online softmax: 分块处理 ---
    # 初始状态
    m = torch.full((x_flat.shape[0],), -float('inf'),
                   device=x.device, dtype=x.dtype)  # running max, shape [batch]
    d = torch.zeros(x_flat.shape[0], device=x.device, dtype=x.dtype)  # running denominator

    num_blocks = (N + block_size - 1) // block_size

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, N)
        block = x_flat[:, start:end]  # [batch, block_size]

        # 找当前块内的 max
        m_block = block.max(dim=-1).values  # [batch]

        # 全局 max 是否被更新?
        m_new = torch.maximum(m, m_block)  # [batch]

        # 修正旧的 denominator (如果 max 变了)
        # d = d * exp(m - m_new)  +  sum(exp(block - m_new))
        correction = torch.exp(m - m_new)          # [batch], ≤ 1
        d = d * correction                         # rescale old sum
        d = d + torch.exp(block - m_new.unsqueeze(-1)).sum(dim=-1)  # add new

        m = m_new

    # 最终归一化: 对每个元素用最终的 m 和 d
    # exp(x - m) / d
    result = torch.exp(x_flat - m.unsqueeze(-1)) / d.unsqueeze(-1)

    if dim != -1 and dim != len(shape) - 1:
        result = result.transpose(-1, dim)
    return result.reshape(shape)


def online_softmax_verbose(x, block_size=4):
    """
    同上, 但打印每一步的中间状态, 帮助理解算法过程。
    只对 1D 输入打印。
    """
    x = x.squeeze().float()  # 确保是 1D FP32
    N = x.shape[0]
    print(f"\n  Input: x = {x.numpy().round(3)}")
    print(f"  Block size: {block_size}\n")

    m = -float('inf')
    d = 0.0
    final_denom = None

    num_blocks = (N + block_size - 1) // block_size

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, N)
        block = x[start:end]

        m_block = block.max().item()
        m_new = max(m, m_block)

        # 旧分母的修正因子
        correction = math.exp(m - m_new)
        d_old = d
        d = d * correction + torch.exp(block - m_new).sum().item()

        print(f"  Block {b}: [{start}:{end}] = {block.numpy().round(3)}")
        print(f"    m_block={m_block:.3f}, m_new={m_new:.3f}")
        if m == -float('inf'):
            print(f"    correction=exp(-inf - {m_new:.3f})=0 (第一次, 旧 d=0)")
        else:
            print(f"    correction=exp({m:.3f} - {m_new:.3f})={correction:.4f}")
        print(f"    d: {d_old:.4f} -> {d:.4f}")
        print(f"    m: {m} -> {m_new}")
        print()

        m = m_new
        final_denom = d

    # 用最终 m 和 d 计算 softmax
    result = torch.exp(x - m) / d
    print(f"  Final: m={m:.3f}, d={d:.4f}")
    print(f"  Result: softmax(x) = {result.numpy().round(4)}")
    print(f"  Sum check: {result.sum().item():.6f} (should be 1.0)")

    return result


# ============================================================================
# Part 2: 验证与对比
# ============================================================================

def test_numerical_correctness():
    """验证 online softmax 和 PyTorch 标准实现数值一致。"""
    print(f"\n{'='*60}")
    print("数值正确性验证")
    print(f"{'='*60}")

    torch.manual_seed(42)

    test_configs = [
        (1, 32),
        (1, 128),
        (2, 16, 64),
        (4, 8, 32, 128),
    ]

    for shape in test_configs:
        x = torch.randn(*shape, dtype=torch.float32) * 3.0  # 大范围值, 测试数值稳定性

        # 三种实现
        ref = F.softmax(x, dim=-1)
        safe = safe_softmax(x, dim=-1)
        online = online_softmax(x, dim=-1, block_size=8)

        # 对比
        err_safe = (safe - ref).abs().max().item()
        err_online = (online - ref).abs().max().item()

        print(f"\n  Shape: {shape}, block_size=8")
        print(f"    safe   vs ref: max_err = {err_safe:.2e}")
        print(f"    online vs ref: max_err = {err_online:.2e}")

        # 验证每行和为 1
        row_sums = online.sum(dim=-1)
        if row_sums.dim() > 0:
            max_dev = (row_sums - 1.0).abs().max().item()
        else:
            max_dev = abs(row_sums.item() - 1.0)
        print(f"    row sum deviation: {max_dev:.2e} (should be ~0)")

        # 额外: 用 outlier 值测试数值稳定性
        x_stress = torch.tensor([[-100.0, 0.0, 100.0, 50.0, -50.0, 25.0, -25.0, 75.0]])
        ref_stress = F.softmax(x_stress, dim=-1)
        online_stress = online_softmax(x_stress, dim=-1, block_size=2)
        err_stress = (online_stress - ref_stress).abs().max().item()
        print(f"    stress test (extreme values): max_err = {err_stress:.2e}")


# ============================================================================
# Part 3: 可视化 Online Softmax 过程
# ============================================================================

def demo_online_softmax():
    """手算一个例子，展示每一步的数值变化。"""
    print(f"\n{'='*60}")
    print("Online Softmax 逐步演示")
    print(f"{'='*60}")

    x = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 5.0, 4.0])

    print(f"\n{'='*40}")
    print("参考: PyTorch 标准 softmax")
    print(f"{'='*40}")
    ref = F.softmax(x, dim=-1)
    for i, v in enumerate(ref):
        print(f"  x[{i}]={x[i]:.2f}  ->  softmax={v.item():.6f}")

    print(f"\n{'='*40}")
    print("Online Softmax (block_size=3)")
    print(f"{'='*40}")
    online_softmax_verbose(x, block_size=3)

    print(f"\n{'='*40}")
    print("Online Softmax (block_size=2) — 更小的块")
    print(f"{'='*40}")
    online_softmax_verbose(x, block_size=2)


# ============================================================================
# Part 4: 性能对比
# ============================================================================

def benchmark_softmax(device="cuda"):
    """对比 naive/safe/online softmax 在大向量上的时间。"""
    print(f"\n{'='*60}")
    print("性能对比 (GPU)")
    print(f"{'='*60}")

    N = 131072  # 128K 元素, 足够看到差异
    x = torch.randn(10000, N, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = F.softmax(x, dim=-1)
    torch.cuda.synchronize()

    # Benchmark
    import time

    def bench(fn, name, **kwargs):
        for _ in range(3):
            fn(x, **kwargs)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            fn(x, **kwargs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"  {name:30s} {elapsed_ms:8.3f} ms")

    bench(lambda x: F.softmax(x, dim=-1), "PyTorch F.softmax (built-in)")
    bench(safe_softmax, "safe_softmax (3-pass)")
    bench(lambda x: online_softmax(x, dim=-1, block_size=1024), "online_softmax (block=1024)")
    bench(lambda x: online_softmax(x, dim=-1, block_size=256), "online_softmax (block=256)")
    bench(lambda x: online_softmax(x, dim=-1, block_size=64), "online_softmax (block=64)")

    print("\n  注意: PyTorch 的 online 版本用 Python 循环, 比内置的慢很多。")
    print("  这不是公平对比 —— 目的是验证算法正确性, 不是拼性能。")
    print("  FlashAttention 的 CUDA kernel 才是真正快的地方。")


# ============================================================================
# Part 5: 从 Online Softmax 到 FlashAttention Forward 的思想跃迁
# ============================================================================

def flash_attention_sketch():
    """
    这个函数展示: 如果我们把 online softmax 的「分母增量更新」逻辑
    嵌入到 attention 计算中, 会得到什么?

    这就是 FlashAttention forward 的伪代码骨架。

    我们不在这里运行它 (阶段 3 会正式实现), 但你可以提前读一下结构。
    """
    print(f"\n{'='*60}")
    print("FlashAttention Forward 伪代码骨架 (阶段 3 预告)")
    print(f"{'='*60}")
    print("""
  # 伪代码:
  for i in range(0, N, Br):          # 遍历 Q 块 (外层循环)
      Qi = Q[i:i+Br]                 # Q 块 [Br, d]
      m = [-inf] * Br                # 这一行的 running max
      d = [0] * Br                   # 这一行的 running denominator
      Oi = [0] * Br * d              # 累加器, 对应输出块

      for j in range(0, N, Bc):      # 遍历 K,V 块 (内层循环)
          Kj = K[j:j+Bc]             # K 块 [Bc, d]
          Vj = V[j:j+Bc]             # V 块 [Bc, d]

          S = Qi @ Kj.T * scale      # [Br, Bc] — 局部 scores, 留在 SRAM!
          m_new = max(m, S.max)      # 更新 running max
          correction = exp(m - m_new)# 旧值的修正因子
          d = d * correction         # 修正旧 denominator
          P = exp(S - m_new)         # 局部 softmax 分子
          d = d + P.sum              # 加入新 denominator

          Oi = Oi * correction       # 修正旧输出!
          Oi = Oi + P @ Vj           # 累加新输出

          m = m_new

      O[i:i+Br] = Oi / d            # 最终归一化, 写回 HBM

  关键洞察:
    1. S 和 P 永远不写回 HBM — 它们在 SRAM 中产生、消费、丢弃
    2. Oi 和 d 需要被 "修正" 当新的 max 出现时
       — 这就是 online softmax 的 exp(m_old - m_new) 修正因子
    3. 外层循环可以并行 (不同 Q 的块之间没有依赖)
    4. 内层循环是顺序的 (因为 online softmax 状态的依赖)

  阶段 3 我们将用 PyTorch 实现这个完整的循环, 并验证正确性。
  """)


# ============================================================================
# Part 6: 手动推导 — 证明 Online Softmax 等价于标准 Softmax
# ============================================================================

def mathematical_derivation():
    """
    对数学推导感兴趣? 这里给出严谨的证明。
    """
    sep = "=" * 60
    print(f"""
{sep}
数学推导: 为什么 Online Softmax 是正确的?
{sep}

标准 softmax(x_1, ..., x_N):
  pi = exp(xi - m) / sum_j exp(xj - m)
  其中 m = max(x_1, ..., x_N)

分块: 将向量分成 K 块 B_1, B_2, ..., B_K

定义部分统计量:
  在块 k 结束时:
    m_k = max(所有 B_1..B_k 中的元素)
    d_k = sum_(j in B_1..B_k) exp(xj - m_k)

我们需要证明: 对最终的 m_K 和 d_K,
  softmax(xi) = exp(xi - m_K) / d_K  对所有 i
  = 标准 softmax 的结果

证明 (归纳法):

基础: k=1
  m_1 = max(B_1)
  d_1 = sum(exp(B_1 - m_1))
  正确

归纳步骤: 假设 (m_k, d_k) 对前 k 块正确
  处理块 B_(k+1):
    m_(k+1) = max(m_k, max(B_(k+1)))

    d_(k+1) = d_k * exp(m_k - m_(k+1))        <-- rescale 旧的
              + sum(exp(B_(k+1) - m_(k+1)))   <-- 加上新的

  如果 m_k == m_(k+1) (新块没有更大的 max):
    d_(k+1) = d_k + sum(exp(B_(k+1) - m_(k+1)))
    简单的累加

  如果 m_k < m_(k+1) (新块的 max 是全局最大):
    此时 exp(m_k - m_(k+1)) < 1
    d_(k+1) = d_k * exp(m_k - m_(k+1)) + sum(exp(B_(k+1) - m_(k+1)))

    注意到: d_k = sum(exp(B_(1..k) - m_k))
    所以 d_k * exp(m_k - m_(k+1)) = sum(exp(B_(1..k) - m_k + m_k - m_(k+1)))
                                   = sum(exp(B_(1..k) - m_(k+1)))
    即用新 m 重缩放旧总和
    d_(k+1) = sum(exp(B_(1..k+1) - m_(k+1)))

因此对任意 xi: softmax(xi) = exp(xi - m_K) / d_K 是标准 softmax.
""")
    print(f"""
核心 insight: exp(m_k - m_(k+1)) 这个修正因子之所以有效,
是因为 exp(a) * exp(b) = exp(a + b), 所以重缩放本质上是:
  "把旧的指数从相对旧 max 的偏移, 转换为相对新 max 的偏移"
""")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # 1. 数值验证: online softmax 结果正确吗?
    test_numerical_correctness()

    # 2. 逐步演示: 一步步看 online softmax 怎么算的
    demo_online_softmax()

    # 3. 数学推导
    mathematical_derivation()

    # 4. 与 FlashAttention 的联系
    flash_attention_sketch()

    # 5. 性能对比 (可选, 主要是感受一下 Python 循环的代价)
    if device == "cuda":
        benchmark_softmax(device)

    print(f"\n{'='*60}")
    print("阶段 2 总结: 你学会了什么?")
    print(f"{'='*60}")
    print("""
  1. Online softmax 用增量更新, 一趟遍历完成 softmax
  2. 核心公式: m_new = max(m_old, m_block)
               d_new = d_old * exp(m_old - m_new) + sum(exp(block - m_new))
  3. 这就是 FlashAttention 内层循环的数学基础
  4. 外层循环遍历 Q 块, 内层循环遍历 K/V 块, 每个内层迭代
     用 online softmax 增量更新 O 和统计量

  阶段 3 预告: 完整实现 Tiled FlashAttention (PyTorch)
    - 把 online softmax 嵌入 attention 计算
    - 实现 forward + backward
    - 验证对长序列的内存节省
  """)

    print("导师的课后问题:")
    print("  1. 如果 m_block == m_old (新块的值都不超过旧 max), correction 是多少?")
    print("  2. 为什么 m 只会增大不会减小? 这给了什么数值稳定性保证?")
    print("  3. 试着把 demo 里的 block_size 改成 1, 猜猜会发生什么?")
