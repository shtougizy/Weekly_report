# FlashAttention 复现实验周记


---

## Lab1：Naive Attention — 理解瓶颈在哪里

### 做了什么

实现了标准的 Scaled Dot-Product Attention，并通过 profiling 和 benchmark 分析了它的性能瓶颈。

核心代码非常简短：

```python
S = Q @ K.transpose(-2, -1)        # [N, N]  ← 万恶之源
S = S / sqrt(d)
P = softmax(S, dim=-1)
O = P @ V
```

### 关键发现

| 序列长度 N | 耗时 | S 矩阵大小 | 理论峰值显存 |
|-----------|------|-----------|-------------|
| 512 | 0.11 ms | 1 MB | 0.001 GB |
| 2048 | 0.12 ms | 8 MB | 0.017 GB |
| 4096 | 0.98 ms | 32 MB | 0.064 GB |
| 8192 | 3.77 ms | 128 MB | 0.254 GB |

在 N=2048, d=64 时，S 矩阵有 4,194,304 个元素，而 Q/K/V 总共只有 131,072 个元素。**中间矩阵是输入数据的 32 倍大**——这就是问题的根源。

### 学到的概念

1. **Memory-bound vs Compute-bound**：RTX 4060 的理论算力约 15 TFLOPS（FP16），显存带宽约 272 GB/s。硬件能提供的 ops/byte 约为 55，而 naive attention 只做到约 20 ops/byte——GPU 的大量时间在等数据从 HBM 搬过来，计算单元在空转。

2. **HBM vs SRAM**：GPU 的 Global Memory（HBM）容量大（8GB）但带宽有限；Shared Memory（SRAM）容量极小（~100KB/SM）但带宽极高。FlashAttention 的核心思想就是把数据搬到 SRAM 里算完，只把结果写回 HBM。

3. **O(N²) 的恐怖**：S = Q @ K^T 产生了 N×N 的中间矩阵。当 N=32768 时，仅这一个矩阵在 FP32 下就需要 4GB 显存，加上 softmax 和后续计算，峰值超过 8GB。而这个矩阵里的值，除了用于 softmax 归一化之外，**全都是写进去、读出来、然后扔掉的一次性数据**。

### 遇到的坑

- PyTorch 2.7 的 profiler API 中属性名从 `cuda_time_total` 变成了 `self_cuda_time_total`，文档滞后。
- Windows 终端下中文输出乱码，但不影响数值结果读取。

---

## Lab2：Online Softmax — FlashAttention 的数学核心

### 做了什么

实现了三种 softmax 并理解它们的递进关系：

| 版本 | 遍历次数 | 数值安全 | 能增量更新 |
|------|---------|---------|-----------|
| Naive softmax | 3 趟 | 否（overflow） | 否 |
| Safe softmax | 3 趟 | 是 | 否 |
| Online softmax | 1 趟 | 是 | **是** |

### 核心公式

Online softmax 的增量更新规则：

```
初始化: m = -inf, d = 0

对每个新块 x_block:
  m_new = max(m, max(x_block))
  d     = d * exp(m - m_new) + sum(exp(x_block - m_new))
  m     = m_new

最终:  softmax(x) = exp(x - m) / d
```

**最关键的一行是 `d * exp(m - m_new)`**。当新来的块出现了更大的 max 值时，`exp(m_old - m_new) < 1`，这个因子把之前所有的 exp 值「重新缩放」到以新 max 为基准的坐标系下。因为 `exp(a) * exp(b) = exp(a + b)`，这个重缩放在数学上是精确的。

### 逐步演示

用 x = [1, 2, 3, 0.5, 1.5, 2.5, 5, 4]，block_size=3：

```
Block 0: [1.0, 2.0, 3.0]
  m: -inf → 3.0    d: 0 → 1.503
  correction: 0（第一次迭代，"旧 d"为 0）

Block 1: [0.5, 1.5, 2.5]
  m: 3.0 → 3.0    d: 1.503 → 2.415
  correction: exp(3.0 - 3.0) = 1.0（max 没变，直接累加）

Block 2: [5.0, 4.0]
  m: 3.0 → 5.0    d: 2.415 → 1.695
  correction: exp(3.0 - 5.0) = 0.135（发现更大的 max！旧 d 被压缩）
  ↑ d 不增反降了！因为用更大的 m 重新缩放后，分母变小了
```

最终结果与 PyTorch 内置 softmax 完全一致（误差 < 1e-7）。

### 学到的概念

1. **Online algorithm**：能在一趟扫描中完成计算、且任意时刻的状态都是「可用的」（虽然精度随信息增多而提高）。这种算法天然适合流式数据和分块计算。

2. **为什么 m 只增不减**：因为 max 操作的性质——扫描更多元素只会让全局 max 增大或不变。这保证了 `exp(m_old - m_new) ≤ 1`，不会放大舍入误差，数值上非常稳定。

3. **这个公式就是 FlashAttention 的灵魂**：把 online softmax 的「分母增量更新」换成「输出矩阵 O 的增量更新」，就是 FlashAttention 的内层循环。

### 遇到的坑

- 在 f-string 中使用数学符号 `{B_1, B_2}` 被 Python 解析为 set literal，需要转义为 `{{...}}` 或改用普通字符串。

---

## Lab3：Tiled FlashAttention — PyTorch 完整实现

### 做了什么

将 online softmax 嵌入完整的 Attention 计算，实现了 forward + backward，并包装为 `torch.autograd.Function`，可以端到端训练。

### Forward 算法

```python
for i in range(0, N, Br):              # 外层: 遍历 Q 块
    Qi = Q[:, i:i+Br, :]
    mi = [-inf] * Br                    # running max
    di = [0] * Br                       # running denominator
    Oi = [0] * Br * d                   # 累加输出

    for j in range(0, N, Bc):           # 内层: 遍历 KV 块
        Kj, Vj = K[:, j:j+Bc], V[:, j:j+Bc]

        S = Qi @ Kj^T / sqrt(d)         # tile 内 scores (留在"SRAM")

        m_new = max(mi, S.max())        # 更新 running max
        correction = exp(mi - m_new)
        di = di * correction + exp(S - m_new).sum()  # 更新分母
        Oi = Oi * correction + exp(S - m_new) @ Vj   # 更新输出
        mi = m_new

    O[:, i:i+Br] = Oi / di              # 最终归一化，写回 HBM
```

核心要点：**S 矩阵和 P 矩阵只在 tile 内临时存在，从不写回 HBM**。全程只有 Q、K、V（输入）和 O（输出）在 HBM 上。

### Backward：最大的坑

Backward 是这次实验最困难的部分。最初的实现犯了一个关键错误：

**错误版本**（每个 tile 内独立做 softmax backward）：
```python
D = (dP * P).sum(dim=-1)           # 只对当前 tile 的 Bc 列求和
dS = P * (dP - D)                  # ← 错了！D 缺少跨 tile 的信息
```

这导致 dQ 和 dK 的梯度误差高达 0.15——完全不可用。

**正确版本**（两趟算法）：
```python
# Pass 1: 先算完整的 D（对整行所有 N 个 key 求和）
D_i = 0
for each KV tile j:
    P = exp(S - mi) / di
    dP = dOi @ Vj^T
    D_i += (dP * P).sum(dim=-1)    # 累加所有 tile 的和

# Pass 2: 用完整的 D_i 计算 dS
for each KV tile j:
    P = exp(S - mi) / di
    dP = dOi @ Vj^T
    dS = P * (dP - D_i) * scale    # 现在正确了
    dQ_i += dS @ Kj
    dK_j += dS^T @ Qi
```

原因是 softmax 的反向公式 $dS = P \odot (dP - \text{rowsum}(P \odot dP))$ 中的 rowsum 必须对**整行的所有 N 个位置**求和，不能只在 tile 内做。这是一个很容易犯的错误，也是理解 FlashAttention backward 的关键。

### 最终验证结果

**梯度正确性**（vs PyTorch 内置实现）：

| 配置 | O 误差 | dQ 误差 | dK 误差 | dV 误差 |
|------|--------|---------|---------|---------|
| N=128, d=64 | 6.6e-07 | 7.8e-07 | 1.4e-06 | 7.2e-07 |
| N=256, d=64 | 1.1e-06 | 8.9e-07 | 1.2e-06 | 4.8e-07 |
| N=512, d=128 | 6.6e-07 | 4.8e-07 | 1.2e-06 | 4.8e-07 |
| N=127, d=63（非整除） | 3.0e-07 | 3.6e-07 | 4.2e-07 | 2.4e-07 |

所有误差在 1e-6 ~ 1e-7 级别。

**显存节省**（与 naive attention 对比）：

| N | Naive | Tiled FA | 节省比例 |
|---|-------|----------|---------|
| 2048 | 63 MB | 55 MB | 12% |
| 4096 | 197 MB | 166 MB | 16% |
| 8192 | 730 MB | 603 MB | 17% |
| 16384 | 2596 MB | 547 MB | **79%** |
| 24576 | 6702 MB | 2093 MB | **69%** |

N=16384 时尤为明显——naive 的 S 矩阵（16384² × 4 bytes = 1GB）完全被 tiled FA 规避了。

### 速度：Python 循环的代价

| N | Tiled FA | Naive | 倍数 |
|---|----------|-------|------|
| 4096 | 1157 ms | 2.4 ms | 慢 480x |
| 8192 | 4530 ms | 8.7 ms | 慢 520x |
| 16384 | 16723 ms | 99 ms | 慢 168x |

Tiled FA 的 Python 实现比 naive 慢几百倍。原因不是算法不好，而是：
- Python 双重循环的开销
- 每个 tile 的 `torch.bmm` 都是一次独立的 GPU kernel launch（kernel launch overhead 通常是 5-10μs，几百个 tile 就是几毫秒的纯 overhead）
- 没有利用 shared memory（数据在 PyTorch tensor 之间反复拷贝）

**这恰恰说明了为什么 FlashAttention 必须写成 CUDA/Triton kernel**：只有在一个 kernel 内完成所有 tile 循环，才能同时获得「省显存」和「省时间」两个收益。

---

## 总结：从零到一复现 FlashAttention

### 技能树变化

| 实验前 | 实验后 |
|--------|--------|
| 知道 attention 公式 | 理解 memory-bound 的含义，能用 ops/byte 分析瓶颈 |
| 用过 softmax | 能手写 online softmax 并证明其正确性 |
| 用过 autograd | 能实现自定义 Function 的 forward + backward |
| 听说过 GPU 内存层次 | 理解 HBM vs SRAM 的区别和对 kernel 设计的影响 |
| 没用过 profiler | 会用 PyTorch profiler 分析 CUDA kernel 时间线 |

### 核心收获

1. **算法和系统是同一枚硬币的两面**。Online softmax 不仅是数学技巧，更是让分块计算成为可能的「算法-系统协同设计」。

2. **Backward 比 Forward 难一个数量级**。Forward 的 online softmax 增量逻辑直观易懂，但 backward 的「完整 rowsum 必须跨 tile 累加」这个约束，只有亲自动手写才会真正理解。

3. **Python 循环 + GPU kernel 的组合是性能杀手**。Tiled FA 省了显存但没省时间，根源在于 kernel launch overhead。下一步必须用 Triton/CUDA 把整个双层循环放进一个 kernel。

### 下一步

用 Triton 把双层循环写成 GPU kernel，利用 shared memory 真正实现「省显存 + 省时间」。

---


