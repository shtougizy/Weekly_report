# 阶段 4 实验笔记：Triton FlashAttention GPU Kernel



---

## 0. 这次到底要干什么

阶段 3 做完之后我有一个很拧巴的结论：我用 Python 双层循环 + `torch.bmm` 实现了分块 FlashAttention——**显存确实省了**（N=16384 时从 2596 MB 降到 547 MB），但**速度慢了 100-500 倍**。

导师说这个结果恰恰是最好的 motivation：它证明了算法本身是对的，但也证明了 Python 循环 + 每次迭代一个独立 GPU kernel launch 的组合是性能杀手。FlashAttention 之所以叫 "FlashAttention"，不是因为分块这个 idea 本身，而是因为 **Tri Dao 把这个 idea 写成了一个 CUDA kernel，让双层循环在 GPU 硬件上直接跑**。

所以阶段 4 的目标是：**用 Triton 把阶段 3 的 Python 双层循环写成 GPU kernel**。如果成功，我们应该同时获得"省显存"和"省时间"两个收益。

---

## 1. 写 Forward Kernel——出乎意料的顺利

### 1.1 从最简单的 mapping 开始

我们没有一上来就搞 BLOCK_M=128、Tensor Core 全开那种复杂版本。策略是先把最简单的事情做对：**一个 program 只处理一个 Q 行**。

这个 mapping 的优势是 online softmax 的状态可以全是标量：

```python
m_val = float('-inf')   # running max，标量
d_val = 0.0             # running 分母，标量
o_acc = tl.zeros([D_HEAD], dtype=tl.float32)  # 累加输出，向量
```

然后在 Python 那层，把整个双层循环翻译成 Triton 的 for loop：

```python
for n_start in range(0, N, BLOCK_N):
    # 加载 K tile
    k = tl.load(K_ptr + ...)
    k_f32 = k.to(tl.float32)

    # S = Q[row] @ K_tile^T * scale → [BLOCK_N]，全在寄存器里
    s = tl.sum(q_f32[None, :] * k_f32, axis=1) * scale

    # Online softmax 增量更新（和阶段 2 的公式一模一样）
    m_new = tl.maximum(m_val, tl.max(s, axis=0))
    correction = tl.exp(m_val - m_new)
    p = tl.exp(s - m_new)
    d_val = d_val * correction + tl.sum(p, axis=0)

    # 加载 V tile，更新输出累加器
    v = tl.load(V_ptr + ...)
    o_acc = o_acc * correction + tl.sum(p[:, None] * v_f32, axis=0)

# 循环结束，最终归一化
o_acc = o_acc / d_val
```

这基本上就是阶段 3 的内层循环逐行翻译成 Triton。**Triton 最让我舒服的一点是：它的 Python-like 语法让你感觉就在写普通 Python，但实际上它在编译 GPU 指令。** 不用学 CUDA 的 `<<<>>>` 语法，不用手动管理 shared memory，Triton 的 compiler 帮你做了这些。

### 1.2 第一次就跑通了

用 `_debug10.py` 测试，N=64/128/256/512/1024 全部通过，误差 ~2e-7。到这里我心态已经稳了——至少 forward 是对的。

但接下来开始进入 Bugland……

---

## 2. Bugland：BLOCK_M > 1 的幽灵 Bug

### 2.1 想要加速，结果全部失败

1-row-per-program 跑通之后，自然想试试 **BLOCK_M > 1**：让每个 program 一次处理比如 16 行或 32 行 Q，这样可以在 `S = Q @ K^T` 这一步用 `tl.dot`（Tensor Core 矩阵乘法），性能应该有数量级提升。

于是有了下面这些尝试：

| 调试文件 | 策略 | 结果 |
|---------|------|------|
| `_debug7.py` | BLOCK_M=32, S 用 `tl.dot(fp16)`, P@V 用 element-wise | 误差 **0.766**（完全不对） |
| `_debug8.py` | BLOCK_M=32, 全部 element-wise（不用 tl.dot） | 误差 **0.766**（同样的错误！） |
| `_debug9.py` | BLOCK_M=1, 但保留 BLOCK_M > 1 的 kernel 结构 | 通过 ✅ |

**_debug9 通过但 _debug8 失败**——这两个的区别只是 BLOCK_M 的值。**我排除了 tl.dot 的问题，排除了 element-wise 的问题，排除了 online softmax 公式的问题。唯一的变量就是 BLOCK_M。**

这意味着：在当前版本的 Windows Triton (3.4.0) 上，只要 BLOCK_M > 1，无论用什么方式计算 S 和 P@V，结果都是错的。而且误差值非常稳定——永远是 0.766。

**我花了很长时间才接受这个结论**——这不是我代码的 bug，是编译器/运行时的 bug。

### 2.2 另一个限制：FP32 tl.dot 精度错误

在 `_debug7.py` 的实验过程中，还发现了一个更隐蔽的问题：

```python
s = tl.dot(q_fp32, k_fp32.trans())  # FP32 tl.dot → 误差 0.01
s = tl.dot(q_fp16, k_fp16.trans())  # FP16 tl.dot → 误差 1e-6 ✓
```

FP32 的 `tl.dot` 在数学上应该精度更高，但在这个 Triton 版本上它是**错的**。FP16 反而是对的。这又是一个工具链 bug。

### 2.3 接受现实，回到 1-row-per-program

所以最终的工程决策是：**在当前工具链约束下，走 1 row per program + element-wise dot product 路线。** 这不是最优方案，但是正确的方案。

这个过程让我学到了一件事：**做系统研究时，区分"算法 bug"和"工具链 bug"是一项很重要的能力**。如果你花三天时间 debug 一个编译器 bug，那你浪费了三天。正确做法是用最小可复现案例确认问题，记录下来，绕过它，继续推进。

---

## 3. Backward——两趟算法再次出场

### 3.1 阶段 3 的教训不能忘

阶段 3 我犯过一个关键错误：在 backward 里对每个 tile 独立做 softmax gradient（D 只对 tile 内求和），导致梯度完全错误。

正确的做法是两趟算法：
- **Pass 1**：遍历所有 KV tiles，累加出**完整的** D[row]（对整行 N 个位置求和）
- **Pass 2**：用完整 D[row] 计算 dS，然后更新 dQ 和 dK

这个教训在写 Triton kernel 时直接沿用了。

### 3.2 atomic_add 的 tradeoff

写 backward kernel 时遇到一个设计选择：

- **dQ[row]** 只被 row 所在的那个 program 写入 → 不需要 atomic_add，直接 store
- **dK** 和 **dV** 会被多个 Q 行的 program 同时更新 → **必须 atomic_add**

```python
# Pass 2 里更新 dK——多个 Q 行可能同时写同一个位置
dk = dS[:, None] * q_f32[None, :]   # [BLOCK_N, D_HEAD]
tl.atomic_add(dK_ptr + ..., dk)      # ← 原子操作

# 但 dQ 直接累加就行——没有竞争
dQ_acc += tl.sum(dS[:, None] * k_f32, axis=0)
# 最后一次性写回
tl.store(dQ_ptr + ..., dQ_acc)
```

这个设计选择反映了一个更 general 的原则：**根据数据依赖关系决定是否需要同步**。不需要同步的地方不加锁，需要同步的地方必须保证正确性。

### 3.3 Batch 维度指针偏移的那个 bug

写完 backward 之后跑多 batch 测试 (2, 4, 256, 64) 时发现不对劲。检查了很久才发现问题：

最初的设计是 `grid = (B * N,)`——一维 grid，每个 program 拿到一个全局的 `row` 索引。但这个 scheme 下，我没法区分"row=257 是 batch 0 的第 257 行"还是"batch 1 的第 1 行"——K 和 V 的指针没有正确地偏移到对应的 batch。

修复方案是改用二维 grid：

```python
grid = (B, N)

pid_bh = tl.program_id(0)  # 哪个 batch*head
pid_m = tl.program_id(1)   # 该 batch 内的哪一行

# 把所有指针推到正确的 batch*head 位置
offset = pid_bh * stride_bh_n   # stride_bh_n = N * d
Q_ptr += offset
K_ptr += offset
# ... 所有指针都要偏移
```

这个 bug 让我意识到：**在 GPU kernel 里做指针运算，你必须对内存布局有非常清晰的心智模型**。一个 `*ptr` 偏移错了，整个结果就是垃圾——而且不会报错，只会悄悄产出错误的数值。

---

## 4. 最终结果

### 4.1 正确性

```
[1x1, N= 128, d= 64]  O: 6.26e-07[OK]  dQ: 7.15e-07[OK]  dK: 1.31e-06[OK]  dV: 8.34e-07[OK]
[1x1, N= 256, d= 64]  O: 1.07e-06[OK]  dQ: 8.34e-07[OK]  dK: 1.31e-06[OK]  dV: 9.54e-07[OK]
[2x4, N= 256, d= 64]  O: 1.04e-06[OK]  dQ: 8.34e-07[OK]  dK: 2.26e-06[OK]  dV: 1.19e-06[OK]
[1x1, N= 512, d=128]  O: 5.51e-07[OK]  dQ: 5.07e-07[OK]  dK: 1.19e-06[OK]  dV: 1.55e-06[OK]
[1x1, N= 127, d= 64]  O: 1.43e-06[OK]  dQ: 1.49e-06[OK]  dK: 2.50e-06[OK]  dV: 9.54e-07[OK]
```

误差全部在 1e-6 ~ 2.5e-6 范围内。forward、backward、autograd 端到端全部正确。多 batch + 多头配置也通过了。

### 4.2 显存——达成目标

| N | Naive | Python Tiled | Triton FA | PyTorch Built-in |
|---|-------|-------------|-----------|-----------------|
| 512 | 10.6 MB | 8.8 MB | 8.8 MB | 8.8 MB |
| 4096 | 140.6 MB | 13.3 MB | 13.2 MB | 13.1 MB |
| 16384 | 2074.1 MB | 28.3 MB | 28.2 MB | 28.1 MB |
| 24576 | **4644.1 MB** | 38.4 MB | **38.3 MB** | 38.1 MB |

Triton FA 的显存曲线与 PyTorch 内置 SDPA 完全重合。N=24576 时，Naive 需要 4.6 GB，Triton 只用 38 MB——**减少 99.2%**。

### 4.3 速度——有改善但仍有差距

| N | Naive | Python Tiled (S3) | **Triton FA (S4)** | PyTorch Built-in |
|---|-------|------------------|-------------------|-----------------|
| 512 | 0.14 ms | 18.73 ms | **0.14 ms** | 0.02 ms |
| 2048 | 0.19 ms | 299.77 ms | **4.24 ms** | 0.20 ms |
| 4096 | 1.12 ms | 1099.17 ms | **11.47 ms** | 0.24 ms |
| 8192 | 3.64 ms | 4369.93 ms | **31.51 ms** | 0.89 ms |

三个关键数字：

1. **vs Python Tiled: 快 140x**（N=8192 时 31.5ms vs 4370ms）。这证明了阶段 3 的性能瓶颈确实是 Python 循环 + kernel launch overhead。

2. **vs Naive: 慢 8.7x**（N=8192 时 31.5ms vs 3.64ms）。Naive 用的 `torch.matmul` 底层是 cuBLAS（优化了十几年的 hand-tuned CUDA kernel）。我们的 Triton kernel 是 1-row-per-program + element-wise 的最简版本。这个差距是预期内的。

3. **vs PyTorch Built-in: 慢 35x**。PyTorch 内置的 `F.scaled_dot_product_attention` 用的是 NVIDIA cuDNN 的 FlashAttention 后端，那是 Tri Dao 的官方 CUDA 实现，BLOCK_M=128、BLOCK_N=64、Tensor Core 全开——我们暂时没法跟它比，也不需要跟它比。我们的目标是**理解原理并自己写出来**。

### 4.4 长序列——这才是 FlashAttention 的真正杀手锏

| N | Triton FA 显存 | Naive 显存 |
|---|-------------|----------|
| 4096 | 12.2 MB | 141.1 MB |
| 8192 | 17.2 MB | 530.1 MB |
| 16384 | 26.2 MB | 2076.1 MB |
| 24576 | 36.3 MB | 4646.1 MB |

N=24576 时 Triton FA 只用 36 MB——这意味着在 8GB 的 RTX 4060 上，理论上可以跑到 N≈100K 甚至更长。而 Naive 在 N≈32768 时就可能触发 OOM。

**这就是 FlashAttention 的核心价值：它让你在消费级 GPU 上跑以前需要 A100 才能跑的序列长度。**

---

## 5. 这次学到了什么

### 5.1 技术层面

1. **Triton 的编程模型**：`tl.jit` 装饰的 Python 函数 → 编译为 GPU 中间表示 (TTIR) → 优化和代码生成。你不需要写 CUDA C，但你需要理解 GPU 的内存层次和并行模型。

2. **Pointers 是第一公民**：在 Triton kernel 里，你操作的是裸指针 (`Q_ptr + offset`)，不是 PyTorch tensor。每一步指针偏移都要自己算清楚——错了不会有 IndexError，只有错误的数值结果。

3. **Online softmax 的三个状态 (m, d, O) 是 FlashAttention 的灵魂**：forward 里它们让你只需要一趟扫描，backward 里它们让你可以高效重算 P。

4. **Backward 两趟算法的必要性**：D 必须对整行求和，不能在 tile 内独立计算。阶段 3 踩过的坑，阶段 4 不会再踩。

5. **atomic_add 的性能代价**：它保证了正确性，但在高竞争场景下会降低吞吐。需要根据数据依赖图谱判断哪里需要它。

### 5.2 工程层面

1. **从简单到复杂，逐步验证**：先用最简单的 mapping（1 row per program）确保正确性，再考虑优化。不要一上来就搞复杂的 tiling。

2. **区分算法 bug 和工具链 bug**：花了两小时确认 BLOCK_M > 1 是 Triton 编译器的问题而非我代码的问题。这个区分能力在系统研究里非常重要。

3. **工具链的成熟度很重要**：同一个 Triton 版本，在 Linux 上可能 BLOCK_M > 1 和 FP32 tl.dot 都是正常的。Windows 上的包（triton-windows）是社区维护的，有滞后和未修复的 bug。

4. **指针偏移错误的排查**：batch 维度指针偏移错误的症状是多 batch 测试全部失败，但单 batch 测试全部通过。这种"局部正确、全局错误"的模式是指针问题的经典特征。

### 5.3 我还没有完全理解的东西

- **Triton 的 shared memory 管理**：我声明了 `tl.zeros([D_HEAD], ...)`，Triton 的 compiler 决定把它放寄存器还是 shared memory。我还没有完全掌握这个分配机制。

- **Occupancy 的量化计算**：我知道每个 program 用多少寄存器，知道每个 SM 有多少寄存器，但我还没有用 Nsight Compute 实际验证 occupancy。

- **为什么 BLOCK_M > 1 会产生固定的 0.766 误差**：这至今是个谜。如果将来有机会在 Linux 上跑同样的代码，可以验证是否是 Windows Triton 特有的 bug。

---

## 6. 和阶段 3 的对比总结

| | 阶段 3 (Python Tiled) | 阶段 4 (Triton Kernel) |
|---|---|---|
| 实现方式 | Python 双层循环 + torch.bmm | Triton GPU kernel |
| Forward kernel launch 次数 | O(外层 × 内层) ≈ 几百次 | **1 次** |
| Backward kernel launch 次数 | O(外层 × 内层 × 2) ≈ 几百次 | **2 次**（Pass 1 + Pass 2） |
| 中间数据 S/P 的位置 | tile 内 tensor，可能写回 HBM | 纯寄存器，从不写 HBM |
| 显存节省 | 是 | 是（完全相同） |
| 速度 vs Naive (N=8192) | 慢 1200x | 慢 8.7x |
| 速度 vs Python Tiled (N=8192) | — | **快 140x** |
| 用了 Tensor Core？ | 间接（torch.bmm → cuBLAS） | 未使用 |
| autograd 可用？ | 是 | 是 |

**核心收获**：我用 Triton 把一个 Python 双层循环变成了 3 个 GPU kernel（forward + backward pass1 + backward pass2），在保留全部显存节省的同时，相对阶段 3 的 Python 实现获得了 140x 加速。

---

## 7. 如果还有时间可以做……

1. **换 Linux 环境跑同样的代码**：验证 BLOCK_M > 1 和 FP32 tl.dot 是否在 Linux Triton 上正常工作。如果可以，实现真正的 tiled matmul 版本。

2. **读 Triton 编译后的 PTX/SASS 代码**：理解 compiler 到底做了什么优化，看看 element-wise dot product 被编译成了什么指令序列。

3. **用 `torch.autograd.gradcheck` 做更严格的梯度验证**：当前只验证了 `.sum().backward()`，gradcheck 会验证每个输入元素的偏导数，能捕捉更精细的梯度 bug。

4. **实现 causal mask 版本**：把 mask 参数传到 kernel 里，在内层循环跳过 j > i 的 tile。这会引入 SM 负载不均衡的问题，是一个很好的后续研究点。

5. **和其他开源 FA 实现对比**：比如 Tri Dao 的官方 FlashAttention-2、Meta 的 xFormers、OpenAI 的 Triton FA tutorial。读他们的代码，理解设计选择的差异。

---

## 8. 写在最后

做完这 4 个阶段的实验，我最大的感受是：**把一个论文里的算法从零实现出来，和读懂这个算法，是完全不同的两件事。**

读 FlashAttention 论文的时候，我觉得"哦，就是分块 + online softmax + kernel fusion，理解了"。但真正动手写的时候：

- 阶段 2 让我理解 online softmax 为什么是"数学上精确的"
- 阶段 3 让我理解 backward 为什么必须两趟
- 阶段 4 让我理解为什么"写成 GPU kernel"才是 FlashAttention 的真正贡献——它不是一个算法创新和一个系统实现的简单加法，而是两者的乘法。**算法和系统是同一枚硬币的两面。**

而且我现在写的 Triton kernel 还远不是最优版本。真正的 FlashAttention-2 里还有 causal mask 的 workload balancing、persistent kernel 的 work-stealing、fp8 量化等一大堆优化——每一项都是一个值得花一周深入研究的子问题。

这条路还很长，但这个方向是对的。
