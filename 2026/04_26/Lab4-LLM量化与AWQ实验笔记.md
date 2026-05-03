# Lab 4 学习笔记：LLM 量化与 AWQ


> **主题**：大语言模型权重量化（Weight-only Quantization）  
> **模型**：OPT-1.3B on WikiText-2  
> **核心目标**：在显著压缩模型体积的同时，最大限度恢复量化带来的精度损失

---


## 1. 背景与动机

大语言模型在众多任务上表现卓越，但其庞大的参数规模对内存容量和内存带宽提出了极高要求。边缘设备（如 Jetson Orin Nano 仅 8GB DRAM）甚至无法以半精度加载最小的 LLaMA-2 模型。在单 batch 解码阶段，每个 token 的生成本质上是一个GEMV操作，计算密度低，主要受限于内存带宽。以 A100 为例，FP16 算力 312 TFLOPS，带宽约 2000 GB/s，计算访存比仅约 $10^2$，远低于硬件平衡点，因此权重量化成为加速推理的关键手段。

本实验聚焦权重仅量化（weight-only quantization），尤其是 **AWQ (Activation-aware Weight Quantization)**。该方法发现：激活中存在少量但稳定的异常值通道，对应权重通道极为重要。直接均匀量化会严重破坏这些通道，造成模型质量急剧下降。AWQ 通过等价缩放将这些通道的量化误差转移至其他通道，从而在不引入混合精度的情况下大幅还原精度。最终结合高效 4-bit 内核，可在 RTX 4090 等硬件上实现实际加速。

---

## 2. 实验环境与基础函数

### 2.1 困惑度评估（Perplexity）

使用 WikiText-2 测试集，评估前 40 个不重叠的 2048-token 序列的困惑度：

```python
def evaluate(model, tokenizer):
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')
    testenc = testenc.input_ids.to(model.device)
    nsamples = 40
    model = model.eval()
    nlls = []
    for i in range(nsamples):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)
    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))
```

困惑度越低表示模型对真实文本的预测越准确。

### 2.2 模型大小计算

```python
def get_model_size(model: nn.Module, data_width=16, group_size=-1):
    if group_size != -1:
        data_width += (16 + 4) / group_size
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width
```

考虑 group-wise 量化时量化参数（scale, zero）的额外存储开销。`group_size=-1` 表示 per-channel 或 per-tensor 量化，无此开销。

### 2.3 FP32 基线模型

加载 `facebook/opt-1.3b`，评估基线：

```
模型：OPT-1.3B，FP32
困惑度：~15.37
模型大小：约 5069 MiB
```

这是后续所有量化实验的对比基准。

---

## 3. 均匀量化与伪量化（Pseudo Quantization）

### 3.1 伪量化原理与实现

量化函数将浮点值映射到 $[0, 2^b-1]$ 整数域，再反量化回浮点，以模拟量化误差而不改变数值精度。

给定权重 $W$，对每个量化组（group）分别计算：

$$s_q = \frac{\alpha - \beta}{2^{b} - 1}, \quad z = -\text{Round}\left(\frac{\beta}{s_q}\right), \quad 0 \le z \le 2^b-1$$

$$\hat{W} = \text{Clamp}\left(\text{Round}\left(\frac{W}{s_q}\right) + z,\ 0,\ 2^b-1\right)$$

$$W_q = (\hat{W} - z) \times s_q$$

核心实现 `pseudo_quantize_tensor`：

```python
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        w = w.reshape(-1, q_group_size)   # 按组 reshape
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)  # 量化
    w = (w - zeros) * scales                                       # 反量化
    w = w.reshape(org_w_shape)
    return w
```

特点：组大小 `q_group_size=128` 时，每 128 个元素共享一个 scale 和 zero point，很好地平衡了粒度与开销。

### 3.2 3-bit 量化的初步尝试

直接对 OPT-1.3B 所有线性层进行 3-bit 组量化（组大小 128）：

```
困惑度：~23.36 → 相比 FP32 急剧恶化
模型大小：~1303 MiB （压缩约 3.2 倍）
```

压缩效果显著，但模型几乎不可用，说明直接均匀量化严重破坏了关键权重，急需保护机制。

---

## 4. 保护显著权重通道 — Question 1

### 4.1 激活异常值与校准数据

AWQ 的核心观察：LLM 的激活中存在异常值（outliers），它们集中出现在少数输入通道上，且跨 token 持续存在。对应这些高激活通道的权重对最终输出影响巨大，即使微小的量化误差也会被异常激活放大，严重损害模型质量。

为识别这些显著通道，需从校准数据集（pile-val-backup）采集每层线性层的输入特征，计算其平均 L1 范数作为重要性：

```python
def get_calib_feat(model, tokenizer):
    input_dict = dict()
    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple): x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]
    # 注册前向 hook 收集数据
    ...
```

`input_feat[name]` 是一个列表，每个元素为一次前向的 `[C_in]` 大小的平均幅度。最终重要性为多次累加求和。

### 4.2 Question 1.1：实现显著通道保护（Salient FP16）

**Question 1.1 (20 pts)**：在量化前后保护 1% 最重要的输入通道，保持其值为原始 FP16。

需要填充 `pseudo_quantize_model_salient_weight_fp16`：

```python
for n, m in model.named_modules():
    if isinstance(m, nn.Linear):
        importance = sum(input_feat[n]).float()

        # Step 1: 找出 top 1% 显著输入通道
        num_salient_channels = max(1, int(0.01 * importance.shape[0]))
        outlier_indices = torch.topk(importance, k=num_salient_channels, largest=True).indices

        # Step 2: 备份这些通道的权重
        outlier = m.weight.data[:, outlier_indices].clone()

        # 对整个权重进行伪量化
        m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

        # Step 3: 将显著通道的权重复原为 FP16 备份值
        m.weight.data[:, outlier_indices] = outlier
```

**思路解析**：

- **重要性**：利用校准数据集收集的激活幅度（绝对值均值）来衡量输入通道的重要性。激活越大的通道，权重量化误差被放大的倍数越高。
- **Top 1% 选取**：`torch.topk(importance, k=..., largest=True)` 返回前 1%（至少 1 个）的索引。这些通道保留 FP16，其他 99% 通道仍进行低比特量化。
- **备份与恢复**：在 `pseudo_quantize_tensor` 调用前备份 `weight[:, outlier_indices]`，量化后覆盖回去。这样显著通道的权重完全不受量化的影响。

**效果**：3-bit 量化下困惑度从 23.36 降至 17.15，接近 FP32 水平，证明仅保护 1% 的通道即可大幅恢复精度，且模型大小仅轻微增加（额外存储 1% 的 16-bit 权重）。

### 4.3 Question 1.2：消融实验 — 随机保护通道

**Question 1.2 (15 pts)**：随机选择 1% 通道保持 FP16，观察困惑度。

需实现 `pseudo_quantize_model_random_weight_fp16`，用 `torch.randperm` 替代 `topk`：

```python
# Step 1: 随机选择 1% 通道
num_random_channels = max(1, int(0.01 * importance.shape[0]))
outlier_indices = torch.randperm(importance.shape[0])[:num_random_channels]

# 备份与恢复逻辑同上
outlier = m.weight.data[:, outlier_indices].clone()
m.weight.data = pseudo_quantize_tensor(...)
m.weight.data[:, outlier_indices] = outlier
```

**思路**：随机选择没有任何信息指导，仅作为对照实验，验证“选择显著通道”是否真正有效。

**结果**：困惑度飙升至100以上，远差于显著通道保护，甚至差于全量化。这证明保护对象的选择至关重要——只有保护那些对应激活异常值的通道，才是在海量参数中“把钱花在刀刃上”。

### 4.4 Question 1.3：为什么显著通道如此重要？

**Question 1.3 (15 pts)**：解释显著权重通道为何如此重要。

**答案**：

大型语言模型的激活分布具有通道间方差异常大的特点：少数几个通道的平均激活幅度远大于其余通道，且这些异常值通道在任意输入 token 下都保持高幅度（persistent outliers）。

设某一线性层为 $Y = X W^\top$（忽略转置细节，实质为 $Y_{out} = W X_{in}$）。量化的误差为：

$$Y_q - Y = (W_q - W) X$$

当某输入通道 $j$ 的激活 $X_{:,j}$ 幅度很大时，该通道对应权重列 $W_{:,j}$ 的量化误差会被放大 $|X_{:,j}|$ 倍，直接污染所有输出 token 的对应输出通道。因此，保留这些幅值占比极小但影响极大的通道的权重在 FP16，能够以极小的存储代价（约 1%）消除最大误差源，使整体困惑度大幅恢复。

相反，随机保护通道可能选到低激活通道，即使权重完美保留，对输出的帮助微乎其微；高激活通道仍被量化破坏，所以性能崩溃。

---

## 5. 缩放保护法 — Question 2

尽管保留 1% 的 FP16 通道极其有效，但混合精度实现复杂，需要特殊的推理内核来处理部分通道高精度、部分通道低精度的计算。AWQ 提出一种巧妙的等价变换：在量化前将显著通道权重放大 $s$ 倍，量化后再缩小 $s$ 倍，同时相应地对激活值进行缩小/放大。这可以将量化误差从显著通道“转移”到非显著通道，从而在统一精度的前提下达到近似保护效果。

### 5.1 缩放法原理：用数学变换替代混合精度

考虑显著通道 $i$ 的权重向量 $\mathbf{w}$ 和对应输入 $x$（省略下标）。原始运算：

$$y = \mathbf{w} \cdot x$$

量化误差为 $Err(Q(\mathbf{w}) x) = \Delta \cdot \text{RoundErr}(\mathbf{w}/\Delta) \cdot x$，其中 $\Delta = \max(|w|)/(2^{N-1}-1)$。

若先将 $\mathbf{w}$ 乘以 $s$，$x$ 除以 $s$（线性层输出不变），则：

$$y = (s \cdot \mathbf{w}) \cdot (x/s)$$

现在量化发生在缩放后的权重上，而缩放后的输入变小了。误差变为：

$$Err(Q(s\mathbf{w}) \cdot (x/s)) = \Delta \cdot \text{RoundErr}(s\mathbf{w}/\Delta) \cdot (x/s)$$

由于量化组较大（128），放大单个通道的权重通常不会增大组内最大值，即 $\Delta$ 保持不变（这是关键前提）。然而输入 $x$ 被 $s$ 缩小，导致整体误差变为原来的 $1/s$。因此，只要 $s > 1$，显著通道的量化误差就被有效抑制。

物理实现上，可通过在量化前将 `weight[:, salient_channels] *= s`，量化后再 `weight[:, salient_channels] /= s` 来完成。对应的输入缩放由前一层的 LayerNorm 或 Linear 权重分担，AWQ 会自动搜索最优 $s$ 来处理跨层影响。

### 5.2 Question 2.1：实现缩放 + 量化

**Question 2.1 (20 pts)**：在 `pseudo_quantize_model_weight_scaleup` 中对显著通道先放大，量化后再缩小。

填充代码（简化自原实现）：

```python
for n, m in model.named_modules():
    if isinstance(m, nn.Linear):
        importance = sum(input_feat[n]).float()
        # Step 1: 找到 1% 显著通道
        num_salient_channels = max(1, int(0.01 * importance.shape[0]))
        outlier_indices = torch.topk(importance, k=num_salient_channels, largest=True).indices

        # Step 2: 放大显著通道权重
        m.weight.data[:, outlier_indices] *= scale_factor

        # Step 3: 伪量化
        m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

        # Step 4: 缩小回去，恢复原始数值范围
        m.weight.data[:, outlier_indices] /= scale_factor
```

**思路**：先乘后除，确保这些通道的最终权重值仍在原始幅度附近，但量化过程在“放大”后的版本上进行，使得量化网格相对更精细（相对误差更小）。配合后续 ACT 缩放（由搜索决定），输入通道的数值会相应缩小，完整重建等价性。

**效果**（scale_factor=2, 3-bit）：困惑度约为 **18.93**，虽不如直接保留 FP16（17.15），但已远好于纯量化（23.36），且无需混合精度。这是迈向完全统一低精度推理的关键一步。

### 5.3 Question 2.2：不同缩放因子的影响

**Question 2.2 (15 pts)**：尝试 `scale_factor = 1, 2, 3, 4`，观察困惑度变化并解释。

**答案**：

随着 `scale_factor` 从 1 增加到 2，困惑度下降（量化误差减小），继续增加到 4 时困惑度反而上升，呈现先降后升的 U 型曲线。

**原因**
基于 AWQ 的核心前提：

- 放大过度会破坏 $\Delta$ 不变的条件。当显著通道的权重被过分放大，其最大值可能超过组内其他通道，导致整组的 $\Delta$ 增大（scale factor 变大），从而增加了其他 99% 非显著通道的量化误差。非显著通道虽然单个影响小，但数量庞大，累积误差不可忽略。
- 同时，过度放大后输入 $x$ 需除以大 $s$，这会使对应激活通道的数值过小，若后续有非对称量化或 zero point 的不准确，也会引入额外误差。

因此最优 $s$ 需要在 保护显著通道 与 避免损害非显著通道 之间权衡，这也正是自动搜索 $s$ 的必要性所在。

### 5.4 Question 2.3：自动搜索最优缩放 (AWQ)

**Question 2.3 (15 pts)**：实现最优缩放搜索，即 AWQ 的核心流程。

AWQ 对每一层的输入特征 $s_X$（由校准集得到），在搜索空间 $s = s_X^\alpha$ 中寻找最佳 $\alpha$（$\alpha \in [0,1]$）。目标是最小化量化前后输出的 MSE：

$$\mathbf{L}(s) = \left\| Q(\mathbf{W} \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} X) - \mathbf{W} X \right\|_F^2$$

其中 $s$ 为每个输入通道的缩放向量。在实现中，将 $s$ 应用于 block 内多个 Linear 层，并使用相同的 $\alpha$ 生成 $s$，避免破坏跨层等价性。

填充的代码：

```python
# Step 1: 初始化最优记录
best_error = float('inf')
best_ratio = -1
best_scales = None

n_grid = 20
for ratio in range(n_grid):
    ratio = ratio * 1 / n_grid   # ratio 即搜索的 α

    # Step 2: 根据当前 α 计算每通道缩放因子
    scales = s_x.pow(ratio)
    scales = torch.clamp(scales, min=1e-4)
    scales = scales / (scales.max() * scales.min()).sqrt()  # 稳定数值范围
    # ...

    for fc in linears2scale:
        # 将权重放大
        fc.weight.mul_(scales)
        # 伪量化
        fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size)
        # Step 3: 缩回原始幅度
        fc.weight.div_(scales)

    out = block(x, **kwargs)  # 前向计算缩放后的输出
    loss = (org_out - out).float().pow(2).mean().item()
    # 记录最佳 α 和 scales
    if loss < best_error:
        best_error = loss
        best_ratio = ratio
        best_scales = scales
    # 恢复 block 状态进行下一轮尝试
    block.load_state_dict(org_sd)
```

**思路解析**：

- **搜索空间**：$\alpha$ 从 0 到 1 离散化为 20 个点，分别尝试。
- **通道缩放因子**：$s = s_X^\alpha$。$s_X$ 是校准集统计所得激活的平均幅度（每个输入通道一个值）。$\alpha$ 越小，$s$ 越接近 1（微弱缩放）；$\alpha$ 越大，异常通道被显著放大。
- **缩放与量化**：在 block 内同时对 Q/K/V 等多个 Linear 层应用相同的 $s$ 进行权重缩放、量化、再还原，以保持跨层一致性。注意这里只模拟了权重侧缩放，真正的 AWQ 还会联动调整前一层的归一化权重或输出缩放，但本实验在 `scale_ln_fcs` / `scale_fc_fc` 中完成等效处理，以保证最终模型在实际推理时无需额外操作。
- **误差度量**：将缩放-量化后的 block 输出与原始浮点输出比较，计算 MSE。最小 MSE 对应的 $\alpha$ 即为该 block 最合适的缩放强度。
- **跨 block 应用**：逐个 `OPTDecoderLayer` 调用 `auto_scale_block`，分别找到每个 block 的最优 $s$，再整体量化全部 Linear 层。

**最终效果**：3-bit 量化下困惑度达到17.92，非常接近留 1% FP16 的 17.15，且模型仍保持纯 3-bit 精度，无需混合精度引擎，可直接部署于标准硬件。

---

## 6. 总结与思考

本实验完整复现了 AWQ 的核心思想：**利用激活分布中的异常值通道知识，通过等价缩放大幅降低显著通道的量化误差，从而实现纯低精度下的高保真权重量化**。

| 方法 | 3-bit 困惑度 | 特性 |
|------|-------------|------|
| 直接均匀量化 | 23.36 | 严重退化，不可用 |
| 保护 1% 显著通道 (FP16) | 17.15 | 混合精度，精度最佳 |
| 随机保护 1% 通道 (FP16) | >100 | 证明选择策略的重要性 |
| 缩放法 (s=2) | 18.93 | 无混合精度，仍有差距 |
| 自动搜索最优缩放 (AWQ) | 17.92 | 无混合精度，接近保护 FP16 的表现 |

**一句话总结：**

> LLM 量化中，权重的重要性极度不平衡：绝大多数通道的权重可以承受粗粒度量化，而极少数与激活异常值耦合的通道需要特殊保护。AWQ 通过激活感知的缩放将这些“重点保护对象”的量化误差有效掩蔽，实现了在标准低精度推理框架下的高质量压缩。后续结合 TinyChatEngine 等高效 4-bit 内核，即可在笔记本等终端硬件上流畅运行原来的“庞然大物”。

