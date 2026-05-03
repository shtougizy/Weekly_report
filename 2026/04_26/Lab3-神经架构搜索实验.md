# Lab 3 学习笔记：神经网络架构搜索


> **主题**：神经网络架构搜索（Neural Architecture Search, NAS）  
> **模型**：MCUNetV2 超网络 on Visual Wake Words (VWW)  
> **核心目标**：利用 Once-for-All 超网络与预测器，在极低资源约束下自动搜索高效子网络

---



## 1. 背景与动机

传统神经网络设计依赖专家手工调整层数、通道数、卷积核大小、输入分辨率等超参数，设计空间极其庞大，手工调参效率低下。**神经网络架构搜索（NAS）** 可自动化这一过程，在精度与效率之间寻找最优平衡。

早期 NAS 方法（如 NAS-RL、NASNet）需反复训练候选网络，计算开销巨大。后来的可微分 NAS（DARTS、ProxylessNAS）和 One-Shot 方法（如 SPOS）虽提升了效率，但每次为新硬件定制模型仍需完整训练+搜索+微调流程，面对数十亿 IoT 设备难以规模化。

本实验基于 **Once-for-All (OFA)** 方法（论文：[Once for All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/abs/1908.09791)）。OFA 训练一个超网络（super network），其中包含所有可能的子网络（sub-networks），子网络可直接提取并部署，无需重新训练。同时，OFA 引入精度预测器和效率预测器，避免在搜索过程中反复跑完整验证集或性能测量，大幅加速架构搜索。

本实验以 **MCUNetV2** 超网络和 **Visual Wake Words (VWW)** 数据集为例，带领我们：
- 实现效率与精度预测器
- 完成随机搜索与进化搜索
- 在内存 ≤ 250KB、MACs ≤ 60M 等真实微控制器约束下找到高精度模型

---

## 2. 实验环境与模型结构

### 2.1 VWW 数据集与预处理

Visual Wake Words 数据集是从 MS COCO 抽取的二分类（有人/无人）图像数据集，专为微控制器上的视觉唤醒词任务设计。

```python
def build_val_data_loader(data_dir, resolution, batch_size=128, split=0):
    # split = 0: 正式验证集, split = 1: holdout 迷你验证集（用于生成精度数据集和校准BN）
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    val_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    val_dataset = torch.utils.data.Subset(val_dataset, list(range(len(val_dataset)))[split::2])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(8, os.cpu_count()), pin_memory=False)
    return val_loader
```

关键点：归一化到 `[-1,1]`，`split=1` 的 holdout 集用于生成 `[architecture, accuracy]` 对和 BN 统计量校准，防止在搜索过程中信息泄露。

### 2.2 MCUNetV2 超网络与设计空间

MCUNetV2 是为微控制器量身定制的高效网络家族，采用 patch-based 推理和感受野重分配。其超网络 `OFAMCUNets` 包含大于10^19 个子网络，设计空间如下：

| 超参数 | 可选值 | 说明 |
|--------|--------|------|
| 输入分辨率 (image_size) | [96, 112, 128, 144, 160] | 影响特征图尺寸与计算量 |
| 宽度乘数 (width_mult) | [0.5, 0.75, 1.0] | 全局缩放每层通道数 |
| 核大小 (ks) | [3, 5, 7] | 每个 inverted block 的深度卷积核尺寸 |
| 扩展比 (e) | [3, 4, 6] | inverted residual block 的 expansion ratio |
| 深度 (d) | [0, 1, 2] | 每个 stage 可额外增加的 block 数量（相对基准深度） |

```python
ofa_network = OFAMCUNets(
    n_classes=2, bn_param=(0.1, 1e-3), dropout_rate=0.0,
    base_stage_width="mcunet384",
    width_mult_list=[0.5, 0.75, 1.0],
    ks_list=[3, 5, 7],
    expand_ratio_list=[3, 4, 6],
    depth_list=[0, 1, 2],
    base_depth=[1, 2, 2, 2, 2],
    fuse_blk1=True,
    se_stages=[False, [False, True, True, True], True, True, True, False],
)
ofa_network.load_state_dict(torch.load("vww_supernet.pth")["state_dict"])
```

### 2.3 提取子网络并评估

OFA 超网络的核心特性是提取即部署，无需微调即可获得较高精度（约88.7%）。

```python
def evaluate_sub_network(ofa_network, cfg, image_size=None):
    if "image_size" in cfg: image_size = cfg["image_size"]
    ofa_network.set_active_subnet(**cfg)
    subnet = ofa_network.get_active_subnet().to(device)
    peak_memory = count_peak_activation_size(subnet, (1, 3, image_size, image_size))
    macs = count_net_flops(subnet, (1, 3, image_size, image_size))
    params = count_parameters(subnet)
    calib_bn(subnet, data_dir, 128, image_size)
    val_loader = build_val_data_loader(data_dir, image_size, 128)
    acc = validate(subnet, val_loader)
    return acc, peak_memory, macs, params
```

`calib_bn` 是 OFA 中的关键步骤：由于超网络训练时 BN 统计量是针对完整超网络计算的，提取子网络后需要用少量数据重新校准 BN 的 running mean/var，以恢复精度。

---

## 3. 准备工作：设计空间探索 (Question 1)

**Question 1 (5 pts)**: Try manually sample different subnets by running the cell above multiple times. You can also vary the input resolution. Talk about your findings.

**答案**：

通过多次采样不同子网络并改变输入分辨率，有以下发现：

- **输入分辨率**对精度影响最大。当分辨率从 96 提升到 160 时，精度通常可提高 3–5 个百分点。因为 VWW 数据集需要识别图像中的人，高分辨率保留了更多细节，大幅提升了判别能力。
- **宽度乘数 (width_mult)** 同样显著影响精度和计算量。1.0× 比 0.5× 的子网络精度高出约 3–4 个百分点，同时 MACs 和参数量增加近 4 倍。
- 核大小和扩展比的影响相对次要：在相同的宽度和分辨率下，不同核大小组合的精度波动通常在 1% 以内，但它们对 MACs 和内存的影响明显。
- 深度增加（如每个 stage 多 1–2 个 block）在 MACs 允许的情况下能稳定提升精度，但边际效益递减。
- 在效率约束下，优先选择较小的宽度乘数和适中的分辨率（如 128）往往能获得较优的精度–效率平衡。

**结论**：在资源极度受限时，优先调整分辨率和宽度乘数这两个全局参数，可获得"性价比"最高的精度提升。

---

## 4. Part 1：预测器 (Question 2–4)

架构搜索需要快速评估子网络的效率和精度，直接用实测方式（跑完整验证集或测量延迟）耗时过长。预测器通过回归模型将评估时间从分钟级降至毫秒级。

### 4.1 效率预测器 (AnalyticalEfficiencyPredictor) — Question 2

**Question 2 (10 pts)**: Implement the efficiency predictor.

效率预测器使用hook 分析直接计算 MACs 和峰值激活内存，无需实际执行推理，属于解析型预测器。

**实现代码**：

```python
class AnalyticalEfficiencyPredictor:
    def __init__(self, net):
        self.net = net

    def get_efficiency(self, spec: dict):
        self.net.set_active_subnet(**spec)
        subnet = self.net.get_active_subnet()
        if torch.cuda.is_available():
            subnet = subnet.cuda()
        data_shape = (1, 3, spec.get("image_size", 224), spec.get("image_size", 224))
        macs = count_net_flops(subnet, data_shape)
        peak_memory = count_peak_activation_size(subnet, data_shape)
        return dict(millionMACs=macs / 1e6, KBPeakMemory=peak_memory / 1024)

    def satisfy_constraint(self, measured: dict, target: dict):
        for key in measured:
            if key not in target: continue
            if measured[key] > target[key]:
                return False
        return True
```

**思路解析**：

- `data_shape` 设为 `(1, 3, H, W)`，其中 `H` 和 `W` 取自 `spec` 中的 `image_size`（默认为 224），代表单个样本的前向传播所需的内存/计算。
- `count_net_flops` 返回 MACs 数（以 M 为单位），`count_peak_activation_size` 返回峰值激活内存（以 Byte 为单位），分别转换为 `millionMACs` 和 `KBPeakMemory`。
- `satisfy_constraint` 检查当前网络是否满足所有给定约束（例如 MACs ≤ 60M, memory ≤ 250KB），只要有一项超出就返回 `False`。这是搜索过程中筛除无效候选的核心。

测试结果：对最小和最大子网络调用效率预测器，结果应与 `evaluate_sub_network` 中计算的 MACs 和内存匹配。

### 4.2 精度预测器结构 (AccuracyPredictor) — Question 3

**Question 3 (10 pts)**: Implement the accuracy predictor (MLP).

精度预测器是一个三层 MLP，输入为子网络架构编码向量，输出为相对于基础精度的偏移量。架构编码由 `MCUNetArchEncoder` 完成，将所有离散超参数转换为one-hot 向量再拼接。

**One-hot 编码示例**：

```python
kernel_size = [3, 5, 7]   # 3 → [1,0,0], 5 → [0,1,0], 7 → [0,0,1]
expand_ratio = [3, 4, 6]  # 同理
```

每 block 的编码 = `concat( one_hot(ks), one_hot(e) )`；全局参数（分辨率、宽度乘数）也分别 one-hot 化。最终拼接成一个长二进制向量。

**MLP 结构实现**：

```python
class AccuracyPredictor(nn.Module):
    def __init__(self, arch_encoder, hidden_size=400, n_layers=3, checkpoint_path=None, device="cuda:0"):
        super().__init__()
        self.arch_encoder = arch_encoder
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        layers = []
        for i in range(self.n_layers):
            in_dim = self.arch_encoder.n_dim if i == 0 else self.hidden_size
            layers.append(nn.Linear(in_dim, self.hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_size, 1, bias=False))
        self.layers = nn.Sequential(*layers)
        self.base_acc = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=False)
        # 加载检查点等
```

**思路解析**：

- 第一层输入维度必须等于架构编码向量的长度 `arch_encoder.n_dim`；后续层输入维度为 `hidden_size=400`。
- 每层后接 ReLU 激活，最后接线性层输出一个标量（相对于 base_acc 的差值）。`base_acc` 作为可学习偏移量，但实验中使用数据集的平均精度固定它：`acc_predictor.base_acc.data += base_acc`。
- 最后网络的输出为 `y + self.base_acc`，即预测的精度绝对值。

### 4.3 精度预测器训练 — Question 4

**Question 4 (10 pts)**: Complete the code for accuracy predictor training.

训练数据来自预先计算的 50000个(架构编码, 精度)对，40k 用于训练，10k 用于验证。目标值已减去 `base_acc`，即训练 `accuracy - base_acc`，使回归目标均值为零，易于优化。

**训练循环填空**：

```python
criterion = torch.nn.L1Loss().to(device)
optimizer = torch.optim.Adam(acc_predictor.parameters())
acc_predictor.base_acc.data += base_acc
for epoch in range(10):
    acc_predictor.train()
    for (data, label) in train_loader:
        data, label = data.to(device), label.to(device)
        pred = acc_predictor(data)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc_predictor.eval()
    with torch.no_grad():
        for (data, label) in valid_loader:
            data, label = data.to(device), label.to(device)
            pred = acc_predictor(data)
            loss = criterion(pred, label)
```

**思路解析**：

- 使用L1Loss而非 MSE，是因为 L1 对异常值不那么敏感，更适合精度预测这种相对平坦的回归任务。
- 训练时常规前向、损失计算、反向传播与参数更新，每步都需 `zero_grad()` 清除梯度。
- 验证阶段只需前向传播和损失计算，无需反向传播，用 `torch.no_grad()` 包裹。

**结果检验**：训练后绘制预测精度 vs 实测精度的散点图，应呈现明显的线性相关性（红色 y=x 对角线附近），表明精度预测器可信。

---

## 5. Part 2：架构搜索 (Question 5–10)

有了两个预测器，就可以在秒级完成子网络评估，从而在庞大的设计空间中搜索满足约束的最优结构。

### 5.1 随机搜索 (RandomSearcher) — Question 5

**Question 5 (5 pts)**: Complete the random search agent.

随机搜索算法不断随机采样保证满足约束的子网络，选出其中预测精度最高的。

```python
class RandomSearcher:
    def random_valid_sample(self, constraint):
        while True:
            sample = self.accuracy_predictor.arch_encoder.random_sample_arch()
            efficiency = self.efficiency_predictor.get_efficiency(sample)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return sample, efficiency

    def run_search(self, constraint, n_subnets=100):
        subnet_pool = []
        for _ in range(n_subnets):
            sample, efficiency = self.random_valid_sample(constraint)
            subnet_pool.append(sample)
        accs = self.accuracy_predictor.predict_acc(subnet_pool)
        best_idx = accs.argmax().item()
        return accs[best_idx], subnet_pool[best_idx]
```

**思路**：`accs.argmax()` 返回预测精度列表中的最大值索引，`.item()` 转为 Python 标量。

### 5.2 搜索与测量准确率 — Question 6

**Question 6 (5 pts)**: Complete the `search_and_measure_acc` function.

该函数调用搜索代理获取最优配置，再从超网络中提取子网络，校准 BN 并在真实验证集上评测精度。

```python
def search_and_measure_acc(agent, constraint, **kwargs):

    best_info = agent.run_search(constraint, **kwargs)
    ofa_network.set_active_subnet(**best_info[1])
    subnet = ofa_network.get_active_subnet().to(device)
    calib_bn(subnet, data_dir, 128, best_info[1]["image_size"])
    val_loader = build_val_data_loader(data_dir, best_info[1]["image_size"], 128)
    acc = validate(subnet, val_loader)
    visualize_subnet(best_info[1])
    return acc, subnet

```

`best_info[1]` 是子网络配置字典，`best_info[0]` 是预测精度。实测精度通常会与预测精度有微小偏差，但趋势一致。

### 5.3 进化搜索 (EvolutionSearcher) — Question 7

**Question 7 (20 pts)**: Complete the evolutionary search agent (crossover and population update).

进化算法模拟自然选择：维护一个种群（population），每代（generation）通过变异（mutation）和交叉（crossover）生成子代，保留精度最高的个体。

交叉操作需要填充的部分：

```python
def crossover_sample(self, sample1, sample2, constraint):
    while True:
        new_sample = copy.deepcopy(sample1)
        for key in new_sample.keys():
            if not isinstance(new_sample[key], list):
                # 非列表参数（如 image_size, width_mult）：随机从父本选取
                new_sample[key] = random.choice([sample1[key], sample2[key]])
            else:
                # 列表参数（如 ks, e, d）：逐元素随机选取
                for i in range(len(new_sample[key])):
                    new_sample[key][i] = random.choice([sample1[key][i], sample2[key][i]])
        efficiency = self.efficiency_predictor.get_efficiency(new_sample)
        if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
            return new_sample, efficiency
```

**思路**：交叉算子将两个父本的各超参数随机混合，生成新的子代。如果新子代违反约束则重新交叉，直到满足。这样既保持多样性又探索新区域。

种群更新需要填充的部分：

```python
# 每一代进化
for i in range(self.max_time_budget):

    # 按精度降序排序
    population = sorted(population, key=lambda x: x[0], reverse=True)
    # 保留前 parents_size 个精英个体
    population = population[:parents_size]
    # ... 记录最佳个体，生成子代（变异 + 交叉）并加入种群
```

**参数说明**：

- `parent_ratio`：保留为精英的比例（如 0.25），决定多少个体直接进入下一代。
- `mutation_ratio`：子代中通过变异产生的比例，其余由交叉产生。
- `arch_mutate_prob` / `resolution_mutate_prob`：控制各维度突变概率。

### 5.4 调参与发现 — Question 8

**Question 8 (10 pts)**: Run evolutionary search and tune `evo_params` to optimize the results. Describe your findings.

参数调整：默认参数 (`population_size=10`, `max_time_budget=10`) 太小，难以充分探索。适当增大：

```python
evo_params = {
    'arch_mutate_prob': 0.1,
    'resolution_mutate_prob': 0.1,
    'population_size': 50,      # 增大种群
    'max_time_budget': 100,     # 更多迭代代数
    'parent_ratio': 0.25,
    'mutation_ratio': 0.5,
}
```

**实验发现**：

- 增大 `population_size` 能显著提升搜索到的最优精度，因为初始采样覆盖更广的设计空间。
- 增加 `max_time_budget` 让进化过程有更多时间迭代优化，精度曲线逐渐收敛。
- `mutation_ratio` 设为 0.5 左右时，变异与交叉的平衡较好；过低则种群缺乏新血，过高则破坏精英结构。
- `arch_mutate_prob` 和 `resolution_mutate_prob` 保持在 0.1 能稳定进化，大幅提高易导致频繁违反约束。

对比随机搜索，进化搜索在相同约束下通常能获得少量的精度提升，因为其利用了过往评估信息进行有向搜索，样本效率更高。

### 5.5 真实约束下的搜索与可行性判断 — Question 9, 10

**Question 9 (15 pts + 10 bonus pts)**: Search under real-world constraints.

- 约束 A：250KB 峰值内存, 60M MACs → 要求精度 ≥ 92.5%
- 约束 B（bonus）：200KB, 30M MACs → 要求精度 ≥ 90%

通过进化搜索，可得到满足要求的模型。对于约束 A，在适当调整 `evo_params` 后，搜索得到子网络实测精度可达到92.8% 。对于约束 B，任务更难，需要更仔细调参，精度勉强达到 90%。

**Question 10 (10 pts)**: Is it possible to find a subnet with the given constraints in the current design space?  
- A: activation ≤ 256KB, MACs ≤ 15M  
- B: activation ≤ 64KB

**答案**：

- **A (256KB, 15M MACs)**：**不可能**。经效率预测器评估，即使使用最小分辨率（96）、最小宽度（0.5×）、最小深度和最小核大小组成的最小网络，MACs 仍远大于 15M（例如最小子网 MACs 约 30–50M，峰值内存约 200KB+）。因此同时满足 15M MACs 和 256KB 的配置不存在于当前设计空间中。

- **B (激活 ≤ 64KB)**：**也不可能**。最小子网的峰值激活内存已在 200KB 左右（输入 96×96×3 的特征图即需 96×96×3×4 bytes ≈ 108KB，加上中间特征图必然超过 64KB）。当前 MCUNet 设计空间不支持如此极端的激活内存限制。

**结论**：若需满足这些极端约束，必须扩大/改变设计空间（例如加入更激进的通道剪枝、更小的输入分辨率 48 或 32、patch-based 推理优化等），这超出了本实验范围，但正体现了 NAS 与系统协同设计的重要性。

---

## 6. 两种搜索方法对比与总结


随机搜索 

 • 策略：随机采样满足约束的子网络       
 • 样本数：n_subnets (如 300)          
 • 优点：实现简单，无超参数调优烦恼     
 • 缺点：样本效率低，无法利用历史信息   
 • 适用场景：快速验证设计空间，或作为基线

进化搜索 

 • 策略：种群迭代、变异、交叉、精英保留 
 • 超参数：种群大小、变异/交叉比例等   
 • 优点：样本效率高，能逼近全局最优    
 • 缺点：需调节多个超参数，收敛速度受设置影响 
 • 适用场景：需要高精度、严格约束的生产级搜索 


### **一句话总结：**

> OFA 超网络 + 预测器使得 NAS 的成本从 GPU 小时级骤降到秒级，让模型定制化变得切实可行。进化搜索结合效率与精度预测器，可在严苛的微控制器约束下找到高精度网络。搜索过程中，输入分辨率和宽度乘数往往是决定精度与效率的关键因子，合理调参能释放进化算法的强大搜索能力。面对极端约束，设计空间本身的限制成为瓶颈，这提示我们需将系统限制与架构联合优化。

---
