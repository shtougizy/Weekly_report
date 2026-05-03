# **Lab2-量化神经网络实验笔记**

## **1. 引言**

本次实验的核心目标是学习并实现量化技术，以减少模型的大小和推理延迟。量化技术通过将权重和激活值压缩为更少的位数（例如8位、16位），来降低内存消耗并加速模型推理。

量化技术有几种类型，本实验主要涉及：

- **K-means量化**（K-means Quantization）
- **线性量化**（Linear Quantization）

## **2. 环境设置，加载数据集和模型，基准测试**

首先，我们需要安装所需的库，并下载CIFAR-10数据集以及预训练的VGG模型。
```
!pip install torchprofile 1>/dev/null  
! pip install fast-pytorch-kmeans 1>/dev/null
```

我们使用CIFAR-10数据集进行训练，并加载预训练的VGG网络模型。
```
checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"  
checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")  
model = VGG().cuda()  
model.load_state_dict(checkpoint['state_dict'])
```

对于模型在FP32精确度下的准确率和大小做一个评估。
```
eval: 0%| | 0/20 [00:00<?, ?it/s]
fp32 model has accuracy=92.95%
fp32 model has size=35.20 MiB
```

## **3. 模型量化**

量化的目标是减少模型的内存占用，并加速推理速度。通过将浮点数表示的权重转换为低精度的整数（如8位整数），可以实现这一目标。在本实验中，我们将通过两种量化方法进行探讨。

### **3.1 K-means量化**

K-means量化是一种通过聚类算法来优化权重的技术。通过K-means聚类，将权重值分成若干组，每组的平均值代表该组的所有权重。在推理时，权重将被量化为这些代表值。
```
def k_means_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):

    if codebook is None:

        # get number of clusters based on the quantization precision
        n_clusters = 2**bitwidth

        # use k-means to get the quantization centroids
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)

    # decode the codebook into k-means quantized tensor for inference
    quantized_tensor = codebook.centroids[codebook.labels].view(fp32_tensor.shape)

    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook
```
在该部分代码中，使用了K-means聚类算法对权重进行量化，目标是将权重的值压缩到特定的bitwidth。在本次实验中，我们将浮点型权重值映射到2位整数，从而减少内存占用。
下面通过在一个虚拟张量上应用上面的函数来验证定义的 K-means 量化的功能。

```test_k_means_quantize()```
```
tensor([[-0.3747, 0.0874, 0.3200, -0.4868, 0.4404], 
		[-0.0402, 0.2322, -0.2024, -0.4986, 0.1814], 
		[ 0.3102, -0.3942, -0.2030, 0.0883, -0.4741], 
		[-0.1592, -0.0777, -0.3946, -0.2128, 0.2675], 
		[ 0.0611, -0.1933, -0.4350, 0.2928, -0.1087]]) 
* Test k_means_quantize() 
	target bitwidth: 2 bits 
		num unique values before k-means quantization: 25 
		num unique values after k-means quantization: 4 
* Test passed.
```
![](./pics/Pasted_image_20260415223103.png)
该代码单元执行了2位K-means量化，并绘制了量化前后的张量。每个簇都用独特的颜色表示。在量化后的张量中绘制了4种不同的颜色，即分为了四个簇。对于n位的K-means量化，应该分类为$2^n$个簇。

下一步是将K-means量化函数封装到一个类中，用于对整个模型进行量化。在类 KMeansQuantizer中，我们需要保存码本记录（即centroids和 labels），以便在模型权重发生变化时可以应用或更新这些码本；并使用这个函数，将模型量化为8/4/2位。

```
8-bit k-means quantized model has size=8.80 MiB
8-bit k-means quantized model has accuracy=92.76%
4-bit k-means quantized model has size=4.40 MiB
4-bit k-means quantized model has accuracy=79.07%
2-bit k-means quantized model has size=2.20 MiB
2-bit k-means quantized model has accuracy=10.00%
```

观察到模型的大小相比原来的大小缩小了$32/n$位，准确率出现了加速下滑现象。2位量化事实上让模型退化为了一个二分类器，在多选一的任务下表现极差。

### **3.2 训练感知的K-Means量化（QAT）**

从上个单元格的结果可以看出，将模型量化为较低位宽时，准确率会显著下降。因此，我们需要进行量化感知训练以恢复准确率。
在k-means量化感知训练过程中，权重也会被更新。以下是量化后权重的梯度计算公式：

> $\frac{\partial \mathcal{L} }{\partial C_k} = \sum_{j} \frac{\partial \mathcal{L} }{\partial W_{j}} \frac{\partial W_{j} }{\partial C_k} = \sum_{j} \frac{\partial \mathcal{L} }{\partial W_{j}} \mathbf{1}(I_{j}=k)$

其中 $\mathcal{L}$ 是损失函数，$C_k$ 是第 $k$ 个质心，$I_{j}$ 是权重 $W_{j}$ 的标签。$\mathbf{1}()$ 是指示函数，$\mathbf{1}(I_{j}=k)$ 表示若 $I_{j}=k$ 则为 $1$，否则为 $0$，即 $I_{j}==k$。在本实验中，为简便起见，我们直接使用相同簇的原始权重的均值作为量化后的该簇更新后的值。

> $C_k = \frac{\sum_{j}W_{j}\mathbf{1}(I_{j}=k)}{\sum_{j}\mathbf{1}(I_{j}=k)}$

```
def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
    # update the centroids in the codebook using updated fp32_tensor
    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)
    for k in range(n_clusters):
        codebook.centroids[k] = fp32_tensor[codebook.labels==k].mean()
```

以下是QAT的结果：
```
8-bit k-means quantized model has accuracy=92.76% before quantization-aware training 
No need for quantization-aware training since accuracy drop=0.19% is smaller than threshold=0.50%

4-bit k-means quantized model has accuracy=79.07% before quantization-aware training 
Quantization-aware training due to accuracy drop=13.88% is larger than threshold=0.50%
Epoch 0 Accuracy 92.47% / Best Accuracy: 92.47%

2-bit k-means quantized model has accuracy=10.00% before quantization-aware training Quantization-aware training due to accuracy drop=82.95% is larger than threshold=0.50%
# 省略四次训练结果
Epoch 4 Accuracy 91.20% / Best Accuracy: 91.20%
```
在2位量化下掉点率极高，多轮微调后也不能恢复到原始性能。

### **3.3 线性量化**

在本节中，我们将实现并执行线性量化。
线性量化在进行范围截断和缩放后，直接将浮点数值舍入到最接近的量化整数。
线性量化可以表示为：
$r = S(q-Z)$
其中 $r$ 是浮点数实数，$q$ 是 $n$ 位整数，$Z$ 是 $n$ 位整数，$S$ 是浮点数实数。$Z$ 是量化的零点，$S$ 是量化的缩放因子。常数 $Z$ 和 $S$ 均为量化参数。
采用补码表示法表示n位整数。
```
def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max
```

```
def linear_quantize(fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8) -> torch.Tensor:
    assert(fp_tensor.dtype == torch.float)
    assert(isinstance(scale, float) or
           (scale.dtype == torch.float and scale.dim() == fp_tensor.dim()))
    assert(isinstance(zero_point, int) or
           (zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim()))
           
    # Step 1: scale the fp_tensor
    scaled_tensor = fp_tensor/scale

    # Step 2: round the floating value to integer value
    rounded_tensor = scaled_tensor.round()
    rounded_tensor = rounded_tensor.to(dtype)

    # Step 3: shift the rounded_tensor to make zero_point 0
    shifted_tensor = rounded_tensor+zero_point

    # Step 4: clamp the shifted_tensor to lie in bitwidth-bit range
    # 将溢出的结果压缩到范围之内
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    quantized_tensor = shifted_tensor.clamp_(quantized_min, quantized_max)
    
    return quantized_tensor


test_linear_quantize()
```

以下是测试结果：
```
* Test linear_quantize() 
  target bitwidth: 2 bits 
  scale: 0.3333333333333333 
  zero point: -1 
* Test passed.
```
![](./pics/Pasted_image_20260415233842.png)

### **3.4 确定线性量化中的缩放因子与零点**

> $r_{\mathrm{max}} = S(q_{\mathrm{max}}-Z)$
> $r_{\mathrm{min}} = S(q_{\mathrm{min}}-Z)$
> $S=(r_{\mathrm{max}} - r_{\mathrm{min}}) / (q_{\mathrm{max}} - q_{\mathrm{min}})$
>  $Z = \mathrm{int}(\mathrm{round}(q_{\mathrm{min}} - r_{\mathrm{min}} / S))$ 

```
def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):

    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()

    # calculating scale
    scale = (fp_max-fp_min)/(quantized_max-quantized_min)

    # calculating zero_point
    zero_point = int(round(quantized_min-fp_min/scale))

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < quantized_min:
        zero_point = quantized_min
    elif zero_point > quantized_max:
        zero_point = quantized_max
    else: # convert from float to int using round()
        zero_point = round(zero_point)

    return scale, int(zero_point)
```

```
def linear_quantize_feature(fp_tensor, bitwidth):
    scale, zero_point = get_quantization_scale_and_zero_point(fp_tensor, bitwidth)
    quantized_tensor = linear_quantize(fp_tensor, bitwidth, scale, zero_point)
    
    return quantized_tensor, scale, zero_point
```

同时观察权重值的分布。
![](./pics/Pasted_image_20260415235513.png)

从上面的直方图可以看出，权重值的分布几乎关于0对称（本例中的分类器除外）。因此，在量化权重时，我们通常将零点设为 $Z=0$。
由 $r = S(q-Z)$ 可得：
> $r_{\mathrm{max}} = S \cdot q_{\mathrm{max}}$

于是：
> $S = r_{\mathrm{max}} / q_{\mathrm{max}}$

我们直接使用权重的最大绝对值作为 $r_{\mathrm{max}}$。

此外，对于二维卷积，权重张量是一个四维张量，形状为（输出通道数，输入通道数，卷积核高度，卷积核宽度）。经验表明，对卷积核进行量化时，按照输出通道逐通道量化能够取得更好的表现。
```
def linear_quantize_weight_per_channel(tensor, bitwidth):

    dim_output_channels = 0
    num_output_channels = tensor.shape[dim_output_channels]
    scale = torch.zeros(num_output_channels, device=tensor.device)

    for oc in range(num_output_channels):
        _subtensor = tensor.select(dim_output_channels, oc)
        _scale = get_quantization_scale_for_weight(_subtensor, bitwidth)
        scale[oc] = _scale

    scale_shape = [1] * tensor.dim()
    scale_shape[dim_output_channels] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_quantize(tensor, bitwidth, scale, zero_point=0)

    return quantized_tensor, scale, 0
```

如下是在不同位宽下对权重应用线性量化时的权重分布和模型大小的展示。

![](./pics/Pasted_image_20260415235941.png)
![](./pics/Pasted_image_20260415235944.png)


## **4. 量化推理**

量化后，卷积层和全连接层的推理过程也会发生变化。

回顾 $r = S(q-Z)$，我们有：

$$
r_{\mathrm{input}} = S_{\mathrm{input}}(q_{\mathrm{input}}-Z_{\mathrm{input}})
$$

$$
r_{\mathrm{weight}} = S_{\mathrm{weight}}(q_{\mathrm{weight}}-Z_{\mathrm{weight}})
$$

$$
r_{\mathrm{bias}} = S_{\mathrm{bias}}(q_{\mathrm{bias}}-Z_{\mathrm{bias}})
$$

由于 $Z_{\mathrm{weight}}=0$，所以 $r_{\mathrm{weight}} = S_{\mathrm{weight}}q_{\mathrm{weight}}$。

浮点卷积可以写为：

$$
\begin{aligned}
r_{\mathrm{output}} &= \mathrm{CONV}[r_{\mathrm{input}}, r_{\mathrm{weight}}] + r_{\mathrm{bias}} \\
&= \mathrm{CONV}[S_{\mathrm{input}}(q_{\mathrm{input}}-Z_{\mathrm{input}}), S_{\mathrm{weight}}q_{\mathrm{weight}}] + S_{\mathrm{bias}}(q_{\mathrm{bias}}-Z_{\mathrm{bias}}) \\
&= \mathrm{CONV}[q_{\mathrm{input}}-Z_{\mathrm{input}}, q_{\mathrm{weight}}]\cdot (S_{\mathrm{input}} \cdot S_{\mathrm{weight}}) + S_{\mathrm{bias}}(q_{\mathrm{bias}}-Z_{\mathrm{bias}})
\end{aligned}
$$

为了进一步简化计算，我们可以令：

$$
Z_{\mathrm{bias}} = 0
$$

$$
S_{\mathrm{bias}} = S_{\mathrm{input}} \cdot S_{\mathrm{weight}}
$$

于是：

$$
\begin{aligned}
r_{\mathrm{output}} &= (\mathrm{CONV}[q_{\mathrm{input}}-Z_{\mathrm{input}}, q_{\mathrm{weight}}] + q_{\mathrm{bias}})\cdot (S_{\mathrm{input}} \cdot S_{\mathrm{weight}}) \\
&= (\mathrm{CONV}[q_{\mathrm{input}}, q_{\mathrm{weight}}] - \mathrm{CONV}[Z_{\mathrm{input}}, q_{\mathrm{weight}}] + q_{\mathrm{bias}})\cdot (S_{\mathrm{input}}S_{\mathrm{weight}})
\end{aligned}
$$

由于：

$$r_{\mathrm{output}} = S_{\mathrm{output}}(q_{\mathrm{output}}-Z_{\mathrm{output}})$$

我们有：

$$
S_{\mathrm{output}}(q_{\mathrm{output}}-Z_{\mathrm{output}}) = (\mathrm{CONV}[q_{\mathrm{input}}, q_{\mathrm{weight}}] - \mathrm{CONV}[Z_{\mathrm{input}}, q_{\mathrm{weight}}] + q_{\mathrm{bias}})\cdot (S_{\mathrm{input}} S_{\mathrm{weight}})
$$

因此：

$$
q_{\mathrm{output}} = (\mathrm{CONV}[q_{\mathrm{input}}, q_{\mathrm{weight}}] - \mathrm{CONV}[Z_{\mathrm{input}}, q_{\mathrm{weight}}] + q_{\mathrm{bias}})\cdot (S_{\mathrm{input}}S_{\mathrm{weight}} / S_{\mathrm{output}}) + Z_{\mathrm{output}}
$$

由于 $Z_{\mathrm{input}}$、$q_{\mathrm{weight}}$、$q_{\mathrm{bias}}$ 在推理前就已确定，则有：

$$
q_{\mathrm{output}} = (\mathrm{CONV}[q_{\mathrm{input}}, q_{\mathrm{weight}}] + Q_{\mathrm{bias}}) \cdot (S_{\mathrm{input}}S_{\mathrm{weight}} / S_{\mathrm{output}}) + Z_{\mathrm{output}}
$$

类似地，对于全连接层，我们有：

$$
q_{\mathrm{output}} = (\mathrm{Linear}[q_{\mathrm{input}}, q_{\mathrm{weight}}] + Q_{\mathrm{bias}})\cdot (S_{\mathrm{input}} \cdot S_{\mathrm{weight}} / S_{\mathrm{output}}) + Z_{\mathrm{output}}
$$

其中：

$$
Q_{\mathrm{bias}} = q_{\mathrm{bias}} - \mathrm{Linear}[Z_{\mathrm{input}}, q_{\mathrm{weight}}]
$$
$$
Q_{\mathrm{bias}} = q_{\mathrm{bias}} - \mathrm{Linear}[Z_{\mathrm{input}}, q_{\mathrm{weight}}]
$$
```
def linear_quantize_bias_per_output_channel(bias, weight_scale, input_scale):

    assert(bias.dim() == 1)
    assert(bias.dtype == torch.float)
    assert(isinstance(input_scale, float))
    
    if isinstance(weight_scale, torch.Tensor):
        assert(weight_scale.dtype == torch.float)
        weight_scale = weight_scale.view(-1)
        assert(bias.numel() == weight_scale.numel())

    bias_scale = weight_scale*input_scale
    quantized_bias = linear_quantize(bias, 32, bias_scale,zero_point=0, dtype=torch.int32)

    return quantized_bias, bias_scale, 0
```

```
def quantized_linear(input, weight, bias, feature_bitwidth, weight_bitwidth,input_zero_point, output_zero_point,input_scale, weight_scale, output_scale):

    assert(input.dtype == torch.int8)
    assert(weight.dtype == input.dtype)
    assert(bias is None or bias.dtype == torch.int32)
    assert(isinstance(input_zero_point, int))
    assert(isinstance(output_zero_point, int))
    assert(isinstance(input_scale, float))
    assert(isinstance(output_scale, float))
    assert(weight_scale.dtype == torch.float)

    # Step 1: integer-based fully-connected (8-bit multiplication with 32-bit accumulation)
    if 'cpu' in input.device.type:
        output = torch.nn.functional.linear(input.to(torch.int32), weight.to(torch.int32), bias)
    else:
        output = torch.nn.functional.linear(input.float(), weight.float(), bias.float())

    # Step 2: scale the output
    #         hint: 1. scales are floating numbers, we need to convert output to float as well
    #               2. the shape of weight scale is [oc, 1, 1, 1] while the shape of output is [batch_size, oc]
    output = output*(input_scale*weight_scale/output_scale).swapaxes(0,1)

    # Step 3: shift output by output_zero_point
    output = output+output_zero_point

    # Make sure all value lies in the bitwidth-bit range
    output=
    output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)

    return output
```

```
test_quantized_fc()
```
```
* Test quantized_fc() 
	  target bitwidth: 2 
	  bits batch size: 4 
	  input channels: 8 
	  output channels: 8 
* Test passed.
```
![](./pics/Pasted_image_20260416111305.png)

量化卷积与量化全连接层的操作相同。

## **5. int8量化实现**

首先，把一个批归一化层融合到它前面的卷积层中，这是量化前的标准做法，为了减少推理过程中的额外乘法运算。
然后使用一些样本数据运行模型，以获取每个特征图的范围，从而得到特征图的范围并计算它们对应的缩放因子和零点。
最后，进行模型量化。为了运行量化模型，需要进行额外的预处理，将输入数据从 (0, 1) 范围映射到int8的 (-128, 127) 范围内。
```
print(quantized_model)

def extra_preprocess(x):

    # hint: you need to convert the original fp32 input of range (0, 1)
    #  into int8 format of range (-128, 127)
    x_scale,x_zero_point=get_quantization_scale_and_zero_point(x,8)
    return linear_quantize(x,8,x_scale,x_zero_point)
  
int8_model_accuracy = evaluate(quantized_model, dataloader['test'],
                               extra_preprocess=[extra_preprocess])
print(f"int8 model has accuracy={int8_model_accuracy:.2f}%")
```

```
VGG( (backbone): Sequential( 
	(0): QuantizedConv2d() 
	(1): QuantizedConv2d() 
	(2): QuantizedMaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) 
	(3): QuantizedConv2d() 
	(4): QuantizedConv2d() 
	(5): QuantizedMaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) 
	(6): QuantizedConv2d() 
	(7): QuantizedConv2d() 
	(8): QuantizedMaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) 
	(9): QuantizedConv2d() 
	(10): QuantizedConv2d() 
	(11): QuantizedMaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) 
	(12): QuantizedAvgPool2d(kernel_size=2, stride=2, padding=0) ) (classifier): QuantizedLinear() )
eval: 0%| | 0/20 [00:00<?, ?it/s]
int8 model has accuracy=92.90%
```

## **6.总结**

在本次实验中，我们首先学习了两种基础的量化方法：K‑means量化和线性量化。对于K‑means量化，我们通过聚类将权重压缩至2/4/8位，但发现低位宽（如2位）会导致准确率急剧下降（掉点率超过80%）。为此，我们引入了量化感知训练，在微调过程中根据簇内原始权重的均值更新质心。对于线性量化，我们推导了缩放因子S与零点Z的计算公式，并观察到权重分布近似对称，因此将零点设为0；同时针对卷积层，我们采用了逐输出通道的独立量化策略，以提升量化后的性能。

在量化推理的实现中，我们首先将批归一化层融合到卷积层中，减少推理时的额外计算。接着，我们使用少量样本数据运行模型，收集每个特征图的动态范围，从而计算出对应的缩放因子和零点。最后，我们将Conv2d、Linear等层替换为自定义的量化版本，并对输入数据做预处理：将原始的(0,1)浮点输入映射到int8的(-128,127)范围。经过上述步骤，模型与原始FP32模型的准确率几乎持平，而模型大小从压缩至8位。这验证了量化技术在显著减小模型体积的同时，能够通过合理的量化和训练策略保持较高的精度。
