# **Lab 1 - 剪枝实验笔记**

## **1. 引言**

本次实验的主要目标是理解并实现剪枝（Pruning），从而减少神经网络模型的大小和推理延迟。剪枝技术通过去除不重要的神经元或权重来达到减少计算资源和存储的目的。本实验包含了两个主要部分：**细粒度剪枝**和**通道剪枝**。



## **2. 环境设置、数据集导入与简单评估**

首先，我们需要安装必要的库并加载数据集和预训练模型。实验中使用的是 **CIFAR-10** 数据集和 **VGG** 网络模型。
```
# 安装所需的库  
!pip install torchprofile
```


加载预训练的 VGG 模型：
```
checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"  
checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")  
model = VGG().cuda()  
model.load_state_dict(checkpoint['state_dict'])
```

我们使用 **CIFAR-10** 数据集进行训练。首先对数据进行预处理，包括随机裁剪、水平翻转等操作，然后将其传入数据加载器中：
```
dataset = CIFAR10(root="data/cifar10", train=True, download=True, transform=transforms["train"])  
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
```

在进行剪枝操作之前，首先对预训练的模型进行评估，查看其在 CIFAR-10 测试集上的准确率和模型大小。
```
dense_model_accuracy = evaluate(model, dataloader['test'])  
dense_model_size = get_model_size(model)
```

```
dense model has accuracy=92.95% 
dense model has size=35.20 MiB
```

在进行剪枝前，我们先观察模型中各层权重的分布。
```
def plot_weight_distribution(model, bins=256):  
    # 绘制权重分布的直方图  
    for name, param in model.named_parameters():  
        if param.dim() > 1:  
            ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True)
```
![](./pics/Pasted_image_20260414223557.png)

如图所示，除最后一层分类头外，其它层均服从均值为 0 的无偏正态分布，这意味着占很大比例的参数是可以被移除的，这为模型压缩留下了很大的空间。


## **3.模型压缩**

### **Q1 细粒度剪枝

细粒度剪枝是通过移除重要性最小的连接来减少模型的规模。常用的剪枝策略是基于权重的大小。具体操作是计算每个权重的大小，设定一个阈值，根据权重的绝对值来进行剪枝。
```
    # Step 1: calculate the #zeros (please use round())
    num_zeros = round(sparsity * num_elements)
    
    # Step 2: calculate the importance of weight
    importance = torch.abs(tensor)
    
    # Step 3: calculate the pruning threshold
    if num_zeros == 0:
        threshold = -float('inf')
    else:
        # kthvalue需要k为第k小的值，第num_zeros小的值即阈值
        threshold = torch.kthvalue(importance.flatten(), num_zeros).values
        
    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = importance > threshold
    
    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)
```


测试

通过一个简单的测试函数，验证剪枝前后的稀疏度变化。
```
def test_fine_grained_prune():  
    # 测试细粒度剪枝函数  
    test_tensor = torch.randn(5, 5)  
    mask = fine_grained_prune(test_tensor, target_sparsity=0.75)  
    print(f"剪枝前稀疏度: {get_sparsity(test_tensor)}")  
    print(f"剪枝后稀疏度: {get_sparsity(test_tensor)}")
```
![](./pics/Pasted_image_20260414230059.png)
```
*Test fine_grained_prune() 
	target sparsity: 0.75 
		sparsity before pruning: 0.04 
		sparsity after pruning: 0.76 
		sparsity of pruning mask: 0.76 
* Test passed.
```

### **Q2 稀疏度计算**

在一个 5 x 5 的矩阵中保留 10 个元素。

稀疏度为：
```
sparsity=1-target_element/total_element
```
在本实验中即为0.6。


### **D1 网络的灵敏度分析与参数量分析**

在灵敏度分析中，每次对于神经网络中的一层，逐步进行剪枝，以获得该层对于剪枝的灵敏度。

![](./pics/Pasted_image_20260414233152.png)

从图中可以看到大部分层中，随着稀疏度的增加，模型精度相应变低，不同层的敏感程度不同，第 0 个卷积层对稀疏度最敏感。

除了准确性，每一层的参数数量也会影响稀疏性选择的决策。参数更多的层需要更大的稀疏度。

![](./pics/Pasted_image_20260414233815.png)

从图中可以看到更深的层的参数量会更多，对这些层进行剪枝能够使模型得到更好的稀疏度的同时保留大量的能力。

###  **Q3 设置剪枝稀疏度** 

根据前面灵敏度分析结果和模型参数计算量，设置每一层剪枝时的稀疏度。
根据前文分析得知，越高的层应该保留更多参数，对于较低的层应该设计更高的稀疏度以减小模型大小。

```
sparsity_dict = {  
	# please modify the sparsity value of each layer 
	# please DO NOT modify the key of sparsity_dict    
	'backbone.conv0.weight': 0, 
    'backbone.conv1.weight': 0.4,
    'backbone.conv2.weight': 0.5,
    'backbone.conv3.weight': 0.6,
    'backbone.conv4.weight': 0.7,
    'backbone.conv5.weight': 0.8,
    'backbone.conv6.weight': 0.8,
    'backbone.conv7.weight': 0.8,
	'classifier.weight': 0  
}
```
![](./pics/Pasted_image_20260414234519.png)
```
Sparse model has size=8.28 MiB = 23.51% of dense model size
Sparse model has accuracy=85.69% before fintuning
```

同时，正如从前一设计的输出中可以看到，尽管细粒度剪枝减少了大部分模型权重，模型的准确性也下降了。因此，我们必须微调稀疏模型以恢复准确性。

```
Sparse model has size=8.28 MiB = 23.51% of dense model size
Sparse model has accuracy=92.84% after fintuning
```
在剪枝+微调后，模型恢复了大部分的的能力，同时压缩了模型大小。

### **Q4 通道剪枝** 

与细粒度剪枝不同，在通道剪枝中可以完全从张量中移除权重。也就是说，输出通道的数量会减少。与细粒度剪枝一样，通道剪枝可以对不同的层使用不同的剪枝率。然而，目前我们对所有层使用统一的剪枝率。目标是计算量减少 2 倍，这大约对应 30% 的统一剪枝率。
具体剪枝标准是只保留前 k 个通道。

```
def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio"""

    preserve_ratio = 1.0 - prune_ratio
    return int(round(channels * preserve_ratio))  

@torch.no_grad()
def channel_prune(model: nn.Module,prune_ratio: Union[List, float]) -> nn.Module:
    """Apply channel pruning to each of the conv layer in the backbone
    Note that for prune_ratio, we can either provide a floating-point number,indicating that we use a uniform pruning rate for all layers, or a list of numbers to indicate per-layer pruning rate."""

    # sanity check of provided prune_ratio
    assert isinstance(prune_ratio, (float, list))
    n_conv = len([m for m in model.backbone if isinstance(m, nn.Conv2d)])

    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
    if isinstance(prune_ratio, list):
        assert len(prune_ratio) == n_conv - 1
    else:  # convert float to list
        prune_ratio = [prune_ratio] * (n_conv - 1)

    # we prune the convs in the backbone with a uniform ratio
    model = copy.deepcopy(model)  # prevent overwrite
    
    # we only apply pruning to the backbone features
    all_convs = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]

    # apply pruning. we naively keep the first k channels
    assert len(all_convs) == len(all_bns)
    for i_ratio, p_ratio in enumerate(prune_ratio):
        prev_conv = all_convs[i_ratio]
        prev_bn = all_bns[i_ratio]
        next_conv = all_convs[i_ratio + 1]
        original_channels = prev_conv.out_channels  # same as next_conv.in_channels
        n_keep = get_num_channels_to_keep(original_channels, p_ratio)
        
        # prune the output of the previous conv and bn
        prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
        prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
        prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
        prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
        prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])
        
        # prune the input of the next conv
        next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])

    return model
```

接着评估在统一通道剪枝30%剪枝率之后模型的性能。如下所示，直接移除30%的通道会导致准确率严重下降。
```
pruned model has accuracy=28.14%
```

### **Q5 通道剪枝改进-以重要性排序通道**


在所有层中移除前 30% 的通道会导致显著的准确率下降。解决该问题的一种方法是寻找不太重要的通道权重进行移除：一个常用的重要性判定标准是使用对应每个输入通道权重的 Frobenius 范数。

> $importance_{i} = \|W_{i}\|_2, \;\; i = 0, 1, 2, \cdots, \mathrm{num\_channels} - 1$

我们可以将通道权重从重要到不重要进行排序，然后为每一层保留前 k 个通道。
```
def get_input_channel_importance(weight):
    in_channels = weight.shape[1]
    # torch.norm()函数已被弃用，故使用torch.linalg.vector_norm()
    # importances = []
    # compute the importance for each input channel
    # for i_c in range(weight.shape[1]):
    #     channel_weight = weight.detach()[:, i_c]
    #     ##################### YOUR CODE STARTS HERE #####################
    #     importance = torch.norm(channel_weight, p=2)
    #     ##################### YOUR CODE ENDS HERE #####################
    #     importances.append(importance.view(1))
    # return torch.cat(importances)
    return torch.linalg.vector_norm(weight, ord=2, dim=(0, 2, 3))

@torch.no_grad()
def apply_channel_sorting(model):
    model = copy.deepcopy(model)  # do not modify the original model
    
    # fetch all the conv and bn layers from the backbone
    all_convs = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
    
    # iterate through conv layers
    for i_conv in range(len(all_convs) - 1):
    
        # each channel sorting index, we need to apply it to:
        # - the output dimension of the previous conv
        # - the previous BN layer
        # - the input dimension of the next conv (we compute importance here)
        prev_conv = all_convs[i_conv]
        prev_bn = all_bns[i_conv]
        next_conv = all_convs[i_conv + 1]
        
        # note that we always compute the importance according to input channels
        importance = get_input_channel_importance(next_conv.weight)
        
        # sorting from large to small
        sort_idx = torch.argsort(importance, descending=True)

        # apply to previous conv and its following bn
        prev_conv.weight.copy_(torch.index_select(
            prev_conv.weight.detach(), 0, sort_idx))
        for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
            tensor_to_apply = getattr(prev_bn, tensor_name)
            tensor_to_apply.copy_(
                torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
            )

        # apply to the next conv input (hint: one line of code)
        next_conv.weight.copy_(torch.index_select(next_conv.weight.detach(), 1, sort_idx))

    return model
```

评估模型性能。

```sorted model has accuracy=92.95% ```
```* Check passed.```
```pruned model has accuracy=36.81%```

相比没有计算重要性的通道剪枝，改进剪枝后的模型的准确率提升到 36.81。经过微调后恢复为 92.95%。

### **Q6 数学分析**

1.为什么剪枝30%，但是计算量减少了大约 50%？

>$FLOPs = K\times K\times C_{in}\times C_{out}\times H \times W$

其中输入和输出通道都变为原来的 70%，因而总计算量变为原来的 49%。


2.为什么延迟（latency）的减少比例略小于计算量的减少比例？

延迟不仅仅来源于计算，还来自于数据的搬运。

### **Q7 实际应用处理**

1.讨论一下 fine-grained pruning 和 channel pruning 的优缺点。  

细粒度剪枝：压缩率更高、对硬件不友好、延迟高；  
通道剪枝：压缩率低、硬件友好、延迟低、易于微调。  

2.如果想在智能手机上加速模型，使用哪种方案更合适。  

通道剪枝。智能手机上一般缺乏对于稀疏矩阵的支持，选取对硬件更友好的方案。


## **4.总结**

本次实验主要学习了两种神经网络剪枝方法：**细粒度剪枝**和**通道剪枝**。细粒度剪枝基于权重的绝对值大小移除不重要的连接，能够大幅压缩模型大小，但对硬件不友好，难以直接降低推理延迟。通道剪枝则直接移除整个输出通道，硬件友好且易于微调，但压缩率相对较低。

在实验中进一步掌握了**灵敏度分析**和**参数分布分析**，用于指导不同层设置不同的剪枝稀疏度。通过计算输入通道的 Frobenius 范数对通道重要性进行排序，可以显著改善通道剪枝的效果，避免简单统一剪枝导致的精度大幅下降。
