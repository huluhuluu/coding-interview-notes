---
title: "NumPy / PyTorch 常用操作"
date: 2026-10-20T16:40:00+08:00
lastmod: 2026-10-20T16:40:00+08:00
draft: false
description: "NumPy / PyTorch 的创建、索引、广播、维度变换、矩阵乘、归约、数据转换与梯度操作"
slug: "numpy-torch-apis"
tags: ["python", "numpy", "pytorch", "张量"]
categories: ["笔试重点"]
comments: true
math: true
---

# NumPy / PyTorch 常用操作

- [NumPy / PyTorch 常用操作](#numpy--pytorch-常用操作)
  - [1. 创建、类型与设备](#1-创建类型与设备)
  - [2. reshape、转置与增删维度](#2-reshape转置与增删维度)
  - [3. 索引、切片与 gather](#3-索引切片与-gather)
    - [3.1 基本索引与布尔索引](#31-基本索引与布尔索引)
    - [3.2 成对索引与按轴选取](#32-成对索引与按轴选取)
  - [4. 广播、expand 与 repeat](#4-广播expand-与-repeat)
  - [5. cat、stack 与拆分](#5-catstack-与拆分)
  - [6. 归约、排序与 Top-K](#6-归约排序与-top-k)
  - [7. 逐元素运算、矩阵乘与 einsum](#7-逐元素运算矩阵乘与-einsum)
  - [8. 拷贝、共享与 NumPy 转换](#8-拷贝共享与-numpy-转换)
  - [9. 梯度与标量](#9-梯度与标量)
  - [10. LLM 常用接口组合](#10-llm-常用接口组合)
  - [参考资料](#参考资料)

以 NumPy 1.26+、PyTorch 2.2+ 为基线，示例使用 CPU。先执行公共导入，后续各代码块自行创建输入。Python 列表与字符串见 [Python 常用接口](../python-apis/index.md)，算子实现见 [LLM 核心算子手撕](../llm-operator-implementations/index.md)。

```python
import numpy as np
import torch
import torch.nn.functional as F
```

## 1. 创建、类型与设备

| 目的 | NumPy | PyTorch |
|------|------|------|
| 从数据创建 | `np.array(data,dtype=...)` | `torch.tensor(data,dtype=...)` |
| 全 0 / 全 1 | `np.zeros/ones(shape)` | `torch.zeros/ones(shape)` |
| 指定填充值 | `np.full(shape,value)` | `torch.full(shape,value)` |
| 等间隔整数 | `np.arange(start,stop,step)` | `torch.arange(start,end,step)` |
| 指定点数的等间隔数 | `np.linspace(start,stop,num)` | `torch.linspace(start,end,steps)` |
| 形状 / 维数 | `a.shape` / `a.ndim` | `t.shape` / `t.ndim` |
| 元素总数 | `a.size` | `t.numel()` |
| 转换类型 | `a.astype(dtype)` | `t.to(dtype=...)` / `t.float()` / `t.long()` |

```python
a = np.arange(6, dtype=np.float32).reshape(2, 3)
t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
ids = torch.tensor([[1, 2, 3]], dtype=torch.long) # 类别与 gather 索引通常用 int64
z = torch.zeros_like(t)                         # 沿用输入的形状、dtype 和 device
converted = t.to(dtype=torch.float64)           # 返回转换结果，不原地修改 t
rng = np.random.default_rng(0)
np_samples = rng.standard_normal((2, 3)).astype(np.float32) # 使用独立随机生成器
torch_samples = torch.randn(2, 3, generator=torch.Generator().manual_seed(0))
assert a.size == t.numel() == 6
```

`arange` 不包含右端点，`linspace` 默认包含两端。`empty` 只分配空间，内容未初始化。PyTorch 的 `.to(device)` 返回目标设备上的结果，应接住返回值；参与同一计算的张量通常需要位于同一设备。

## 2. reshape、转置与增删维度

| 操作 | NumPy | PyTorch |
|------|------|------|
| 重排形状 | `a.reshape(...)` | `t.reshape(...)` / `t.view(...)` |
| 交换两个轴 | `a.swapaxes(i,j)` | `t.transpose(i,j)` |
| 指定全部轴顺序 | `a.transpose(axes)` | `t.permute(dims)` |
| 增加大小为 1 的轴 | `np.expand_dims(a,axis)` / `a[None,...]` | `t.unsqueeze(dim)` |
| 删除大小为 1 的轴 | `np.squeeze(a,axis)` | `t.squeeze(dim)` |
| 展平 | `a.reshape(-1)` | `t.flatten()` |

```python
a = np.arange(24).reshape(2, 3, 4)
t = torch.arange(24).reshape(2, 3, 4)
np_transposed = a.transpose(0, 2, 1)            # [2,4,3]，改变轴顺序
transposed = t.transpose(1, 2)                 # [2,4,3]，通常不再连续
flattened = transposed.contiguous().view(2, -1) # [2,12]，先整理布局再 view
inserted = t.unsqueeze(1)                     # [2,1,3,4]
restored = inserted.squeeze(1)                # [2,3,4]，只移除指定轴
assert restored.shape == t.shape
```

`reshape` 保持遍历顺序，只调整形状，不能替代转置。NumPy 的 `.T` 会反转全部轴，不能把高维数组的 `.T` 一概理解为交换最后两轴。

PyTorch `view` 要求 stride 与目标形状兼容；`reshape` 可能返回视图，也可能复制。NumPy `reshape` 同样可能复制。不带轴参数的 `squeeze()` 会移除所有大小为 1 的轴，B=1 时可能误删 batch 维。参见 [NumPy Copies and Views](https://numpy.org/doc/stable/user/basics.copies.html) 和 [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)。

指定轴不为 1 时，NumPy `squeeze(axis=...)` 会报错，Torch `squeeze(dim)` 保持原状。

## 3. 索引、切片与 gather

### 3.1 基本索引与布尔索引

```python
a = np.arange(12).reshape(3, 4)
t = torch.arange(12).reshape(3, 4)
column = a[:, 1]           # [3]，整数索引移除被索引轴
column_kept = t[:, 1:2]    # [3,1]，切片保留该轴
np_reverse = a[:, ::-1]    # NumPy 支持负步长切片
reverse = torch.flip(t, dims=[-1]) # PyTorch 不支持 t[:,::-1]，用 flip
selected = t[(t > 3) & (t < 8)]    # [4,5,6,7]，本例 mask 与 t 同形状，结果为一维
replaced = torch.where(t > 5, t, torch.zeros_like(t)) # 同形状条件选择
rows, cols = torch.nonzero(t > 9, as_tuple=True)     # 满足条件的各轴下标
```

数组 / 张量条件用 `&`、`|`、`~`，比较式两侧加括号；不能用 Python 的 `and/or` 逐元素组合。`where(condition,x,y)` 与只给 condition 的取下标形式不同。

基本切片在 NumPy/PyTorch 中返回视图，修改结果可能影响原数据；整数数组或布尔高级索引读取返回副本。Python list 切片则创建新列表。直接赋值 `a[index]=value` 仍会修改原数组，不能根据“读取返回副本”推断赋值无效。参见 [NumPy 索引](https://numpy.org/doc/stable/user/basics.indexing.html)。

### 3.2 成对索引与按轴选取

```python
a = np.array([[10, 11, 12], [20, 21, 22]])
t = torch.tensor([[10, 11, 12], [20, 21, 22]])
paired = a[[0, 1], [2, 0]]           # [12,20]，两个坐标成对使用
cartesian = a[np.ix_([0, 1], [2, 0])] # [[12,10],[22,20]]，行列笛卡尔积
indices = torch.tensor([[2, 0], [1, 1]])
gathered = torch.gather(t, dim=1, index=indices) # [[12,10],[21,21]]，每行使用自己的索引
np_gathered = np.take_along_axis(a, indices.numpy(), axis=1)
assert np.array_equal(gathered.numpy(), np_gathered)
```

`torch.gather` 的 input 与 index 维数相同，index 使用 long；除选取轴外，index 的尺寸不能超过 input 的对应尺寸，返回形状与 index 相同。`index_select` 则沿指定轴对所有其他位置使用同一个一维索引。参见 [gather](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)。

## 4. 广播、expand 与 repeat

广播从最右侧轴对齐，各轴尺寸相等或其中一个为 1 才兼容；缺失的前导轴按 1 处理。

```python
x = torch.zeros(2, 3, 4)
bias = torch.arange(4, dtype=torch.float32)
y = x + bias                     # [2,3,4] + [4]，沿末维广播
row = torch.tensor([[1., 2., 3.]])
expanded = row.expand(4, 3)       # [1,3] -> [4,3]，共享存储，不复制四份
writable = expanded.clone()      # 需要逐位置独立写入时先复制
np_interleaved = np.repeat([1, 2], 2)       # [1,1,2,2]
np_tiled = np.tile([1, 2], 2)               # [1,2,1,2]
tiled = torch.tensor([1, 2]).repeat(2)      # [1,2,1,2]，复制完整模式
interleaved = torch.repeat_interleave(torch.tensor([1, 2]), 2) # [1,1,2,2]
```

NumPy `broadcast_to` 返回只读广播视图。Torch `expand` 只能扩展大小为 1 的轴，并可能用 stride=0 让多个位置指向相同元素；避免对该视图做需要独立位置的原地写入。`repeat` 会复制数据。参见 [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) 和 [expand](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html)。

## 5. cat、stack 与拆分

```python
a, b = np.array([1, 2, 3]), np.array([4, 5, 6])
x, y = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])
np_joined = np.concatenate((a, b), axis=0) # [6]，沿已有轴拼接
np_stacked = np.stack((a, b), axis=0)      # [2,3]，新增一个轴
joined = torch.cat((x, y), dim=0)         # [6]
stacked = torch.stack((x, y), dim=0)      # [2,3]
chunks = torch.chunk(torch.arange(7), 3)  # 三块，长度为 3、3、1
sizes = torch.split(torch.arange(6), [1, 2, 3]) # 按明确长度拆分
np_chunks = np.array_split(np.arange(7), 3)    # 长度为 3、2、2
```

cat/concatenate 要求除拼接轴外的尺寸一致；stack 要求所有输入形状相同。`np.split(a,n)` 要求能等分，`array_split` 允许不等长。`torch.chunk` 返回的块数可能少于指定值；`torch.split(t,n)` 的 n 表示每块大小，不是块数。

## 6. 归约、排序与 Top-K

| 操作 | NumPy | PyTorch |
|------|------|------|
| 求和 / 均值 | `sum/mean(axis=...,keepdims=True)` | `sum/mean(dim=...,keepdim=True)` |
| 最大值 | `np.max(a,axis=...)` | `torch.amax(t,dim=...)` |
| 最大值与下标 | 分别 `max`、`argmax` | `t.max(dim=...)` 返回 values/indices |
| 排序值 / 下标 | `np.sort` / `np.argsort` | `torch.sort` / `torch.argsort` |
| Top-K | 用 argsort 或 argpartition 组合 | `torch.topk` 返回 values/indices |

```python
a = np.array([[1., 3., 2.], [4., 0., 5.]], dtype=np.float32)
t = torch.from_numpy(a)
means = t.mean(dim=-1, keepdim=True)   # [2,1]，保留维度便于广播
centered = t - means                 # [2,3]
max_values, max_indices = t.max(dim=1) # [3,5] 与 [1,2]
top_values, top_indices = torch.topk(t, k=2, dim=-1) # [2,2]，默认降序
np_top_indices = np.argsort(-a, axis=-1)[:, :2]
np_top_values = np.take_along_axis(a, np_top_indices, axis=-1)
assert np.allclose(top_values.numpy(), np_top_values)

# 与 LayerNorm 对照时，方差除以 D，须显式关闭无偏修正。
np_var = np.var(a, axis=-1, ddof=0)
torch_var = t.var(dim=-1, unbiased=False)
assert np.allclose(np_var, torch_var.numpy())
```

NumPy `np.sort` 返回排序副本，`a.sort()` 原地修改。Torch `sort` 返回排序值和原下标，不原地修改输入。`topk` 不保证相同值的下标稳定；比较结果时不能仅凭相同值要求相同排列。

整数 Torch 张量求 mean 通常需先 `.float()`。NumPy var 默认 `ddof=0`，Torch var 默认带无偏修正；手写归一化时应显式对齐。

## 7. 逐元素运算、矩阵乘与 einsum

`*` 是逐元素乘；`@` / matmul 是矩阵乘。批量矩阵乘只把最后两轴当作矩阵，前导轴按广播规则处理。

```python
a = np.ones((2, 3, 4), dtype=np.float32)
b = np.ones((2, 4, 5), dtype=np.float32)
np_product = a @ b                         # [2,3,5]
np_einsum = np.einsum("bmk,bkn->bmn", a, b) # 同一个批量矩阵乘的下标写法
x, y = torch.from_numpy(a), torch.from_numpy(b)
product = x @ y                            # [2,3,5]
by_einsum = torch.einsum("bmk,bkn->bmn", x, y) # 对 k 求和，保留 b/m/n
assert torch.allclose(product, by_einsum)
assert np.allclose(np_product, np_einsum)

q = torch.ones(2, 4, 3, 8)                 # [B,H,Lq,d]
k = torch.ones(2, 4, 5, 8)                 # [B,H,Lkv,d]
scores = q @ k.transpose(-1, -2) / (8 ** 0.5) # [B,H,Lq,Lkv]
probabilities = torch.softmax(scores, dim=-1) # 沿 key 轴归一化
```

`torch.bmm` 只接收三维输入且不做 batch 广播；`torch.matmul` 更通用。`torch.dot` 只处理一维向量；NumPy 的高维 dot 有自己的轴规则，不能一概替代 matmul。`einsum` 中重复且未出现在输出侧的下标表示求和。

## 8. 拷贝、共享与 NumPy 转换

```python
a = np.array([1., 2., 3.], dtype=np.float32)
shared = torch.from_numpy(a)  # CPU 上共享数据
copied = torch.tensor(a)      # 复制数据
a[0] = 9
assert shared[0].item() == 9 and copied[0].item() == 1

t = torch.tensor([1., 2.], requires_grad=True)
detached = t.detach()         # 不再跟踪原计算图，但仍共享存储
cloned = t.clone()            # 独立存储，仍保留梯度关系
independent = t.detach().clone() # 独立存储，且与原图断开
array = t.detach().cpu().numpy() # CPU 且不需要梯度时才能直接转 NumPy
array_copy = array.copy()        # 需要独立数据时再复制
```

`torch.as_tensor` 在类型与设备兼容时可共享已有数据；类型或设备转换可能复制。`.to()` 无需改变类型/设备时可能直接返回原张量。

列表切片创建新列表，NumPy/Torch 基本切片通常共享底层存储；`reshape` 是否复制取决于布局。NumPy 负 stride 视图传给 `torch.from_numpy` 前可先 `.copy()`。参见 [from_numpy](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html) 与 [detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)。

## 9. 梯度与标量

```python
x = torch.tensor([1., 2.], requires_grad=True) # 浮点叶子张量，开启梯度跟踪
loss = (x * x).sum()                         # 标量损失
loss.backward()
assert torch.allclose(x.grad, torch.tensor([2., 4.]))
(x * x).sum().backward()                     # 梯度默认累加，不自动清零
assert torch.allclose(x.grad, torch.tensor([4., 8.]))
x.grad = None                               # 下一次 backward 重新建立梯度
with torch.no_grad():
    prediction = x * 2                       # 不记录该段计算的梯度
scalar = loss.item()                         # Python 数值，脱离计算图
```

非标量输出执行 backward 时需要显式给出上游梯度。非叶子张量的 grad 默认不保留，需要时调用 `retain_grad()`。`model.eval()` 切换 Dropout/BatchNorm 等模块行为，不负责关闭梯度记录；`no_grad()` 也不负责切换模型模式。

`.item()` 要求张量只有一个元素；训练损失的后续可微计算应继续使用 Tensor。CUDA 上频繁提取标量还可能引入同步。参见 [Autograd 说明](https://docs.pytorch.org/docs/stable/notes/autograd.html)。

## 10. LLM 常用接口组合

```python
ids = torch.tensor([[1, 2, 0], [3, 4, 1]], dtype=torch.long) # [B,S]
weight = torch.arange(20, dtype=torch.float32).reshape(5, 4) # [V,D]
embedded = F.embedding(ids, weight)                       # [B,S,D]=[2,3,4]
np_embedded = weight.numpy()[ids.numpy()]                  # NumPy 可直接用 token ID 索引词表
assert np.array_equal(embedded.numpy(), np_embedded)
one_hot = F.one_hot(ids, num_classes=5).float()            # [B,S,V]，原输出是整数类型

keep = torch.ones(3, 3, dtype=torch.bool).tril()           # True 表示当前 Query 可见的 key
scores = torch.zeros(2, 2, 3, 3)                          # [B,H,S,S]
masked_probs = torch.softmax(scores.masked_fill(~keep, -torch.inf), dim=-1)

logits = torch.zeros(2, 3, 5)                             # [B,S,V]，交叉熵直接接收 logits
labels = ids.clone()
labels[ids == 0] = -100                                   # 本例将 padding 的目标位置忽略
loss = F.cross_entropy(logits[:, :-1].reshape(-1, 5),
                       labels[:, 1:].reshape(-1), ignore_index=-100) # 下一 token 对齐
padded = F.pad(torch.tensor([1, 2]), (0, 2), value=0)      # [1,2,0,0]，末维左补 0 个、右补 2 个
```

`F.embedding` 的权重为 `[V,D]`，输入是整数 token ID；它与线性层输入乘权重的接口不同。`F.cross_entropy` 的类别下标必须是整数，忽略位置不参与均值；应保证至少有一个有效目标。

`masked_fill` 的 True 表示要替换的位置，因此使用 `~keep`。全 -inf 的一行不能直接套普通 Softmax；自定义全 mask 语义见 [LLM 核心算子手撕](../llm-operator-implementations/index.md)。

## 参考资料

- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy 索引](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy 广播](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [PyTorch Tensor API](https://docs.pytorch.org/docs/stable/tensors.html)
- [PyTorch functional](https://docs.pytorch.org/docs/stable/nn.functional.html)
