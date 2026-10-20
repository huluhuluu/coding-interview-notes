---
title: "LLM 核心算子手撕：PyTorch / Python / C++"
date: 2026-10-20T15:20:00+08:00
draft: false
description: "面向笔试整理 Linear、Softmax、LayerNorm、RMSNorm、SwiGLU、RoPE、MHA/GQA、KV Cache 与交叉熵的三种实现"
slug: "acm-llm-operators"
tags: ["cpp", "python", "pytorch", "llm", "算子", "手撕"]
categories: ["笔试重点"]
comments: true
math: true
---

# LLM 核心算子手撕：PyTorch / Python / C++

这篇按笔试中常见的拆解顺序整理：矩阵乘法、归约与归一化、激活、位置编码，再组合成 Attention。每个算子给出公式、形状和三种实现；PyTorch 使用张量基础操作，Python 使用列表和标准库，C++ 使用循环和标准容器。

这里实现前向计算，并补充 Softmax 的反向公式。CPU 循环用于检查逻辑和回答手撕题；GPU 分块、流水线和 FlashAttention 见[LLM 算子优化系列](/p/llm-kernel-notes/)。

## 1. 统一约定

| 符号 | 含义 |
|------|------|
| B / L | batch size / sequence length |
| D / F | hidden size / FFN intermediate size |
| Hq / Hkv | Query / KV head 数 |
| d / dv | QK / V 的单 head 维度 |
| Lq / Lkv | Query / KV 长度 |

矩阵非空且各行等长，维度满足公式要求。输入为有限实数，mask 位置允许用负无穷。PyTorch 接口支持文中说明的 batch 维；Python/C++ 为便于手写，默认按一行、一个 head 或一个请求处理，batch 在最外层循环即可。

按顺序使用下方片段。Python 公共导入：

```python
import math
```

PyTorch 公共导入与累加类型：

```python
import torch

def acc_torch(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
```

归约通常将 FP16/BF16 提升到 FP32，最后转换回输入类型。FP32/FP64 保留原类型。C++17 公共头文件和类型：

```cpp
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;
using Heads = vector<Mat>; // [H,L,d]，不含 batch
```

## 2. MatMul 与 Linear

矩阵乘法 `A[M,K] × B[K,N] -> C[M,N]`：

$$
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj}.
$$

Linear 按 PyTorch 的权重约定存储 `W[N,K]`，所以 `Y=XWᵀ+b`。不要把权重的存储形状误写成实际右操作数的形状。

**PyTorch**，X 可为 `[...,K]`，bias 为 `[N]`：

```python
def linear_torch(x, weight, bias=None):
    y = x @ weight.transpose(-1, -2)
    return y if bias is None else y + bias
```

**Python**，输入 X 为 `[M,K]`，weight 为 `[N,K]`：

```python
def linear_py(x, weight, bias=None):
    return [
        [sum(a * b for a, b in zip(row, w))
         + (0.0 if bias is None else bias[j])
         for j, w in enumerate(weight)]
        for row in x
    ]
```

**C++**：

```cpp
Mat linear_cpp(const Mat& x, const Mat& weight, const Vec& bias = {}) {
    int m = int(x.size()), n = int(weight.size()), k = int(x[0].size());
    Mat y(m, Vec(n, 0.0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            for (int t = 0; t < k; ++t) y[i][j] += x[i][t] * weight[j][t];
            if (!bias.empty()) y[i][j] += bias[j];
        }
    return y;
}
```

时间 O(MNK)，输出空间 O(MN)。这些循环没有额外验证形状；尤其 Python 的 `zip` 会在较短的一侧结束，手写测试时应主动检查维度。

## 3. 稳定 Softmax 与反向

沿一行计算：

$$
p_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\qquad m=\max_j x_j.
$$

减去最大值不改变结果，可避免直接 `exp(1000)` 溢出。本文额外约定：一整行都被 mask 时返回全 0。这是一种显式边界语义；各框架或后端的处理可能不同。

**PyTorch**，沿最后一维归约：

```python
def softmax_torch(x):
    work = acc_torch(x)
    maximum = work.amax(dim=-1, keepdim=True)
    maximum = torch.where(torch.isneginf(maximum), 0.0, maximum)
    e = (work - maximum).exp()
    denom = e.sum(dim=-1, keepdim=True)
    denom = torch.where(denom == 0, 1.0, denom)
    return (e / denom).to(x.dtype)
```

**Python**：

```python
def softmax_py(row):
    maximum = max(row)
    if maximum == -math.inf:
        return [0.0] * len(row)
    e = [math.exp(x - maximum) for x in row]
    total = sum(e)
    return [x / total for x in e]
```

**C++**：

```cpp
Vec softmax_cpp(const Vec& row) {
    double maximum = *max_element(row.begin(), row.end());
    Vec p(row.size(), 0.0);
    if (maximum == -numeric_limits<double>::infinity()) return p;
    double total = 0.0;
    for (int i = 0; i < int(row.size()); ++i) {
        p[i] = exp(row[i] - maximum);
        total += p[i];
    }
    for (double& x : p) x /= total;
    return p;
}
```

单行时间 O(D)，空间 O(D)。假设上游梯度为 g，利用 Softmax 的 Jacobian 可化简为：

$$
\frac{\partial L}{\partial x_i}
=p_i\left(g_i-\sum_jg_jp_j\right).
$$

不需要显式构造 D×D 的 Jacobian：

```python
def softmax_backward_torch(p, grad):
    pw, gw = acc_torch(p), acc_torch(grad)
    return (pw * (gw - (pw * gw).sum(dim=-1, keepdim=True))).to(p.dtype)
```

```python
def softmax_backward_py(p, grad):
    dot = sum(a * b for a, b in zip(p, grad))
    return [a * (b - dot) for a, b in zip(p, grad)]
```

```cpp
Vec softmax_backward_cpp(const Vec& p, const Vec& grad) {
    double dot = 0.0;
    for (int i = 0; i < int(p.size()); ++i) dot += p[i] * grad[i];
    Vec dx(p.size());
    for (int i = 0; i < int(p.size()); ++i) dx[i] = p[i] * (grad[i] - dot);
    return dx;
}
```

## 4. LayerNorm 与 RMSNorm

按最后一维 D 归一化。LayerNorm 减均值、除标准差；RMSNorm 不减均值，使用均方根。

$$
\mu=\frac1D\sum_i x_i,\quad
\sigma^2=\frac1D\sum_i(x_i-\mu)^2,\quad
\operatorname{LN}(x)_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma_i+\beta_i.
$$
$$
\operatorname{RMSNorm}(x)_i=
\frac{x_i}{\sqrt{\frac1D\sum_jx_j^2+\epsilon}}\gamma_i.
$$

LayerNorm 的方差分母是 D，不是 D-1；PyTorch 对应 `unbiased=False` 的定义。这里统一显式传 `eps=1e-5`，对照框架时保持相同值。参见 [LayerNorm 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)。

**PyTorch**：

```python
def layernorm_torch(x, gamma, beta, eps=1e-5):
    work = acc_torch(x)
    centered = work - work.mean(dim=-1, keepdim=True)
    var = centered.square().mean(dim=-1, keepdim=True)
    y = centered * torch.rsqrt(var + eps)
    return (y * gamma + beta).to(x.dtype)

def rmsnorm_torch(x, gamma, eps=1e-5):
    work = acc_torch(x)
    y = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)
    return (y * gamma).to(x.dtype)
```

**Python**，对一行计算：

```python
def layernorm_py(row, gamma, beta, eps=1e-5):
    mean = sum(row) / len(row)
    var = sum((x - mean) ** 2 for x in row) / len(row)
    inv = 1.0 / math.sqrt(var + eps)
    return [(x - mean) * inv * g + b for x, g, b in zip(row, gamma, beta)]

def rmsnorm_py(row, gamma, eps=1e-5):
    inv = 1.0 / math.sqrt(sum(x * x for x in row) / len(row) + eps)
    return [x * inv * g for x, g in zip(row, gamma)]
```

**C++**：

```cpp
Vec layernorm_cpp(const Vec& row, const Vec& gamma, const Vec& beta,
                  double eps = 1e-5) {
    double mean = 0.0, var = 0.0;
    for (double x : row) mean += x;
    mean /= row.size();
    for (double x : row) var += (x - mean) * (x - mean);
    double inv = 1.0 / sqrt(var / row.size() + eps);
    Vec y(row.size());
    for (int i = 0; i < int(row.size()); ++i)
        y[i] = (row[i] - mean) * inv * gamma[i] + beta[i];
    return y;
}

Vec rmsnorm_cpp(const Vec& row, const Vec& gamma, double eps = 1e-5) {
    double square_sum = 0.0;
    for (double x : row) square_sum += x * x;
    double inv = 1.0 / sqrt(square_sum / row.size() + eps);
    Vec y(row.size());
    for (int i = 0; i < int(row.size()); ++i) y[i] = row[i] * inv * gamma[i];
    return y;
}
```

每行时间 O(D)，输出空间 O(D)。常量行的 LayerNorm 输出为 beta，RMSNorm 一般不会变成 0。`eps` 放在根号内；不要误把 RMSNorm 写成 LayerNorm 去掉 beta。

## 5. SiLU 与 SwiGLU

$$
\operatorname{SiLU}(x)=x\sigma(x),\qquad
\operatorname{SwiGLU}(x)=\operatorname{SiLU}(xW_g^T)\odot(xW_u^T).
$$

门控后的张量为 `[...,F]`，再乘 `Wdown[D,F]` 回到 `[...,D]`。本文分别传入 gate/up，不依赖某个库的打包顺序。[GLU Variants](https://arxiv.org/abs/2002.05202) 讨论了这种门控激活结构。

**PyTorch**：

```python
def swiglu_torch(gate, up):
    work = acc_torch(gate)
    return (work * torch.sigmoid(work) * acc_torch(up)).to(gate.dtype)

def ffn_torch(x, w_gate, w_up, w_down):
    hidden = swiglu_torch(linear_torch(x, w_gate), linear_torch(x, w_up))
    return linear_torch(hidden, w_down)
```

**Python**，稳定 sigmoid 根据符号分支，避免很大的负数导致 `exp(-x)` 溢出：

```python
def sigmoid_py(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)

def swiglu_py(gate, up):
    return [g * sigmoid_py(g) * u for g, u in zip(gate, up)]

def ffn_py(x, w_gate, w_up, w_down):
    gate, up = linear_py(x, w_gate), linear_py(x, w_up)
    hidden = [swiglu_py(g, u) for g, u in zip(gate, up)]
    return linear_py(hidden, w_down)
```

**C++**：

```cpp
double sigmoid_cpp(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + exp(-x));
    double e = exp(x);
    return e / (1.0 + e);
}

Vec swiglu_cpp(const Vec& gate, const Vec& up) {
    Vec y(gate.size());
    for (int i = 0; i < int(gate.size()); ++i)
        y[i] = gate[i] * sigmoid_cpp(gate[i]) * up[i];
    return y;
}

Mat ffn_cpp(const Mat& x, const Mat& wg, const Mat& wu, const Mat& wd) {
    Mat gate = linear_cpp(x, wg), up = linear_cpp(x, wu);
    for (int i = 0; i < int(gate.size()); ++i) gate[i] = swiglu_cpp(gate[i], up[i]);
    return linear_cpp(gate, wd);
}
```

逐元素门控为 O(LF)，三个线性映射合计为 O(LDF)。`SiLU(gate*up)` 与 `SiLU(gate)*up` 不相同。

## 6. RoPE：先确认通道配对方式

本文使用相邻通道配对，d 为偶数，对第 i 对通道：

$$
\theta_{p,i}=p\cdot\operatorname{base}^{-2i/d},\quad
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}
=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}.
$$

[RoFormer](https://arxiv.org/abs/2104.09864) 使用旋转编码位置。实际模型还可能以前后半区配对，或只旋转部分维度；配对和权重布局必须匹配。

**PyTorch**，输入 `[...,L,d]`，positions 为 `[L]`，可以直接传 decode 的绝对位置：

```python
def rope_torch(x, positions, base=10000.0):
    work = acc_torch(x)
    d = work.shape[-1]
    freq = base ** (-torch.arange(0, d, 2, device=x.device, dtype=work.dtype) / d)
    angles = positions.to(device=x.device, dtype=work.dtype)[:, None] * freq[None, :]
    c, s = angles.cos(), angles.sin()
    even, odd = work[..., 0::2], work[..., 1::2]
    out = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1)
    return out.flatten(-2).to(x.dtype)
```

**Python**，处理一个 token 的一个 head：

```python
def rope_py(row, position, base=10000.0):
    d = len(row)
    out = [0.0] * d
    for i in range(0, d, 2):
        angle = position * base ** (-i / d)
        c, s = math.cos(angle), math.sin(angle)
        out[i] = row[i] * c - row[i + 1] * s
        out[i + 1] = row[i] * s + row[i + 1] * c
    return out
```

**C++**：

```cpp
Vec rope_cpp(const Vec& row, int position, double base = 10000.0) {
    int d = int(row.size());
    Vec out(d);
    for (int i = 0; i < d; i += 2) {
        double angle = position * pow(base, -double(i) / d);
        double c = cos(angle), s = sin(angle);
        out[i] = row[i] * c - row[i + 1] * s;
        out[i + 1] = row[i] * s + row[i + 1] * c;
    }
    return out;
}
```

每个 head/token 时间 O(d)。RoPE 通常作用于 Q、K，V 不旋转。位置为 0 时结果不变，旋转前后平方和不变；这两个性质适合手算检查。实现不能原地覆盖 even 后再用修改后的 even 计算 odd。

## 7. Scaled Dot-Product Attention

单 head 输入 `Q[Lq,d]`、`K[Lkv,d]`、`V[Lkv,dv]`：

$$
S=QK^T/\sqrt d,\qquad P=\operatorname{softmax}(S+M),\qquad O=PV.
$$

先除以 `sqrt(d)`，再 Softmax；归约方向是 key 长度 Lkv。本文 mask 名为 `keep`，**True 表示可以参与 Attention**。每行全 False 时，沿用前文 Softmax 返回全 0 的约定。

**PyTorch**，支持 `[B,H,L,d]`，keep 需能广播到 `[B,H,Lq,Lkv]`：

```python
def attention_torch(q, k, v, keep=None):
    scores = acc_torch(q) @ acc_torch(k).transpose(-1, -2)
    scores = scores / math.sqrt(q.shape[-1])
    if keep is not None:
        scores = scores.masked_fill(~keep, -torch.inf)
    p = softmax_torch(scores)
    return (p @ acc_torch(v)).to(q.dtype)
```

**Python**，先算一行 scores，再得到一行输出：

```python
def attention_py(q, k, v, keep=None):
    d, dv = len(q[0]), len(v[0])
    output = []
    for i, query in enumerate(q):
        scores = []
        for j, key in enumerate(k):
            allowed = keep is None or keep[i][j]
            score = sum(a * b for a, b in zip(query, key)) / math.sqrt(d)
            scores.append(score if allowed else -math.inf)
        p = softmax_py(scores)
        output.append([sum(p[j] * v[j][c] for j in range(len(v))) for c in range(dv)])
    return output
```

**C++**，mask 采用 `vector<vector<bool>>`：

```cpp
Mat attention_cpp(const Mat& q, const Mat& k, const Mat& v,
                  const vector<vector<bool>>& keep = {}) {
    int lq = int(q.size()), lk = int(k.size());
    int d = int(q[0].size()), dv = int(v[0].size());
    Mat output(lq, Vec(dv, 0.0));
    for (int i = 0; i < lq; ++i) {
        Vec scores(lk, 0.0);
        for (int j = 0; j < lk; ++j) {
            for (int t = 0; t < d; ++t) scores[j] += q[i][t] * k[j][t];
            scores[j] /= sqrt(double(d));
            if (!keep.empty() && !keep[i][j])
                scores[j] = -numeric_limits<double>::infinity();
        }
        Vec p = softmax_cpp(scores);
        for (int j = 0; j < lk; ++j)
            for (int c = 0; c < dv; ++c) output[i][c] += p[j] * v[j][c];
    }
    return output;
}
```

时间 O(Lq·Lkv·(d+dv))。PyTorch 这里显式保存 scores/P，中间空间 O(Lq·Lkv) 每 head；Python/C++ 按 query 行计算，额外工作空间 O(Lkv)，另有输出 O(Lq·dv)。不保存全矩阵不代表已经实现 FlashAttention 的分块复用。

### 7.1 Causal mask 与 KV Cache 对齐

初始 prefill 中 `Lq=Lkv=L`，允许 `key_pos<=query_pos`，即主对角线及其下方。若 KV 包含过去缓存和本次追加的 Query，对齐到序列末尾：

$$
\operatorname{keep}[i,j]=(j\le L_{kv}-L_q+i),\qquad L_{kv}\ge L_q.
$$

三种构造方式：

```python
def causal_keep_torch(lq, lkv, device):
    qpos = torch.arange(lq, device=device) + lkv - lq
    kpos = torch.arange(lkv, device=device)
    return kpos[None, :] <= qpos[:, None]
```

```python
def causal_keep_py(lq, lkv):
    return [[j <= lkv - lq + i for j in range(lkv)] for i in range(lq)]
```

```cpp
vector<vector<bool>> causal_keep_cpp(int lq, int lkv) {
    vector<vector<bool>> keep(lq, vector<bool>(lkv));
    for (int i = 0; i < lq; ++i)
        for (int j = 0; j < lkv; ++j) keep[i][j] = j <= lkv - lq + i;
    return keep;
}
```

例如 `Lq=1,Lkv=4` 应得到 `[True,True,True,True]`，新 Query 能看到已提供的全部历史和当前 token。

对照 PyTorch `F.scaled_dot_product_attention` 时，布尔 mask 也是 True 表示保留，但 `nn.MultiheadAttention` 的布尔 mask 使用 True 表示屏蔽。SDPA 的非方阵 `is_causal=True` 使用左上对齐，和本文的追加场景不同，应传显式 `attn_mask`，并设置 `is_causal=False,dropout_p=0.0`。参见 [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 与 [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)。

### 7.2 MHA、MQA 与 GQA

| 形式 | head 数关系 |
|------|-------------|
| MHA | Hq=Hkv |
| MQA | Hkv=1 |
| GQA | Hq 是 Hkv 的整数倍，多个 Query heads 共享一个 KV head |

映射为 `kv_head=query_head//(Hq/Hkv)`，不要用 `query_head%Hkv`，二者对应不同的分组顺序。

**PyTorch**，Q 为 `[B,Hq,Lq,d]`，K/V 为 `[B,Hkv,Lkv,d/dv]`：

```python
def gqa_torch(q, k, v, keep=None):
    group = q.shape[1] // k.shape[1]
    kr = k.repeat_interleave(group, dim=1)
    vr = v.repeat_interleave(group, dim=1)
    return attention_torch(q, kr, vr, keep)
```

**Python**，这里输入不含 batch，为 `[H,L,d]`：

```python
def gqa_py(q, k, v, keep=None):
    group = len(q) // len(k)
    return [attention_py(query, k[h // group], v[h // group], keep)
            for h, query in enumerate(q)]
```

**C++**：

```cpp
Heads gqa_cpp(const Heads& q, const Heads& k, const Heads& v,
              const vector<vector<bool>>& keep = {}) {
    int group = int(q.size() / k.size());
    Heads output;
    for (int h = 0; h < int(q.size()); ++h)
        output.push_back(attention_cpp(q[h], k[h / group], v[h / group], keep));
    return output;
}
```

要求 `Hq>=Hkv>0` 且 `Hq%Hkv==0`。PyTorch 复制 KV 是为了展示映射，实际高效 kernel 可以复用 KV 而不物理展开。对照输出时可先显式展开，再调用 SDPA，避免依赖某个后端的 GQA 支持范围。

### 7.3 完整 MHA：投影、拆头、合头

以下使用无 bias 的标准 MHA，输入 `X[B,L,D]`，四个权重都是 `[D,D]`，要求 D 能被 heads 整除。

**PyTorch**：

```python
def mha_torch(x, wq, wk, wv, wo, heads, keep=None):
    b, length, hidden = x.shape
    d = hidden // heads
    def split(weight):
        return linear_torch(x, weight).reshape(b, length, heads, d).transpose(1, 2)
    output = attention_torch(split(wq), split(wk), split(wv), keep)
    merged = output.transpose(1, 2).contiguous().reshape(b, length, hidden)
    return linear_torch(merged, wo)
```

**Python**，一个请求 `[L,D]`：

```python
def mha_py(x, wq, wk, wv, wo, heads, keep=None):
    d = len(x[0]) // heads
    def split(weight):
        projected = linear_py(x, weight)
        return [[row[h * d:(h + 1) * d] for row in projected] for h in range(heads)]
    output = gqa_py(split(wq), split(wk), split(wv), keep)
    merged = [[value for head in output for value in head[i]] for i in range(len(x))]
    return linear_py(merged, wo)
```

**C++**：

```cpp
Mat mha_cpp(const Mat& x, const Mat& wq, const Mat& wk, const Mat& wv,
            const Mat& wo, int heads, const vector<vector<bool>>& keep = {}) {
    int length = int(x.size()), hidden = int(x[0].size()), d = hidden / heads;
    auto split = [&](const Mat& weight) {
        Mat projected = linear_cpp(x, weight);
        Heads result(heads, Mat(length, Vec(d)));
        for (int h = 0; h < heads; ++h)
            for (int i = 0; i < length; ++i)
                for (int c = 0; c < d; ++c) result[h][i][c] = projected[i][h * d + c];
        return result;
    };
    Heads output = gqa_cpp(split(wq), split(wk), split(wv), keep);
    Mat merged(length, Vec(hidden));
    for (int i = 0; i < length; ++i)
        for (int h = 0; h < heads; ++h)
            for (int c = 0; c < d; ++c) merged[i][h * d + c] = output[h][i][c];
    return linear_cpp(merged, wo);
}
```

时间 O(LD²+L²D)。拆头的关键是 `[B,L,H,d] -> [B,H,L,d]`，直接 reshape 到 `[B,H,L,d]` 会改变逻辑元素的对应关系。转置后若用 `view`，还需考虑内存连续性。

## 8. KV Cache：缓存 K/V，不缓存旧 Q

每层缓存已经完成该层投影及所需位置编码的 K 和 V。新 token 只产生自己的 Q/K/V，将新 K/V 追加到历史后，当前 Q 对完整缓存做 Attention。

**PyTorch**，cache 和 new 为 `[B,Hkv,L,d]`，这里 K/V 的最后一维均设为 d：

```python
def append_kv_torch(cache_k, cache_v, new_k, new_v):
    return torch.cat((cache_k, new_k), dim=-2), torch.cat((cache_v, new_v), dim=-2)
```

**Python**，一个 head，沿序列维追加：

```python
def append_kv_py(cache_k, cache_v, new_k, new_v):
    return cache_k + new_k, cache_v + new_v
```

**C++**，一个 head：

```cpp
pair<Mat, Mat> append_kv_cpp(const Mat& cache_k, const Mat& cache_v,
                           const Mat& new_k, const Mat& new_v) {
    Mat k = cache_k, v = cache_v;
    k.insert(k.end(), new_k.begin(), new_k.end());
    v.insert(v.end(), new_v.begin(), new_v.end());
    return {k, v};
}
```

这些是动态拼接的参考写法。Torch/C++ 需要复制已有数据；Python 复制外层列表，行对象仍共享引用，使用后不要原地修改旧行。真实 decode 常预分配或分页管理缓存，逐步拼接会引入反复复制。

对已使用 RoPE 的 K，缓存时保留旋转后的结果，不要每一步重新旋转历史 K。新 token 的 position 应继续历史长度；已有 3 个 token 时，下一 token 的位置是 3。

长度为 L 的历史，KV 存储量为每层每个请求 `2*Hkv*L*d` 个元素。KV Cache 省去历史 K/V 投影的重复计算，当前 Query 仍需要访问历史 KV。

## 9. 稳定交叉熵与 Next-token 对齐

给定一行 logits 和目标类别 t：

$$
L=-\log p_t
=\log\sum_j e^{x_j-m}-(x_t-m),\qquad m=\max_jx_j.
$$

直接使用 logits 计算，避免先 Softmax 后对极小概率取 log。下面统一输入 `[N,C]`，目标为 `[N]`，按 N 求均值，不处理 ignore index、类别权重或 label smoothing。

**PyTorch**：

```python
def cross_entropy_torch(logits, targets):
    work = acc_torch(logits)
    shifted = work - work.amax(dim=-1, keepdim=True)
    log_denom = shifted.exp().sum(dim=-1).log()
    selected = shifted.gather(1, targets[:, None]).squeeze(1)
    return (log_denom - selected).mean()
```

**Python**：

```python
def cross_entropy_py(logits, targets):
    losses = []
    for row, target in zip(logits, targets):
        maximum = max(row)
        log_denom = math.log(sum(math.exp(x - maximum) for x in row))
        losses.append(log_denom - (row[target] - maximum))
    return sum(losses) / len(losses)
```

**C++**：

```cpp
double cross_entropy_cpp(const Mat& logits, const vector<int>& targets) {
    double loss = 0.0;
    for (int i = 0; i < int(logits.size()); ++i) {
        const Vec& row = logits[i];
        double maximum = *max_element(row.begin(), row.end()), total = 0.0;
        for (double x : row) total += exp(x - maximum);
        loss += log(total) - (row[targets[i]] - maximum);
    }
    return loss / logits.size();
}
```

时间 O(NC)。这里 targets 必须为合法类别下标；Torch 需要整数索引类型。本文训练损失返回累加精度，不再转回半精度。

Causal LM 中，位置 i 的 logits 预测下一个 token。若输入标签尚未由模型内部 shift，应对 `logits[:,:-1,:]` 和 `token_ids[:,1:]` 计算损失，而不是与同位置 token 对齐。带 padding 时只对有效 target 求均值。框架参考见 [cross_entropy 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)。

## 10. 手写后的检查顺序

| 算子 | 最小检查 |
|------|----------|
| Linear | 非方阵输入、bias 广播、权重是否转置 |
| Softmax | 每行和为 1；`[1000,1001]` 不溢出；全 mask 为 0 |
| Softmax backward | 有限差分或 autograd；梯度元素之和接近 0 |
| LayerNorm / RMSNorm | 常量行、零行、D=1、相同 eps |
| SwiGLU | gate/up 顺序；大幅负输入的 sigmoid |
| RoPE | position=0 不变；范数守恒；缓存位置连续 |
| Attention | 非方阵 Lq/Lkv；全 mask 行；V 维度 dv 与 d 不同 |
| GQA / MHA | Hq/Hkv 分组映射；拆头与合头元素顺序 |
| KV Cache | 完整 causal prefill 与分段追加结果一致 |
| 交叉熵 | 大 logits；类别下标；预测目标偏移一位 |

三种实现先使用小尺寸、FP64 对照数学结果，再检查 FP32 和半精度容差。不要要求不同累加顺序逐位相等。对照 Torch 内置接口时，还需要对齐 dtype、归约维度、mask、eps 和 reduction，而不仅是函数名字。
