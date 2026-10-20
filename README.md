# 笔试重点

C++、Python 笔试备忘，记录输入处理、常用接口、算法与数论、LLM 算子和模型结构。

## 目录

### 输入与接口

| 文章 | 说明 | 状态 |
|------|------|------|
| [ACM 输入处理：C++ / Python](blog/acm-input-cpp-python/index.md) | 单组、多组、EOF、整行、矩阵与字符串输入 | ✅ 完成 |
| [C++ 常用接口](blog/cpp-apis/index.md) | vector、排序、二分、string.substr、字符串修改与标准算法 | ✅ 完成 |
| [Python 常用接口](blog/python-apis/index.md) | 列表、切片、字符串、字典、heapq、deque 与遍历 | ✅ 完成 |
| [NumPy / PyTorch 常用操作](blog/numpy-torch-apis/index.md) | 索引、广播、维度变换、矩阵乘、数据转换与梯度 | ✅ 完成 |

### 算法与技巧

| 文章 | 说明 | 状态 |
|------|------|------|
| [笔试常见算法](blog/interview-algorithms/index.md) | 前缀和、双指针、二分、搜索、图论与动态规划 | ✅ 完成 |
| [笔试常见技巧](blog/interview-tricks/index.md) | lowbit、树状数组、快速幂、筛法、逆元与组合数 | ✅ 完成 |

### LLM 算子与模型

| 文章 | 说明 | 状态 |
|------|------|------|
| [LLM 核心算子手撕](blog/llm-operator-implementations/index.md) | PyTorch / Python / C++ 实现 Linear、Softmax、归一化、RoPE、MHA/GQA 与交叉熵 | ✅ 完成 |
| [DeepSeek V4 Flash 模型结构与维度](blog/deepseek-v4-flash-architecture/index.md) | 对照官方实现梳理 mHC、压缩 Attention、MoE、MTP 与输入输出维度 | ✅ 完成 |

## 阅读顺序

1. 输入处理与常用接口：确认输入边界、容器操作和接口复杂度。
2. 算法与技巧：结合数据范围选择模板，检查使用前提。
3. LLM 算子与模型：先实现核心算子，再对照 DeepSeek V4 Flash 串起计算路径与维度。

代码基线为 C++17、Python 3；LLM 算子另提供 PyTorch 实现。

## 相关笔记

- [语言与文件格式备忘](https://github.com/huluhuluu/language-file-cheatsheets/blob/main/blog/_index.md)
- [LLM 算子优化](https://github.com/huluhuluu/llm-kernel-notes/blob/main/index.md)
