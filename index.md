---
title: "笔试重点"
date: 2026-10-20T12:00:00+08:00
lastmod: 2026-10-20T16:00:00+08:00
draft: false
description: "C++ / Python 笔试备忘：ACM 输入处理、常用接口、算法模板、数论技巧与 LLM 算子实现"
slug: "coding-interview-notes"
tags: ["cpp", "python", "llm"]
categories: ["笔试重点"]
build:
  list: never
comments: true
math: true
---

# 笔试重点

C++、Python 笔试备忘，记录输入处理、常用接口、算法模板、数论技巧与 LLM 核心算子实现。

## 目录

### 输入与接口

| 文章 | 说明 | 状态 |
|------|------|------|
| [ACM 输入处理：C++ / Python](blog/acm-input-cpp-python/index.md) | 单组、多组、EOF、整行、矩阵与字符串输入 | ✅ 完成 |
| [C++ / Python 常用接口](blog/cpp-python-apis/index.md) | 容器、排序、二分、堆、字符串与计数 | ✅ 完成 |

### 算法与技巧

| 文章 | 说明 | 状态 |
|------|------|------|
| [笔试常见算法](blog/interview-algorithms/index.md) | 前缀和、双指针、二分、搜索、图论与动态规划 | ✅ 完成 |
| [笔试常见技巧](blog/interview-tricks/index.md) | lowbit、树状数组、快速幂、筛法、逆元与组合数 | ✅ 完成 |

### LLM 算子

| 文章 | 说明 | 状态 |
|------|------|------|
| [LLM 核心算子手撕](blog/llm-operator-implementations/index.md) | PyTorch / Python / C++ 实现 Linear、Softmax、归一化、RoPE、MHA/GQA 与交叉熵 | ✅ 完成 |

## 阅读顺序

1. 输入处理与常用接口：确认输入边界、容器操作和接口复杂度。
2. 算法与技巧：结合数据范围选择模板，检查使用前提。
3. LLM 算子：按矩阵乘、归一化、位置编码、Attention 的顺序实现与验证。

代码基线为 C++17、Python 3；LLM 算子另提供 PyTorch 实现。

## 相关笔记

- [语言与文件格式备忘](https://github.com/huluhuluu/language-file-cheatsheets/blob/main/blog/_index.md)
- [LLM 算子优化](https://github.com/huluhuluu/llm-kernel-notes/blob/main/index.md)
