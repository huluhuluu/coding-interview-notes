---
title: "笔试常见算法"
date: 2026-10-20T12:20:00+08:00
draft: false
description: "从数据范围与性质选择算法，整理前缀和、双指针、二分、搜索、图论和动态规划模板"
slug: "acm-common-algorithms"
tags: ["cpp", "python", "acm", "算法"]
categories: ["笔试重点"]
comments: true
math: true
---

# 笔试常见算法

先看数据范围，再找问题性质。复杂度估计只能排除明显不可行的写法，实际耗时还取决于语言、常数、时限和内存访问。

本文片段是独立模板：C++ 使用 C++17，需要包含所用标准库头文件并使用 `std::` 或 `using namespace std;`；Python 使用 Python 3。

## 1. 根据规模初步筛选

| 规模或性质 | 常见候选 |
|------------|----------|
| n 很小，约二十个元素 | 子集枚举、回溯、状态压缩 |
| n 为几千 | O(n²) DP、枚举两端点 |
| n 为十万量级 | 排序、堆、双指针、O(n log n) 数据结构 |
| 连续区间的和 | 前缀和；满足单调条件时可用滑动窗口 |
| 答案可行性单调 | 二分答案 |
| 无权最短路 | BFS |
| 非负权最短路 | Dijkstra |
| 动态连通关系 | 并查集 |

`2^n` 和 `n!` 的增长差别很大，不要统一理解为“小数据能暴力”。Python 还要考虑整数、列表、字典对象的额外内存。

## 2. 前缀和、差分与哈希计数

采用半开区间 `[l,r)`：`prefix[0]=0`，`prefix[i+1]=prefix[i]+a[i]`，于是区间和为 `prefix[r]-prefix[l]`。

```python
def prefix_sums(a):
    prefix = [0]
    for x in a:
        prefix.append(prefix[-1] + x)
    return prefix
```

构建 O(n)，每次查询 O(1)。若多次给区间 `[l,r)` 加 `v`，差分数组执行 `diff[l]+=v; diff[r]-=v`，最后求前缀和还原。差分数组多开一格以容纳 `r=n`。

求“和恰好为 k 的子数组个数”时，负数会破坏普通窗口的单调性，但前缀和加哈希表仍成立：当前前缀为 `s`，需要统计此前出现过多少次 `s-k`。

```python
def count_subarrays(a, k):
    freq = {0: 1}
    s = answer = 0
    for x in a:
        s += x
        answer += freq.get(s - k, 0)
        freq[s] = freq.get(s, 0) + 1
    return answer
```

先查询再加入当前前缀，避免 `k=0` 时把空子数组算进去。平均时间 O(n)，空间 O(n)；C++ 的前缀和与答案通常都需要 `long long`。

## 3. 双指针与滑动窗口

对非负数组求和不超过 `limit` 的最长连续区间，要求 `limit >= 0`。右端点扩张不会减小区间和，左端点移动不会增大区间和，因此每个指针只前进 O(n) 次。

```cpp
int longest_window(const vector<int>& a, long long limit) {
    int left = 0, answer = 0;
    long long sum = 0;
    for (int right = 0; right < int(a.size()); ++right) {
        sum += a[right];
        while (sum > limit) sum -= a[left++];
        answer = max(answer, right - left + 1);
    }
    return answer;
}
```

数组包含负数时，这个模板不成立。比如 `[5,-4]`，提前删掉 `5` 会错过后面和为 `1` 的长度为 2 的区间。

排序数组的两数和也能使用首尾指针：和偏小就增大左端点，偏大就减小右端点。若题目要求原下标，排序前保存下标。

## 4. 二分答案

把判定函数组织为 `False ... False True ... True`，寻找第一个可行值。`[lo,hi)` 中 `hi` 可以充当“没有答案”的哨兵。

```python
def first_true(lo, hi, feasible):
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

例如有 `piles` 堆任务，每小时处理一堆中的至多 `speed` 个单位，允许 `hours` 小时：

```python
def minimum_speed(piles, hours):
    # 所有堆为正数，且 hours >= len(piles)
    def feasible(speed):
        need = sum((x + speed - 1) // speed for x in piles)
        return need <= hours

    return first_true(1, max(piles) + 1, feasible)
```

时间 O(n log U)，`U=max(piles)`。如果无法保证存在答案，返回后要检查哨兵。C++ 常写 `mid=lo+(hi-lo)/2`，并确认区间差不会溢出。

## 5. BFS 与 DFS

无权图 BFS 按距离分层；在入队时标记访问，避免同一个点被反复加入。

```python
from collections import deque

def bfs(graph, start):
    dist = [-1] * len(graph)
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in graph[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist
```

邻接表时间 O(V+E)，额外空间 O(V)。多源 BFS 把所有源点的距离置 0 后一起入队。

DFS 常用于连通块、树遍历和回溯。链状图可能递归到 O(V) 层；Python 增大递归上限也不能保证底层栈安全，深图优先用显式栈。回溯中修改的路径、访问标记需要在返回时恢复；普通图遍历的访问标记通常不恢复。

## 6. Dijkstra：非负权最短路

下面用小根堆，允许同一个点多次入堆。弹出后跳过旧距离，避免重复扩展。

```cpp
vector<long long> dijkstra(
    const vector<vector<pair<int, int>>>& graph, int start) {
    const long long INF = numeric_limits<long long>::max() / 4;
    vector<long long> dist(graph.size(), INF);
    using Node = pair<long long, int>;
    priority_queue<Node, vector<Node>, greater<Node>> heap;
    dist[start] = 0;
    heap.push({0, start});
    while (!heap.empty()) {
        auto [d, u] = heap.top();
        heap.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : graph[u]) {
            if (d + w < dist[v]) {
                dist[v] = d + w;
                heap.push({dist[v], v});
            }
        }
    }
    return dist;
}
```

前提是边权非负，合法最短距离小于 `INF`，加法在类型范围内。简单图的常用时间上界为 O((V+E) log V)；允许任意多重边时，可写成 O(V+E log(E+1))。存在负权时需另选算法；边权只有 0/1 时可考虑 0-1 BFS。

## 7. 并查集与拓扑排序

并查集使用路径压缩和按大小合并，用于判断无向图连通性。

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def merge(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True
```

单次操作均摊 O(α(n))。普通并查集不能直接处理删边。

有向依赖关系使用拓扑排序：统计入度，把入度为 0 的点入队；弹出一个点后减少后继入度，减到 0 再入队。处理点数不足 V 说明图中有环，时间 O(V+E)。并查集不能替代有向图判环。

## 8. 动态规划：先定义状态再写转移

状态必须说明“处理到哪里、保留了什么信息、值代表什么”。初始化区分可达和不可达，遍历方向由依赖关系决定。

0/1 背包每件物品最多选一次，容量倒序；完全背包每件物品可重复选，常用容量正序。

```python
def knapsack_01(weights, values, capacity):
    # dp[c]：容量不超过 c 时的最大价值，允许不选物品
    dp = [0] * (capacity + 1)
    for w, value in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + value)
    return dp[capacity]
```

时间 O(nC)，空间 O(C)。若要求“恰好装满”，只有 `dp[0]=0`，其余初始化为不可达，不能全部置 0。

严格递增子序列长度使用 `tails[i]` 保存长度为 `i+1` 的递增子序列中最小的末尾值：

```python
from bisect import bisect_left

def lis_length(a):
    tails = []
    for x in a:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
```

时间 O(n log n)。非严格递增改用 `bisect_right`；`tails` 本身不一定是原数组的一条合法子序列，恢复方案需要另外保存前驱。

## 9. 单调栈与单调队列

下一个更大元素：栈中保留尚未找到答案的下标，值从栈底到栈顶单调不增。

```python
def next_greater(a):
    answer = [-1] * len(a)  # 返回下标，-1 表示不存在
    stack = []
    for i, x in enumerate(a):
        while stack and a[stack[-1]] < x:
            answer[stack.pop()] = i
        stack.append(i)
    return answer
```

每个下标最多入栈、出栈一次，时间 O(n)。求窗口最大值可用单调队列：队首移除过期下标，队尾移除不优于新元素的下标，再把新下标入队。保存下标才能判断是否过期；相等元素如何处理要和题目的计数要求一致。
