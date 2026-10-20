---
title: "C++ / Python 常用接口"
date: 2026-10-20T12:10:00+08:00
draft: false
description: "整理笔试常用的容器、排序、二分、堆、字符串与计数接口，标明复杂度和容易写错的语义"
slug: "acm-common-apis"
tags: ["cpp", "python", "acm", "标准库"]
categories: ["笔试重点"]
comments: true
math: true
---

# C++ / Python 常用接口

默认使用 C++17 和 Python 3。选择接口时同时确认返回值、是否修改原对象，以及复杂度。下面的片段相互独立，C++ 片段省略 `std::`；实际使用时包含对应头文件。

## 1. 顺序容器

| 操作 | C++ `vector` | Python `list` | 复杂度 |
|------|--------------|---------------|--------|
| 尾部追加 | `a.push_back(x)` | `a.append(x)` | 均摊 O(1) |
| 尾部删除 | `a.pop_back()` | `a.pop()` | O(1) |
| 获取长度 | `a.size()` | `len(a)` | O(1) |
| 下标访问 | `a[i]` | `a[i]` | O(1) |
| 中间插入/删除 | `insert` / `erase` | `insert` / `pop(i)` | O(n) |
| 截取副本 | 迭代器范围构造 | `a[l:r]` | O(r-l) |

```cpp
#include <vector>
using namespace std;

vector<int> a(5, 0);
a.reserve(100);     // 改容量，不改 size
a.resize(8, -1);    // 改长度，新增位置填 -1
vector<vector<int>> grid(3, vector<int>(4, 0));
```

```python
a = [0] * 5
grid = [[0] * 4 for _ in range(3)]
b = a[:]                    # 一层副本
```

不要用 `[[0] * m] * n` 创建需要逐行修改的矩阵，这会让每行引用同一个列表。Python 浅拷贝也不会递归复制内部列表。

C++ `vector` 扩容可能使已有迭代器、指针和引用失效。删除过程中不要继续使用被删除位置的旧迭代器。

## 2. 排序、反转与去重

```cpp
#include <algorithm>
#include <utility>
#include <vector>
using namespace std;

vector<pair<int, int>> a{{2, 5}, {1, 3}, {2, 1}};
sort(a.begin(), a.end()); // pair 先比较 first，再比较 second
sort(a.begin(), a.end(), [](const auto& x, const auto& y) {
    if (x.first != y.first) return x.first > y.first;
    return x.second < y.second;
});
reverse(a.begin(), a.end());

vector<int> b{3, 1, 3, 2};
sort(b.begin(), b.end());
b.erase(unique(b.begin(), b.end()), b.end()); // 1 2 3
```

比较器应使用严格比较，不能写 `x <= y`。`unique` 只把相邻重复元素移到逻辑末尾，不会自动缩短容器；全局去重通常先排序。

```python
a = [(2, 5), (1, 3), (2, 1)]
a.sort(key=lambda p: (-p[0], p[1]))
b = sorted(a)                 # 新列表
a.reverse()                   # 原地反转，返回 None
values = sorted(set([3, 1, 3, 2]))
```

Python 排序稳定；C++ `sort` 不保证相等元素的原始顺序，需要时用 `stable_sort`。两者排序通常按 O(n log n) 考虑。C++17 `stable_sort` 在辅助内存不足时比较次数上界会变为 O(n log² n)。

## 3. 二分接口

数组必须已经按同样规则有序。以下使用升序数组。

| 目标 | C++ | Python |
|------|-----|--------|
| 第一个 `>= x` | `lower_bound` | `bisect_left` |
| 第一个 `> x` | `upper_bound` | `bisect_right` |
| 等于 x 的个数 | 两个边界之差 | 两个边界之差 |

```cpp
#include <algorithm>
#include <vector>
using namespace std;

vector<int> a{1, 2, 2, 4};
auto l = lower_bound(a.begin(), a.end(), 2);
auto r = upper_bound(a.begin(), a.end(), 2);
int count = int(r - l); // 2
int pos = int(l - a.begin());
bool found = l != a.end() && *l == 2;
```

```python
from bisect import bisect_left, bisect_right, insort

a = [1, 2, 2, 4]
l, r = bisect_left(a, 2), bisect_right(a, 2)
found = l < len(a) and a[l] == 2
insort(a, 3)
```

数组上的边界查询是 O(log n)，但 Python `insort` 还需要移动元素，整体是 O(n)。C++ `set/map` 应使用成员函数 `lower_bound`；通用算法在非随机访问迭代器上可能执行线性次数的迭代器移动。

## 4. 哈希表、集合与计数

```cpp
#include <map>
#include <set>
#include <unordered_map>
using namespace std;

unordered_map<int, int> freq;
++freq[7];                          // 不存在时插入值 0，再自增
bool exists = freq.find(8) != freq.end(); // 查询不插入
freq.erase(7);
map<int, int> ordered;
set<int> values{1, 3, 5};
auto it = values.lower_bound(2);     // 指向 3
```

C++17 没有标准容器的 `contains` 成员函数，可用 `find`。`map/set` 查询和修改 O(log n)；`unordered_map/unordered_set` 平均 O(1)，最坏 O(n)。哈希集合不保证有序。

```python
from collections import Counter, defaultdict

freq = Counter([2, 2, 3])
freq[9]                   # 0，不插入该键
d = {"a": 1}
d.get("b", 0)             # 默认值，不插入
groups = defaultdict(list)
groups[2].append(7)        # 访问不存在的键时创建列表
s = {1, 2, 3}
s.discard(9)              # 不存在也不报错
```

Python `dict/set` 查询与修改平均 O(1)，最坏可退化为 O(n)。`dict` 保留插入顺序，`set` 不提供排序语义。遍历字典时不要增删键；需要删除时先保存待删键。

## 5. 堆、队列与双端队列

```cpp
#include <functional>
#include <queue>
#include <vector>
using namespace std;

priority_queue<int> big; // 大根堆
priority_queue<int, vector<int>, greater<int>> small; // 小根堆
small.push(3);
int x = small.top();
small.pop();            // pop 没有返回值
queue<int> q;
q.push(1);
int front = q.front();
q.pop();
```

```python
import heapq
from collections import deque

heap = [4, 1, 3]
heapq.heapify(heap)       # O(n) 建小根堆
heapq.heappush(heap, 2)
x = heapq.heappop(heap)
largest = []             # 兼容旧评测环境：负数模拟大根堆
heapq.heappush(largest, -7)
q = deque([1])
q.append(2)
front = q.popleft()
```

堆的插入、删除堆顶 O(log n)，查看堆顶 O(1)。双端队列两端插入删除 O(1)。BFS 不要使用 Python `list.pop(0)`，它需要移动剩余元素。

## 6. 字符串

```cpp
#include <string>
using namespace std;

string s = "abcabc";
string part = s.substr(1, 3); // bca，第二个参数是长度
auto pos = s.find("bc");
if (pos != string::npos) s.replace(pos, 2, "X");
long long value = stoll("12345678900");
string text = to_string(value);
```

```python
s = "abcabc"
part = s[1:4]               # bca，右端点不包含
pos = s.find("bc")           # 不存在时返回 -1
text = ",".join(["a", "b"])
words = "a  b\tc".split()    # 连续空白合并
fields = "a,,b".split(",")   # 保留中间空字段
```

Python 字符串不可原地修改，频繁追加建议先保存片段再 `join`。C++ `substr` 和 Python 切片通常创建副本，循环里反复复制长字符串会增加时间和内存开销。

## 7. 数值、位运算与初始化

| 操作 | C++17 | Python 3 |
|------|-------|----------|
| 最大公约数 | `<numeric>` 中 `gcd(a,b)` | `math.gcd(a,b)` |
| 求和 | `accumulate(a.begin(), a.end(), 0LL)` | `sum(a)` |
| 商与余数 | `/` 与 `%` | `divmod(a,b)` |
| 模幂 | 自写快速幂 | `pow(a, e, mod)` |
| 置位数 | GCC 扩展 `__builtin_popcountll(x)` | `int.bit_count()`，Python 3.10+ |

C++ 整数除法向零截断，Python `//` 向下取整，例如 `-3 / 2` 在 C++ 整数运算中为 `-1`，Python `-3 // 2` 为 `-2`。两种语言负数取余的结果也可能不同。

C++ 用 `numeric_limits<long long>::max() / 4` 等作为距离上界时，先确认它大于所有合法答案；不可达状态不能直接参与加法。`memset` 按字节写入，不能把整数数组通用地填成任意整数。

## 参考

- [Python 标准库：collections](https://docs.python.org/3/library/collections.html)
- [Python 标准库：heapq](https://docs.python.org/3/library/heapq.html)
- [Python 标准库：bisect](https://docs.python.org/3/library/bisect.html)
- [C++ 标准草案：算法库](https://eel.is/c++draft/algorithms)
