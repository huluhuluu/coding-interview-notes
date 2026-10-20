---
title: "Python 常用接口"
date: 2026-10-20T16:40:00+08:00
lastmod: 2026-10-20T16:40:00+08:00
draft: false
description: "Python 列表、切片、字符串、字典、堆与遍历接口的语义和边界"
slug: "python-apis"
tags: ["python", "标准库"]
categories: ["笔试重点"]
comments: true
math: true
---

# Python 常用接口

- [Python 常用接口](#python-常用接口)
  - [1. list 与增删查找](#1-list-与增删查找)
  - [2. 切片](#2-切片)
  - [3. 排序、反转与去重](#3-排序反转与去重)
  - [4. bisect 二分](#4-bisect-二分)
  - [5. dict、set 与计数](#5-dictset-与计数)
  - [6. heapq 与 deque](#6-heapq-与-deque)
  - [7. 字符串](#7-字符串)
  - [8. 数值与转换](#8-数值与转换)
  - [9. 遍历与归约](#9-遍历与归约)
  - [参考资料](#参考资料)

Python 笔试接口，覆盖列表、切片、字符串、字典、堆与遍历。各代码块独立；需要 Python 3.9+ 的接口单独标出。数组与张量操作见 [NumPy / PyTorch 常用操作](../numpy-torch-apis/index.md)。

## 1. list 与增删查找

```python
a = [0] * 5
grid = [[0] * 4 for _ in range(3)]
b = a[:]                    # 一层副本
```

不要用 `[[0] * m] * n` 创建需要逐行修改的矩阵，这会让每行引用同一个列表。Python 浅拷贝也不会递归复制内部列表。

```python
a = [1, 2, 2, 3]
a.append([4, 5])      # 把整个列表作为一个元素加入
a.extend([6, 7])      # 把可迭代对象的各元素依次加入
index = a.index(2)    # 1，第一个等于 2 的位置；不存在时抛 ValueError
count = a.count(2)    # 2，扫描整个列表计数
a.remove(2)          # 只删除第一个 2，按值删除
last = a.pop()        # 7，按位置删除并返回；默认删除末尾
del a[1]             # 按下标删除，不返回被删元素
```

`list.index/count/remove` 和 `x in list` 都可能扫描 O(n) 个元素。

`append/extend/insert/remove/sort/reverse` 修改原列表并返回 None；`pop` 返回被删元素。中间插入、删除需移动元素，通常 O(n)，尾部追加均摊 O(1)。

## 2. 切片

列表、元组和字符串均支持 `seq[start:stop:step]`。取值沿 step 方向移动，**不包含 stop**。列表切片创建新列表，内部对象仍共享引用；字符串切片返回子串，不能原地修改源字符串。

| 写法 | 含义 |
|------|------|
| `a[l:r]` | 下标区间 `[l,r)` |
| `a[:r]` / `a[l:]` | 开头到 r 之前 / 从 l 到结尾 |
| `a[-k:]` | 最后 k 个元素，k>0 |
| `a[::k]` | 从开头每隔 k 个下标取一个，k>0 |
| `a[::-1]` | 逆序副本 |
| `a[start:stop:-1]` | 从 start 向前取，遇到 stop 之前停止 |
| `a[:]` | 一层副本 |

```python
a = [0, 1, 2, 3, 4, 5]
middle = a[1:5]       # [1,2,3,4]，stop=5 不包含
suffix = a[-3:]       # [3,4,5]，负下标从末尾计数
even = a[::2]         # [0,2,4]，步长 2
backward = a[5:1:-1]  # [5,4,3,2]，倒序也不包含 stop
reversed_a = a[::-1]  # [5,4,3,2,1,0]，负步长的默认边界适合取完整逆序
clipped = a[-100:100] # [0,1,2,3,4,5]，切片边界自动截到有效范围
empty = a[3:1]        # []，正步长无法从 3 走到 1
also_empty = a[:-1:-1] # []；显式 -1 是最后一个下标，不是省略的反向 stop
```

单元素访问 `a[len(a)]` 抛出 `IndexError`，切片 `a[len(a):]` 返回空列表。step 不能为 0，否则抛出 `ValueError`。`a[-0:]` 等于 `a[:]`，不能用它表示“取最后 0 个”。

列表还支持切片赋值与删除；字符串、元组不支持原地切片赋值：

```python
a = [0, 1, 2, 3, 4, 5]
b = a[:]
b[1:3] = [8]         # [0,8,3,4,5]，普通切片赋值允许改变长度
c = a[:]
c[::2] = [6, 7, 8]   # [6,1,7,3,8,5]；扩展切片要求右侧长度等于选中位置数
del c[1::2]           # [6,7,8]，删除原下标 1、3、5 的元素
```

取 k 个元素的列表/字符串切片通常需要 O(k) 时间和空间。循环内反复截取长区间时，应把复制成本计入复杂度。

## 3. 排序、反转与去重

```python
a = [(2, 5), (1, 3), (2, 1)]
a.sort(key=lambda p: (-p[0], p[1])) # 第一项降序，第二项升序
b = sorted(a)                 # 新列表
a.reverse()                   # 原地反转，返回 None
values = sorted(set([3, 1, 3, 2]))
```

`list.sort` 原地排序，`sorted` 返回新列表，时间通常 O(n log n)。排序稳定；key 可以返回元组表达多级排序。set 去重不保留排序语义，`sorted(set(a))` 才得到有序结果。

## 4. bisect 二分

数组必须有序。`bisect_left(a,x)` 找第一个 `>=x`，`bisect_right(a,x)` 找第一个 `>x`；两者相减得到出现次数。

```python
from bisect import bisect_left, bisect_right, insort

a = [1, 2, 2, 4]
l, r = bisect_left(a, 2), bisect_right(a, 2)
found = l < len(a) and a[l] == 2
insort(a, 3)                         # 保持有序，但插入要移动后面的元素
```

查询 O(log n)，`insort` 还需移动列表元素，整体 O(n)。边界可能等于 len(a)，读取该位置前先检查范围。

## 5. dict、set 与计数

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

d.setdefault("b", 0)      # 不存在才插入默认值，返回该键对应的值
removed = d.pop("b", -1)  # 删除并返回；不存在时返回 -1
pairs = list(d.items())   # 遍历时常写 for key, value in d.items()
s.add(4)                 # 加入单个元素
s.update([4, 5])          # 加入多个元素
intersection = s & {2, 5} # 交集；| 为并集，- 为差集，^ 为对称差
```

Python `dict/set` 查询与修改平均 O(1)，最坏可退化为 O(n)。`dict` 保留插入顺序，`set` 不提供排序语义。遍历字典时不要增删键；需要删除时先保存待删键。

## 6. heapq 与 deque

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
q.appendleft(0)          # 从左端加入；pop() 从右端删除
top_two = heapq.nlargest(2, [4, 1, 3]) # [4,3]，结果按降序排列
```

堆的插入、删除堆顶 O(log n)，查看堆顶 O(1)。双端队列两端插入删除 O(1)。BFS 不要使用 Python `list.pop(0)`，它需要移动剩余元素。

## 7. 字符串

```python
s = "abcabc"
part = s[1:4]               # "bca"，沿用列表的切片语义
pos = s.find("bc")           # 不存在时返回 -1
last = s.rfind("bc")         # 4，从右侧寻找最后一次出现
count = s.count("ab")        # 2，按非重叠匹配计数
changed = s.replace("ab", "X", 1) # "Xcabc"，只替换第一次，返回新字符串
prefix = s.startswith("ab") and s.endswith("bc")
text = ",".join(["a", "b"])
words = "a  b\tc".split()    # 连续空白合并
fields = "a,,b".split(",")   # 保留中间空字段
left, sep, right = "a=b=c".partition("=") # ("a","=","b=c")，仅分割第一次
last_split = "a=b=c".rsplit("=", 1)       # ["a=b","c"]，从右侧分割一次
```

`str.find` 不存在时返回 -1，`str.index` 则抛出 `ValueError`。不要用 `if s.find(sub)` 判断存在性：找到下标 0 时是假，返回 -1 时反而是真。

`strip/lstrip/rstrip` 按**字符集合**裁剪两端，例如 `"abbaXba".strip("ab")` 得到 `"X"`。精确删除前后缀可用 `removeprefix/removesuffix`，需要 Python 3.9+。

```python
line = "  hello\n"
content = line.rstrip("\r\n") # "  hello"，保留行首空格
trimmed = line.strip()        # "hello"，两端空白都移除
name = "pre_name.txt".removeprefix("pre_").removesuffix(".txt") # "name"，Python 3.9+
numbers = [1, 2, 3]
joined = " ".join(map(str, numbers)) # "1 2 3"，join 的各元素必须是字符串
code = ord("A")              # 65，字符转码点
char = chr(65)               # "A"，码点转字符
```

Python 字符串不可原地修改，`replace/strip` 等方法返回新字符串。反复拼接长字符串时可先收集片段再 `join`。Python `str` 按 Unicode 码点索引。

## 8. 数值与转换

| 操作 | 接口 |
|------|------|
| 最大公约数 | `math.gcd(a,b)` |
| 整数平方根 | `math.isqrt(n)`，n>=0 |
| 商和余数 | `divmod(a,b)` |
| 模幂 | `pow(a,e,mod)` |
| 整数转换 | `int(text,base)` / `str(value)` |
| 二进制 / 十六进制 | `bin(x)` / `hex(x)` |
| 码点转换 | `ord(ch)` / `chr(code)` |

`//` 向下取整，`-3//2==-2`；`int(-1.5)==-1` 是向零截断。`int.bit_count()` 需要 Python 3.10+。模逆元与组合数见 [笔试常见技巧](../interview-tricks/index.md)。

## 9. 遍历与归约

```python
a = [3, 1, 3, 2]
indexed = list(enumerate(a, start=1)) # [(1,3),(2,1),(3,3),(4,2)]
pairs = list(zip([1, 2, 3], ["a", "b"])) # [(1,"a"),(2,"b")]，默认到较短的一侧结束
steps = list(range(5, 0, -2))        # [5,3,1]，不包含右端点 0
positive = all(x > 0 for x in a)     # True，所有元素满足条件
has_even = any(x % 2 == 0 for x in a) # True，任意元素满足条件
minimum = min([], default=-1)       # -1，空序列时使用默认值
total = sum(x * x for x in a)       # 23，生成器逐项产生平方
```

`range` 可重复遍历；`map/zip/enumerate` 返回一次性迭代器，消费后不能自动从头开始。通常无需先转 list。`all/any` 会短路，空可迭代对象上分别返回 True/False。`sum/min/max` 通常扫描 O(n) 个元素。

## 参考资料

- [Python 序列、列表与字符串](https://docs.python.org/3/library/stdtypes.html)
- [Python 内置函数](https://docs.python.org/3/library/functions.html)
- [collections](https://docs.python.org/3/library/collections.html)
- [heapq](https://docs.python.org/3/library/heapq.html)
- [bisect](https://docs.python.org/3/library/bisect.html)
