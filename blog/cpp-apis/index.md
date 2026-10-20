---
title: "C++ 常用接口"
date: 2026-10-20T16:40:00+08:00
lastmod: 2026-10-20T17:15:00+08:00
draft: false
description: "C++17 容器、排序、二分、string.substr、数值边界与标准算法"
slug: "cpp-apis"
tags: ["cpp", "标准库"]
categories: ["笔试重点"]
comments: true
math: true
---

# C++ 常用接口

- [C++ 常用接口](#c-常用接口)
  - [1. vector 与区间操作](#1-vector-与区间操作)
  - [2. 排序、反转与去重](#2-排序反转与去重)
  - [3. 二分查找](#3-二分查找)
  - [4. map、set 与哈希容器](#4-mapset-与哈希容器)
  - [5. 堆、队列与 deque](#5-堆队列与-deque)
  - [6. string](#6-string)
    - [6.1 substr 与区间](#61-substr-与区间)
    - [6.2 查找、修改与转换](#62-查找修改与转换)
  - [7. 数值与初始化](#7-数值与初始化)
    - [7.1 整数边界](#71-整数边界)
    - [7.2 浮点边界](#72-浮点边界)
    - [7.3 最值与距离初始化](#73-最值与距离初始化)
  - [8. 常用标准算法](#8-常用标准算法)
  - [参考资料](#参考资料)

C++17 笔试接口，按容器、排序、查找和字符串操作分类。各代码块独立，包含所需头文件；区间默认采用 `[first,last)`。

## 1. vector 与区间操作

| 操作 | 接口 | 复杂度 |
|------|------|------|
| 尾部追加 / 删除 | `push_back` / `pop_back` | 均摊 O(1) / O(1) |
| 长度 / 容量 | `size` / `capacity` | O(1) |
| 中间插入 / 删除 | `insert` / `erase` | O(n) |
| 下标访问 | `operator[]` / `at` | O(1) |
| 首尾元素 | `front` / `back` | O(1) |

```cpp
#include <vector>
using namespace std;

vector<int> a(5, 0);
a.reserve(100);     // 改容量，不改 size
a.resize(8, -1);    // 改长度，新增位置填 -1
vector<vector<int>> grid(3, vector<int>(4, 0));

vector<int> values{10, 20, 30, 40};
// 迭代器范围为 [first,last)，复制下标 1、2 的元素。
vector<int> middle(values.begin() + 1, values.begin() + 3); // {20,30}
auto it = values.erase(values.begin() + 1); // 删除 20，返回指向后继 30 的迭代器
it = values.insert(it, 99);                // {10,99,30,40}，返回指向新元素的迭代器
values.erase(values.begin() + 1, values.begin() + 3); // 删除 [1,3)，得到 {10,40}
```

`reserve` 只改变容量，`resize` 改变长度。`at(i)` 越界会抛出异常，`operator[]` 不检查越界；访问首尾、弹出元素之前要求容器非空。

扩容可能使全部迭代器、指针和引用失效；`erase` 返回后继位置，删除循环中用该返回值继续遍历。

## 2. 排序、反转与去重

```cpp
#include <algorithm>
#include <utility>
#include <vector>
using namespace std;

vector<pair<int, int>> a{{2, 5}, {1, 3}, {2, 1}};
sort(a.begin(), a.end()); // pair 先比较 first，再比较 second
sort(a.begin(), a.end(), [](const auto& x, const auto& y) {
    if (x.first != y.first) return x.first > y.first; // 第一关键字降序
    return x.second < y.second;
});
reverse(a.begin(), a.end());

vector<int> b{3, 1, 3, 2};
sort(b.begin(), b.end());
b.erase(unique(b.begin(), b.end()), b.end()); // 1 2 3
```

比较器应使用严格比较，不能写 `x <= y`。`unique` 只把相邻重复元素移到逻辑末尾，不会自动缩短容器；全局去重通常先排序。

`sort` 通常按 O(n log n) 考虑，不保证稳定。需要保留相等元素的原始顺序时使用 `stable_sort`；C++17 中辅助内存不足时，它的比较次数上界可变为 O(n log² n)。

## 3. 二分查找

范围须按相同规则有序。`lower_bound` 找第一个 `>=x`，`upper_bound` 找第一个 `>x`；相减得到 x 的出现次数。

```cpp
#include <algorithm>
#include <vector>
using namespace std;

vector<int> a{1, 2, 2, 4};
auto l = lower_bound(a.begin(), a.end(), 2); // 第一个 >= 2 的位置
auto r = upper_bound(a.begin(), a.end(), 2); // 第一个 > 2 的位置
int count = int(r - l); // 2
int pos = int(l - a.begin());
bool found = l != a.end() && *l == 2; // 先检查 end，再解引用
```

vector 上查询 O(log n)。set/map 使用成员函数 `lower_bound`；通用算法在非随机访问迭代器上可能执行线性次数的迭代器移动。

## 4. map、set 与哈希容器

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

## 5. 堆、队列与 deque

```cpp
#include <deque>
#include <functional>
#include <queue>
#include <vector>
using namespace std;

priority_queue<int> big; // 大根堆
priority_queue<int, vector<int>, greater<int>> small; // 小根堆
small.push(3);
int x = small.top();    // 查看堆顶；访问前要求堆非空
small.pop();            // pop 没有返回值
queue<int> q;
q.push(1);
int front = q.front();
q.pop();
deque<int> dq{2, 3};
dq.push_front(1);       // 两端都能插入、删除
dq.push_back(4);
dq.pop_front();
```

堆插入、删除堆顶 O(log n)，查看堆顶 O(1)。deque 两端插入、删除 O(1)；queue 只开放队首读取和队尾加入。

## 6. string

### 6.1 substr 与区间

`s.substr(pos,count)` 从 pos 开始，最多复制 count 个字符。第二个参数是**长度**，不是右端点。

| 写法 | 含义 |
|------|------|
| `s.substr(pos,len)` | 从 pos 开始复制最多 len 个字符 |
| `s.substr(pos)` | 从 pos 复制到结尾 |
| `s.substr(l,r-l)` | 对应 `[l,r)`，要求 `0<=l<=r<=s.size()` |
| `s.substr(s.size())` | 返回空字符串 |

```cpp
#include <string>
using namespace std;

string s = "abcdef";
string part = s.substr(1, 3);     // "bcd"，取下标 1、2、3
string suffix = s.substr(3);     // "def"，省略长度表示取到结尾
string clipped = s.substr(4, 9); // "ef"，长度超过剩余字符数时截到结尾
string empty = s.substr(s.size()); // ""，pos 恰好等于 size 合法
int l = 1, r = 4;
string interval = s.substr(l, r - l); // "bcd"，用长度表达 [l,r)
// s.substr(7) 会抛出 out_of_range；不能照搬 Python 的边界截断规则。
```

`pos>s.size()` 抛出 `out_of_range`；负数传入 pos/count 会转换为无符号数，也不是 Python 的负下标语义。复制 k 个字符需要 O(k) 时间和空间，原字符串不变。

C++17 的 `string_view::substr` 返回视图，O(1) 且不复制字符，但原字符串必须仍然存活，且底层存储不能失效；它的 pos 越界同样会抛出异常。

### 6.2 查找、修改与转换

```cpp
#include <string>
using namespace std;

string s = "abcabc";
auto pos = s.find("bc");     // 1，第一次出现的位置
auto last = s.rfind("bc");   // 4，最后一次出现的位置
if (pos != string::npos) s.replace(pos, 2, "X"); // "aXabc"，替换 pos 起的 2 个字符
s.insert(1, "YZ");         // "aYZXabc"，在位置 1 前插入
s.erase(1, 2);              // "aXabc"，删除位置 1 起的 2 个字符
long long value = stoll("12345678900");
string text = to_string(value);
bool prefix = s.compare(0, 2, "aX") == 0; // C++17 检查前缀
```

`find/rfind` 找不到时返回 `string::npos`，应使用原返回类型与 npos 比较，再访问或截取。`erase(pos,count)`、`replace(pos,count,text)` 的 count 也表示长度；`erase(first,last)` 则是迭代器半开区间。

C++17 可用 `compare` 检查前后缀；`starts_with/ends_with` 是 C++20 接口。`stoi/stoll/stod` 解析失败或超出数值范围会抛出异常，若要求整串都被解析，应检查可选的结束位置参数。

`string` 的长度和下标按 `char` 元素计算，UTF-8 中文不能按一个字节切分成字符。

## 7. 数值与初始化

| 操作 | C++17 接口 |
|------|------|
| 最大公约数 / 最小公倍数 | `<numeric>` 的 `gcd` / `lcm` |
| 求和 / 前缀和 | `accumulate` / `partial_sum` |
| 向上 / 向下取整 | `<cmath>` 的 `ceil` / `floor` |
| 置位数 | GCC 扩展 `__builtin_popcountll(x)` |
| 最大值界限 | `<limits>` 的 `numeric_limits<T>::max()` |
| 按元素填充 | `fill(first,last,value)` |

整数除法向零截断，例如 `-3/2==-1`；需要向下取整时不能直接照搬 Python 的 `//`。乘法前用 `1LL*a*b` 提升类型，仍需保证最终结果不溢出。

### 7.1 整数边界

C++ 宏名区分大小写，写 `INT_MAX/INT_MIN`。`<climits>` 提供整数边界；`<limits>` 的 `numeric_limits<T>` 可以按类型统一查询。

| 类型 | 最小值 | 最大值 |
|------|------|------|
| `int` | `INT_MIN` | `INT_MAX` |
| `long long` | `LLONG_MIN` | `LLONG_MAX` |
| `unsigned int` | `0` | `UINT_MAX` |
| `unsigned long long` | `0` | `ULLONG_MAX` |
| 整数类型 `T` | `numeric_limits<T>::min()` 或 `lowest()` | `numeric_limits<T>::max()` |

常见 32 位 int 的范围为 `[-2147483648,2147483647]`，64 位 long long 为 `[-2^63,2^63-1]`；代码应使用接口读取边界，避免依赖手写常量。

```cpp
#include <climits>
#include <limits>
using namespace std;

int int_max = INT_MAX;
int int_min = INT_MIN;
long long ll_max = LLONG_MAX;
long long ll_min = numeric_limits<long long>::lowest(); // 等于 LLONG_MIN
unsigned int uint_max = UINT_MAX; // 无符号整数的下界为 0
```

边界值不能当成可继续任意运算的普通值。`INT_MAX+1`、`-INT_MIN`、`-LLONG_MIN` 都会发生有符号整数溢出，行为未定义；无符号整数运算则按其位宽取模。

### 7.2 浮点边界

`<cfloat>` 提供 `FLT_*`、`DBL_*`、`LDBL_*`，分别对应 float、double、long double。**浮点数的 `min()` / `FLT_MIN` 表示最小正正规数，不是最负值。**

| 含义 | float | double |
|------|------|------|
| 最大有限值 | `FLT_MAX` | `DBL_MAX` |
| 最小正正规数 | `FLT_MIN` | `DBL_MIN` |
| 最负有限值 | `numeric_limits<float>::lowest()` | `numeric_limits<double>::lowest()` |
| 最小正非正规数，支持时 | `numeric_limits<float>::denorm_min()` | `numeric_limits<double>::denorm_min()` |
| 1 与下一个可表示数的间距 | `FLT_EPSILON` | `DBL_EPSILON` |
| 正无穷，支持时 | `numeric_limits<float>::infinity()` | `numeric_limits<double>::infinity()` |

常见 IEEE 754 格式下，float 的最大有限值约 `3.402823e38`，最小正正规数约 `1.175494e-38`；double 分别约 `1.797693e308`、`2.225074e-308`。最负有限值分别为对应最大有限值的相反数。

```cpp
#include <cfloat>
#include <cmath>
#include <limits>
using namespace std;

float float_max = FLT_MAX;
float float_min = FLT_MIN; // 正数；不要用来初始化“寻找最大值”
double lowest = numeric_limits<double>::lowest();
double tiny = numeric_limits<double>::denorm_min(); // IEEE 754 binary64 下约 4.94e-324
double eps = numeric_limits<double>::epsilon();    // 等于 DBL_EPSILON
static_assert(numeric_limits<double>::has_infinity, "requires infinity");
double inf = numeric_limits<double>::infinity();
bool infinite = isinf(inf);        // true；负无穷为 -inf
bool finite = isfinite(DBL_MAX);   // true；最大有限值与无穷不同
```

`epsilon()` 是 1 附近的间距，不是所有数量级通用的误差阈值；比较容差应结合题目要求和数据范围。

### 7.3 最值与距离初始化

```cpp
#include <algorithm>
#include <limits>
#include <vector>
using namespace std;

vector<int> a{-8, -3, -5}; // 已知非空
int minimum = numeric_limits<int>::max();
int maximum = numeric_limits<int>::lowest();
for (int x : a) {
    minimum = min(minimum, x); // -8
    maximum = max(maximum, x); // -3；初始化为 0 会错
}

// 假设合法路径长度不超过 10^15，边权非负且不超过 10^9。
const long long INF = numeric_limits<long long>::max() / 4;
long long dist_u = INF, dist_v = INF, weight = 7;
if (dist_u != INF) { // 不可达状态不参与加法
    dist_v = min(dist_v, dist_u + weight);
}
```

`max()/4` 只是留出加法余量，不保证任意输入都安全：INF 必须严格大于所有合法答案，参与运算的中间结果也须在类型范围内。浮点最值可用正负无穷初始化；整数没有无穷值。

`memset` 按字节写入，不能通用地把整数数组填成任意整数。模幂与逆元见 [笔试常见技巧](../interview-tricks/index.md)。

## 8. 常用标准算法

```cpp
#include <algorithm>
#include <numeric>
#include <vector>
using namespace std;

vector<int> a{3, 1, 3, 2};
long long total = accumulate(a.begin(), a.end(), 0LL); // 9，初值决定累加类型
int occurrences = int(count(a.begin(), a.end(), 3));  // 2，扫描计数
auto it = find(a.begin(), a.end(), 2);                // 指向第一个 2；不存在返回 end
auto largest = max_element(a.begin(), a.end());       // 指向最大元素，空范围返回 end
vector<int> prefix(a.size());
partial_sum(a.begin(), a.end(), prefix.begin());      // {3,4,7,9}，输出空间需提前准备
vector<int> order(4);
iota(order.begin(), order.end(), 0);                  // {0,1,2,3}，连续赋值
fill(order.begin(), order.end(), -1);                // {-1,-1,-1,-1}，按元素赋值
```

上述操作均为 O(n)。`find/count` 不要求有序，`lower_bound` 要求范围按对应规则有序。`max_element/min_element` 返回迭代器，解引用前检查是否等于 end；`accumulate` 的 `0` 与 `0LL` 可能导致不同的溢出结果。

## 参考资料

- [C++ 标准草案：容器](https://eel.is/c++draft/containers)
- [C++ 标准草案：string.substr](https://eel.is/c++draft/string.substr)
- [C++ 标准草案：算法库](https://eel.is/c++draft/algorithms)
- [C++ 标准草案：整数边界宏](https://eel.is/c++draft/climits.syn)
- [C++ 标准草案：浮点边界宏](https://eel.is/c++draft/cfloat.syn)
- [C++ 标准草案：numeric_limits](https://eel.is/c++draft/numeric.limits.members)
