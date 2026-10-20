---
title: "笔试常见技巧"
date: 2026-10-20T12:30:00+08:00
lastmod: 2026-10-20T16:00:00+08:00
draft: false
description: "整理 lowbit、树状数组、快速幂、筛法、逆元、组合数、离散化和子集枚举的用法与前提"
slug: "interview-tricks"
tags: ["cpp", "python", "数论", "位运算"]
categories: ["笔试重点"]
comments: true
math: true
---

# 笔试常见技巧

- [笔试常见技巧](#笔试常见技巧)
  - [1. lowbit 与删除最低位的 1](#1-lowbit-与删除最低位的-1)
  - [2. 树状数组：单点修改、前缀求和](#2-树状数组单点修改前缀求和)
  - [3. 快速幂](#3-快速幂)
    - [3.1 C++](#31-c)
    - [3.2 Python](#32-python)
  - [4. 最大公约数与扩展欧几里得](#4-最大公约数与扩展欧几里得)
  - [5. 模逆元](#5-模逆元)
  - [6. 埃氏筛与线性筛](#6-埃氏筛与线性筛)
    - [6.1 Python](#61-python)
    - [6.2 C++](#62-c)
  - [7. 阶乘与逆阶乘求组合数](#7-阶乘与逆阶乘求组合数)
  - [8. 离散化](#8-离散化)
  - [9. 子集枚举与整除向上取整](#9-子集枚举与整除向上取整)
  - [参考资料](#参考资料)

位运算与数论模板，包括 lowbit、树状数组、快速幂、筛法、逆元与组合数。各节注明使用前提；代码使用 C++17 / Python 3，C++ 片段需包含相应头文件。

## 1. lowbit 与删除最低位的 1

对正整数 `x`，`x & -x` 取出最低位的 1 对应的数值；`x & (x-1)` 删除最低位的 1。

```text
x             = 12 = 1100₂
x & -x        =  4 = 0100₂
x & (x - 1)   =  8 = 1000₂
```

```cpp
unsigned long long lowbit(unsigned long long x) {
    return x & (0ULL - x);
}

int count_bits(unsigned long long x) {
    int count = 0;
    while (x) {
        x &= x - 1;
        ++count;
    }
    return count;
}
```

无符号运算按模 $2^w$ 回绕，避免对最小有符号整数取负的溢出问题。`lowbit(0)=0`，不能用它推进一个从 0 开始的树状数组更新循环。非负整数判断 2 的幂可用 `x > 0 && (x & (x-1)) == 0`。

## 2. 树状数组：单点修改、前缀求和

使用 **1-based** 下标。`tree[i]` 管理长度为 `lowbit(i)`、右端点为 `i` 的区间。

```python
class Fenwick:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i, delta):
        # 前提：1 <= i <= n
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, i):
        # 返回 [1, i] 的和，允许 i=0
        result = 0
        while i > 0:
            result += self.tree[i]
            i -= i & -i
        return result

    def range_sum(self, left, right):
        return self.prefix(right) - self.prefix(left - 1)
```

更新和查询均为 O(log n)。它适合可做前缀相减的求和类问题；动态区间最值不能直接照搬这个模板。

## 3. 快速幂

指数的二进制分解把 O(e) 次乘法减少为 O(log e) 次。下面限定 `e>=0`、`1<=mod<=2^31-1`，因此两个模内数的乘积可放进有符号 64 位整数。

### 3.1 C++

```cpp
long long mod_pow(long long a, long long e, long long mod) {
    a %= mod;
    if (a < 0) a += mod;
    long long result = 1 % mod;
    while (e > 0) {
        if (e & 1) result = result * a % mod;
        a = a * a % mod;
        e >>= 1;
    }
    return result;
}
```

### 3.2 Python

```python
answer = pow(3, 100, 1_000_000_007)
```

模数接近 64 位上限时，C++ 的 `a*b` 可能在取模前溢出，需要更宽中间类型或专门的模乘。矩阵快速幂使用同样结构，把标量乘法换为矩阵乘法，并把初始值换为单位矩阵。

## 4. 最大公约数与扩展欧几里得

欧几里得算法使用 `gcd(a,b)=gcd(b,a%b)`。扩展形式还求出 $ax+by=\gcd(a,b)$，用于求逆元和线性同余方程。

```python
def extended_gcd(a, b):
    # 本模板要求 a >= 0, b >= 0
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    return old_r, old_x, old_y
```

最小公倍数可写为 `a // gcd(a,b) * b`，先除后乘降低中间值；若两数都为 0，需要单独约定结果。C++ 即使先除再乘，最终结果仍可能溢出。

## 5. 模逆元

模 $m$ 下，$a^{-1}$ 满足 $aa^{-1}\equiv1\pmod m$。当且仅当 `gcd(a,m)=1` 时存在逆元。

若 `p` 为质数，且 `a % p != 0`，费马小定理给出 $a^{-1}\equiv a^{p-2}\pmod p$。

```python
p = 1_000_000_007
inverse = pow(3, p - 2, p)
answer = 7 * inverse % p  # 模意义下的 7 / 3
```

一般模数使用扩展欧几里得：

```python
def mod_inverse(a, mod):
    # 使用上一节 extended_gcd，要求 mod > 1
    g, x, _ = extended_gcd(a % mod, mod)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % mod
```

Python 3.8+ 也支持 `pow(a,-1,mod)`，不互质会抛出 `ValueError`。不要对合数模数直接套 `a^(mod-2)`。例如 2 在模 6 下没有逆元，模意义下的“除以 2”无法唯一成立。

## 6. 埃氏筛与线性筛

埃氏筛从每个质数的平方开始标记，因为更小的倍数已经被更小质因子处理。

### 6.1 Python

```python
from math import isqrt

def sieve(n):
    # n >= 0；返回 [0,n] 的质数标记
    prime = bytearray([1]) * (n + 1)
    prime[0] = 0
    if n >= 1:
        prime[1] = 0
    for p in range(2, isqrt(n) + 1):
        if prime[p]:
            for multiple in range(p * p, n + 1, p):
                prime[multiple] = 0
    return prime
```

时间 O(n log log n)，空间 O(n)。C++ 判断 `i*i<=n` 时要注意乘法溢出，可提升类型或使用 `i<=n/i`。

线性筛保存最小质因子，每个合数只由其最小质因子对应的分解标记一次：

### 6.2 C++

```cpp
vector<int> smallest_prime_factor(int n) {
    vector<int> spf(n + 1, 0), primes;
    for (int i = 2; i <= n; ++i) {
        if (spf[i] == 0) {
            spf[i] = i;
            primes.push_back(i);
        }
        for (int p : primes) {
            if (p > spf[i] || p > n / i) break;
            spf[p * i] = p;
        }
    }
    return spf;
}
```

时间 O(n)，空间 O(n)。它适合后续大量质因数分解；在 Python 中，线性时间并不保证比其他筛法更快，解释器循环常数也需要考虑。

## 7. 阶乘与逆阶乘求组合数

当 `p` 为质数、预处理上界 `N<p` 时：

$$
\binom nk=\operatorname{fact}[n]\operatorname{invfact}[k]
\operatorname{invfact}[n-k]\pmod p.
$$

```python
def combination_table(n, p):
    # p 为质数，0 <= n < p
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % p
    invfact = [1] * (n + 1)
    invfact[n] = pow(fact[n], p - 2, p)
    for i in range(n, 0, -1):
        invfact[i - 1] = invfact[i] * i % p

    def choose(a, b):
        if b < 0 or b > a:
            return 0
        return fact[a] * invfact[b] % p * invfact[a - b] % p
    return choose
```

预处理 O(N+log p)，单次查询 O(1)，空间 O(N)，要求查询 `0<=a<=N`。`N>=p` 时阶乘可能为 0，逆阶乘模板失效，需根据题意考虑 Lucas 等方法。Lucas 只适用于质数模数，且模数很大时不能直接预处理到 `p-1`。

## 8. 离散化

坐标很大但只有少量实际取值时，排序去重并映射为排名。

```python
a = [1000000000, -7, 42, -7]
values = sorted(set(a))
rank = {x: i for i, x in enumerate(values)}
compressed = [rank[x] for x in a]  # [2, 0, 1, 0]
```

时间 O(n log n)，空间 O(n)。离散化保留大小和相等关系，不保留距离。原坐标区间的真实长度要用 `values[r]-values[l]`，不能用排名差替代。接入 1-based 树状数组时，排名加 1。

## 9. 子集枚举与整除向上取整

枚举非负掩码 `mask` 的全部子集：

```python
sub = mask
while True:
    # 处理 sub，包含空集 0
    if sub == 0:
        break
    sub = (sub - 1) & mask
```

单个掩码有 k 个 1 时枚举 O(2^k) 个子集；枚举 n 位所有掩码的所有子集，总数为 O(3^n)。不要忽略外层枚举的成本。

对 `a>=0,b>0`，`ceil(a/b)=(a+b-1)//b`。C++ 的 `a+b-1` 可能溢出，可写 `a/b + (a%b != 0)`。Python 对任意整数 `a`、正整数 `b` 可写 `-(-a//b)`；负数情况下不能直接照搬正数模板。

## 参考资料

- [Python 内置函数：pow](https://docs.python.org/3/library/functions.html#pow)
- [Python 标准库：math](https://docs.python.org/3/library/math.html)
- [C++ 标准草案：无符号整数运算](https://eel.is/c++draft/basic.fundamental)
