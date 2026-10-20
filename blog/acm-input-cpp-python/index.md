---
title: "ACM 输入处理：C++ / Python"
date: 2026-10-20T12:00:00+08:00
draft: false
description: "按输入格式整理 C++ 和 Python 的 ACM 模板，区分 token、整行、多组数据和 EOF"
slug: "acm-input-cpp-python"
tags: ["cpp", "python", "acm", "输入输出"]
categories: ["笔试重点"]
comments: true
math: true
---

# ACM 输入处理：C++ / Python

ACM 模式需要自己读输入、调用算法并输出结果。读题时先确定三件事：有几组数据，空白是否有含义，一组数据在哪里结束。不要根据样例的换行猜输入格式。

以下各代码块是独立示例。C++ 使用 C++17；Python 使用 Python 3。

## 1. 按 token 读取与按行读取

`cin >> x` 和 `split()` 都按空白切分，连续空格、制表符和换行通常没有区别。姓名、句子、空行等内容需要保留行边界。

| 输入特点 | C++ | Python |
|----------|-----|--------|
| 整数、单词，空白只作分隔 | `cin >> x` | `read().split()` 或逐行 `split()` |
| 每行是一个独立记录 | `getline(cin, line)` | `readline()` |
| 字符串可能包含空格 | `getline` | 去除行末换行，保留其他空白 |
| 读到文件结束 | 检查读取表达式是否成功 | 遍历文件对象或检查空字符串 |

## 2. 单组数据：n 后面跟 n 个数

输入中的数组可以分布在多行。按 token 读取时不必关心换行位置。

```text
5
1 2 3
4 5
```

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    long long total = 0;
    for (auto& x : a) {
        cin >> x;
        total += x;
    }
    cout << total << '\n';
}
```

```python
import sys

tokens = iter(sys.stdin.buffer.read().split())
n = int(next(tokens))
a = [int(next(tokens)) for _ in range(n)]
print(sum(a))
```

Python 这里假定输入非空且符合题意。整体读取便于跨行取数，但会同时保存原始输入和切分后的对象；大输入且内存紧张时改用逐行处理。不要在每轮循环重新执行一次 `read()`。

## 3. 首行给定 T 组数据

每组先读 `n`，再读 `n` 个数。数组、答案和计数器应在每组内部重新初始化。

```cpp
#include <iostream>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        long long total = 0, x;
        for (int i = 0; i < n; ++i) {
            cin >> x;
            total += x;
        }
        cout << total << '\n';
    }
}
```

```python
import sys

it = iter(sys.stdin.buffer.read().split())
t = int(next(it))
answers = []
for _ in range(t):
    n = int(next(it))
    total = sum(int(next(it)) for _ in range(n))
    answers.append(str(total))
sys.stdout.write("\n".join(answers))
```

## 4. 读到 EOF：每组两个整数

EOF 是文件结束。C++ 直接检查读取是否成功，不要写 `while (!cin.eof())`，因为 EOF 标志通常在读取失败后才设置。

```cpp
#include <iostream>
using namespace std;

int main() {
    long long a, b;
    while (cin >> a >> b) cout << a + b << '\n';
}
```

如果题目只要求每组两个数，不保证每组恰好一行，Python 可以按 token 流读取：

```python
import sys

def integers():
    for line in sys.stdin.buffer:
        for token in line.split():
            yield int(token)

it = iter(integers())
for a in it:
    b = next(it)  # 题目保证每组两个数
    print(a + b)
```

若题目明确每行两个数，使用 `for line in sys.stdin.buffer` 后逐行解包即可。是否跳过空行由题目决定；空行也可能表示一条空记录。

## 5. 哨兵结束：遇到 0 0 停止

先成功读取，再检查结束标志。结束标志不参与计算。

```cpp
#include <iostream>
using namespace std;

int main() {
    long long a, b;
    while (cin >> a >> b) {
        if (a == 0 && b == 0) break;
        cout << a + b << '\n';
    }
}
```

```python
import sys

for line in sys.stdin.buffer:
    if not line.strip():
        continue  # 本例约定空行不表示数据
    a, b = map(int, line.split())
    if a == 0 and b == 0:
        break
    print(a + b)
```

## 6. 每行数量不定

题目要求对每行分别求和时，换行就是一组数据的边界，不能把整个文件 `split()` 后丢掉行信息。

```cpp
#include <iostream>
#include <sstream>
#include <string>
using namespace std;

int main() {
    string line;
    while (getline(cin, line)) {
        istringstream in(line);
        long long x, total = 0;
        while (in >> x) total += x;
        cout << total << '\n'; // 本例中空行的和为 0
    }
}
```

```python
import sys

for line in sys.stdin.buffer:
    print(sum(map(int, line.split())))
```

## 7. 整行字符串与混合输入

`cin >> n` 会留下后面的换行。若题目规定 `n` 独占一行，先丢弃该行剩余内容，再读后面的完整行：

```cpp
#include <iostream>
#include <limits>
#include <string>
using namespace std;

int main() {
    int n;
    cin >> n;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    string line;
    for (int i = 0; i < n; ++i) {
        getline(cin, line);
        cout << line << '\n';
    }
}
```

`getline(cin >> ws, line)` 会跳过所有前导空白，包括空行和行首空格。只有这些内容确实可以忽略时才使用。

```python
import sys

n = int(sys.stdin.readline())
for _ in range(n):
    line = sys.stdin.readline().removesuffix("\n").removesuffix("\r")
    print(line)
```

Python 的 `strip()` 会同时删除行首、行尾的空白；只去除换行时不要直接 `strip()`。`readline()` 返回 `""` 表示 EOF，返回 `"\n"` 表示读到一个空行。

## 8. 矩阵、字符网格与逗号分隔

数字矩阵只需要按 `n * m` 个 token 读取。无空格的字符网格按字符串读取，再用下标访问。

```python
import sys

read = sys.stdin.readline
n, m = map(int, read().split())
matrix = [list(map(int, read().split())) for _ in range(n)]
# 本例要求每行恰好 m 个数字
```

字符网格常写成 `grid = [read().rstrip("\r\n") for _ in range(n)]`。C++ 可用 `vector<string> grid(n); for (auto& row : grid) cin >> row;`，前提是网格中没有空白字符。

简单逗号分隔数字可用 `line.split(",")`；C++ 使用 `getline(stream, field, ',')`。带引号、转义逗号的完整 CSV 需要 CSV 解析规则，不能直接按逗号切分。

## 9. 输出与提交前检查

Python 批量输出用 `"\n".join(...)`，元素需要先转为字符串；矩阵一行用 `print(*row)`。C++ 输出换行通常用 `'\n'`，`endl` 还会刷新缓冲区。

关闭 C++ iostream 同步后，避免混用 `cin/cout` 和 `scanf/printf`。交互题需要按协议主动刷新输出；普通 ACM 输入模板不能直接照搬到交互题。

- 数据组数按题意读取，EOF 和哨兵条件分别处理。
- 求和、乘法和距离使用足够宽的整数；C++ 的 `1LL * a * b` 在乘法前提升类型。
- 不额外输出提示语、调试信息或多余空格。
- 字符串的空格、空行、大小写是否保留，应由题目规定。

## 参考

- [Python：sys.stdin 与标准流](https://docs.python.org/3/library/sys.html#sys.stdin)
- [Python：str.split](https://docs.python.org/3/library/stdtypes.html#str.split)
- [C++ 标准草案：字符串输入输出](https://eel.is/c++draft/string.io)
