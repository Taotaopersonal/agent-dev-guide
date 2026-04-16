# Python 基础 · 初级

::: info 学习目标
- 理解 Agent 开发为什么首选 Python
- 完成 Python 环境搭建（Python 3.12+ 和 uv 包管理器）
- 通过 JS 对比掌握 Python 核心语法：变量、函数、数据结构
- 理解类、dataclass 和异常处理
- 学完能写 Python 脚本，读懂 Agent 相关的 Python 代码

预计学习时间：3-4 小时
:::

## 为什么选 Python

如果你是前端工程师，可能会想："我用 Node.js 不行吗？"

行，但不是最优选择。在 AI Agent 开发领域，Python 拥有压倒性的生态优势：

| 领域 | Python 库 | Node.js 替代方案 | 差距 |
|------|----------|-----------------|------|
| Agent 框架 | CrewAI, AutoGen, LangGraph | 几乎没有成熟选择 | 断层级 |
| LLM 框架 | LangChain, LlamaIndex | LangChain.js (功能子集) | Python 版多 30%+ |
| LLM 官方 SDK | openai, anthropic | 有对应 SDK | 基本持平 |
| 数据处理 | pandas, numpy | 没有对等方案 | 不可替代 |
| 向量计算 | sentence-transformers | 没有对等方案 | 不可替代 |

::: tip 关键事实
当你在 GitHub 上搜索一个新的 Agent 论文的参考实现时，99% 的概率它是 Python 写的。当你在 AI 社区看到新的最佳实践，代码示例几乎一定是 Python。
:::

## 环境搭建

### 安装 Python

macOS 用户推荐 Homebrew：

```bash
# 安装 Python 3.12+
brew install python@3.12

# 验证安装
python3 --version
# Python 3.12.x
```

Windows 用户从 [python.org](https://www.python.org/downloads/) 下载安装包，安装时勾选 "Add Python to PATH"。

### 安装 uv（推荐的包管理器）

uv 是 Rust 编写的 Python 包管理器，速度比 pip 快 10-100 倍，功能类似 npm：

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 验证安装
uv --version
```

::: tip 为什么推荐 uv 而不是 pip
uv 之于 Python，就像 pnpm 之于 Node.js -- 更快、更可靠、内置虚拟环境管理。它已经成为 Python 社区的新标准。
:::

### 创建第一个项目

```bash
# 创建项目目录
mkdir my-first-agent && cd my-first-agent

# 用 uv 初始化项目（类似 npm init）
uv init

# 查看生成的文件
ls -la
# pyproject.toml   <-- 类似 package.json
# .python-version  <-- 锁定 Python 版本
# hello.py         <-- 入口文件

# 安装依赖（类似 npm install xxx）
uv add anthropic

# 运行 Python 文件
uv run hello.py
```

### VS Code 配置

安装以下扩展：
- **Python** (Microsoft)：语法高亮、IntelliSense
- **Pylance** (Microsoft)：类型检查
- **Ruff** (Astral)：代码格式化和 Lint（类比 ESLint + Prettier）

在项目根目录创建 `.vscode/settings.json`：

```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true
    }
}
```

## JS 到 Python 语法速通

### 变量声明

```javascript
// JavaScript
let name = "Alice";        // 可变变量
const age = 25;            // 不可变变量
const { x, y } = point;   // 解构赋值
const [first, ...rest] = arr;
```

```python
# Python
name = "Alice"        # 所有变量都是"可变的"引用
age = 25              # 没有 const 关键字
MAX_RETRY = 3         # 约定：全大写 = 常量（但 Python 不强制）

# 解包赋值（类似解构）
x, y = point
first, *rest = arr    # * 收集剩余元素，类似 JS 的 ...
```

### 基本类型

```python
# Python 类型
type("hello")       # <class 'str'>
type(42)            # <class 'int'>   （整数和浮点数区分！）
type(3.14)          # <class 'float'>
type(True)          # <class 'bool'>
type(None)          # <class 'NoneType'>  （None 是 Python 的 null）
type([])            # <class 'list'>

# 带类型注解的变量（可选但推荐）
name: str = "Alice"
age: int = 25
scores: list[float] = [98.5, 87.0]
config: dict[str, str] = {"key": "value"}
```

### 字符串

```python
name = "World"
print(f"Hello, {name}!")          # f-string（类似模板字符串）
"hello".upper()                   # "HELLO"
"hello world".split(" ")         # ["hello", "world"]
"  spaces  ".strip()             # "spaces"（strip 不是 trim）

# 多行字符串（常用于 Prompt 模板）
prompt = """你是一个有用的助手。
请根据以下信息回答问题：
{context}
"""
```

### 函数

```javascript
// JavaScript
function add(a, b) { return a + b; }
const multiply = (a, b) => a * b;
function greet(name = "World") { return `Hello, ${name}!`; }
```

```python
# Python
def add(a, b):
    return a + b

multiply = lambda a, b: a * b  # lambda 仅限单行

def greet(name="World"):
    return f"Hello, {name}!"

# 带类型注解的函数（Agent 开发中必用）
def search_web(query: str, max_results: int = 10) -> list[dict]:
    """搜索网页并返回结果列表。"""
    results = []
    # ... 搜索逻辑
    return results

# 可选返回值
def find_user(user_id: str) -> dict | None:
    """查找用户，未找到返回 None。"""
    return None
```

### *args 和 **kwargs

```python
# *args 收集位置参数，**kwargs 收集关键字参数
def log(level: str, *messages: str, **options):
    print(f"[{level}]", *messages)
    if options.get("timestamp"):
        print(f"Time: {options['timestamp']}")

log("INFO", "hello", "world", timestamp="2026-01-01")
```

## 数据结构

### List（对应 Array）

```python
arr = [1, 2, 3]
arr.append(4)                   # [1, 2, 3, 4]
len(arr)                        # 4（注意：是函数，不是属性）
3 in arr                        # True（用 in 操作符）
[x * 2 for x in arr]           # [2, 4, 6, 8]（列表推导式）
[x for x in arr if x > 2]     # [3, 4]（带条件过滤）
arr[1:3]                        # [2, 3]（切片语法）
```

### Dict（对应 Object）

```python
obj = {"name": "Alice", "age": 25}
obj["name"]                     # "Alice"
obj.get("name")                 # "Alice"（不存在时返回 None，更安全）
obj.get("email", "N/A")        # "N/A"（带默认值）
list(obj.keys())                # ["name", "age"]
list(obj.items())               # [("name", "Alice"), ("age", 25)]
"name" in obj                   # True

# 没有可选链，用链式 get
obj.get("address", {}).get("city")
```

### Tuple 和 Set

```python
# Tuple - 不可变的列表
point = (10, 20)
x, y = point               # 解包

# 函数返回多个值时，实际上就是返回 tuple
def get_position() -> tuple[int, int]:
    return (100, 200)

x, y = get_position()

# Set - 不重复的集合
tags = {"python", "agent", "ai"}
tags.add("llm")
"python" in tags            # True（查找比 list 快得多）

# 集合运算
a = {1, 2, 3}
b = {2, 3, 4}
a | b                       # {1, 2, 3, 4} 并集
a & b                       # {2, 3} 交集
```

## 类与 dataclass

### 基础类

```python
class Animal:
    def __init__(self, name: str):
        self.name = name           # self 类似 JS 的 this

    def speak(self) -> str:        # 方法必须显式写 self
        return f"{self.name} makes a sound"


class Dog(Animal):                 # 继承用括号
    def speak(self) -> str:
        return f"{self.name} barks"


dog = Dog("Buddy")
print(dog.speak())                 # "Buddy barks"
```

### dataclass（推荐）

Agent 开发中大量使用 dataclass 定义数据结构，它类比 TypeScript 的 interface：

```python
from dataclasses import dataclass, field

@dataclass
class Message:
    """对话消息 - 自动生成 __init__、__repr__、__eq__"""
    role: str
    content: str
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def is_user(self) -> bool:
        """计算属性（类似 JS 的 getter）"""
        return self.role == "user"

# 使用
msg = Message(role="user", content="你好")
print(msg)           # Message(role='user', content='你好', tool_calls=[])
print(msg.is_user)   # True
```

## 异常处理

```python
import json

try:
    data = json.loads(text)
except json.JSONDecodeError as e:    # 捕获特定异常
    print(f"JSON 解析失败: {e}")
except Exception as e:               # 捕获所有异常
    print(f"未知错误: {e}")
else:
    # 只在 try 成功时执行（JS 没有这个）
    print(f"解析成功，共 {len(data)} 个字段")
finally:
    cleanup()
```

## 列表推导式

Python 的特色语法，Agent 开发中非常常用：

```python
numbers = [1, 2, 3, 4, 5]

# 相当于 JS 的 numbers.map(n => n ** 2)
squares = [n ** 2 for n in numbers]           # [1, 4, 9, 16, 25]

# 相当于 JS 的 numbers.filter(n => n > 3)
big = [n for n in numbers if n > 3]           # [4, 5]

# 字典推导式
name_lengths = {name: len(name) for name in ["Alice", "Bob"]}
# {"Alice": 5, "Bob": 3}

# 生成器表达式（惰性计算，省内存）
total = sum(n ** 2 for n in range(1_000_000))
```

## 模块系统

```python
# 标准库导入
import json
from pathlib import Path

# 第三方包导入
from anthropic import Anthropic
from pydantic import BaseModel, Field

# 相对导入（在包内部使用）
from .utils import format_message
from ..config import settings

# Python 没有 export default
# 约定：以 _ 开头的名称视为私有
def public_func():
    pass

def _private_func():
    pass
```

## 小结

| 概念 | JavaScript | Python | Agent 开发中的用途 |
|------|-----------|--------|------------------|
| 变量 | let/const | 直接赋值 | 存储 API 响应、状态管理 |
| 函数 | function/=> | def | 工具函数、数据处理 |
| 类型 | TypeScript | 类型注解 + Pydantic | 工具定义、响应解析 |
| 数据结构 | Array/Object | list/dict/dataclass | 消息历史、工具结果 |
| 装饰器 | @decorator(实验) | @decorator(原生) | 路由、工具注册 |
| 模块 | import/export | import/from | 代码组织 |

## 练习

1. **环境搭建**：完成 Python + uv 安装，用 `uv init` 创建项目，安装 `anthropic` 依赖，用 `uv run hello.py` 运行一个打印 "Hello from Python!" 的脚本。

2. **数据处理**：写一个函数 `format_messages(messages: list[dict]) -> str`，把消息列表格式化为字符串，每条一行，格式为 `[role]: content`。

3. **列表推导式**：给定 `users = [{"name": "Alice", "score": 85}, {"name": "Bob", "score": 92}, {"name": "Carol", "score": 78}]`，用列表推导式筛选出 score > 80 的用户名列表。

4. **dataclass 练习**：用 dataclass 定义一个 `Tool` 类，包含 `name: str`、`description: str`、`parameters: dict` 三个字段，创建一个 "web_search" 工具实例并打印。

## 参考资源

- [Python 官方教程](https://docs.python.org/3/tutorial/)
- [uv 文档](https://docs.astral.sh/uv/)
- [Real Python - 面向初学者的教程](https://realpython.com/)
