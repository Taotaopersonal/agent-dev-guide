# 前沿方向 · 进阶篇

::: tip 学习目标
- 理解 Computer Use 的实现原理：截图-分析-操作循环
- 掌握 Anthropic Computer Use API 的使用方式
- 理解 Code Agent 的架构和核心工具设计
- 实现一个简单的测试驱动 Code Agent
:::

::: info 学完你能做到
- 用 Anthropic API 实现一个能操作电脑的 Agent
- 设计 Code Agent 的搜索、读取、编辑、执行四类工具
- 理解 Claude Code "精确替换而非重写" 的设计思想
- 构建一个能自动修 Bug 并运行测试的 Code Agent
:::

## Computer Use 实现原理

### 核心循环：截图-分析-操作

```python
"""computer_use_loop.py — Computer Use 核心循环"""

import anthropic
import base64
import subprocess
import json
import time

client = anthropic.Anthropic()

def take_screenshot() -> str:
    """截取屏幕截图，返回 base64 编码"""
    path = "/tmp/screenshot.png"
    subprocess.run(["screencapture", "-x", path], check=True)
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()

def execute_action(action: dict) -> bool:
    """执行 AI 返回的操作指令"""
    action_type = action.get("type")

    if action_type == "click":
        x, y = action["x"], action["y"]
        subprocess.run(["cliclick", f"c:{x},{y}"])

    elif action_type == "type":
        subprocess.run(["cliclick", f"t:{action['text']}"])

    elif action_type == "key":
        subprocess.run(["cliclick", f"kp:{action['key']}"])

    elif action_type == "done":
        return True  # 任务完成

    return False

def computer_use_loop(task: str, max_steps: int = 20):
    """Computer Use 主循环"""
    messages = [{"role": "user", "content": task}]

    for step in range(max_steps):
        screenshot_b64 = take_screenshot()

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64,
                    },
                },
                {"type": "text", "text": "当前屏幕截图，请决定下一步操作。"},
            ],
        })

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="""你是电脑操作助手。返回 JSON 格式的操作指令。
可用操作：
- {"type": "click", "x": 100, "y": 200}
- {"type": "type", "text": "hello"}
- {"type": "key", "key": "enter"}
- {"type": "done"}
只返回一个 JSON 对象。""",
            messages=messages,
        )

        try:
            action = json.loads(response.content[0].text)
            if execute_action(action):
                return "任务完成"
        except json.JSONDecodeError:
            pass

        time.sleep(0.5)

    return "达到最大步数"
```

### Anthropic 官方 Computer Use API

Anthropic 把鼠标、键盘、截图操作封装成了标准的 Tool，模型可以直接调用：

```python
"""anthropic_computer_use.py — 官方 Computer Use API"""

import anthropic

client = anthropic.Anthropic()

# 三个内置工具
COMPUTER_USE_TOOLS = [
    {
        "type": "computer_20250124",  # 电脑操控
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080,
        "display_number": 0,
    },
    {
        "type": "bash_20250124",      # 终端命令
        "name": "bash",
    },
    {
        "type": "text_editor_20250124",  # 文本编辑
        "name": "str_replace_editor",
    },
]

def execute_computer_tool(tool_name: str, tool_input: dict) -> str:
    """执行 Computer Use 工具调用

    需根据 tool_name 分发到对应的处理逻辑：
    - "computer": 执行鼠标/键盘操作并返回截图
    - "bash": 执行终端命令并返回输出
    - "str_replace_editor": 执行文件编辑操作
    """
    # 这里需要实现具体的工具执行逻辑
    # 参考 Anthropic Computer Use Demo: https://github.com/anthropics/anthropic-quickstarts
    raise NotImplementedError(f"请实现 {tool_name} 的执行逻辑")

def run_computer_use_task(task: str, max_steps: int = 30):
    """使用官方 API 执行 Computer Use 任务"""
    messages = [{"role": "user", "content": task}]

    for step in range(max_steps):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=COMPUTER_USE_TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(
                b.text for b in response.content if hasattr(b, "text")
            )

        # 处理工具调用
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                result = execute_computer_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

    return "达到最大步数"
```

::: warning 安全提醒
Computer Use 让 AI 直接控制你的电脑，务必在 Docker 沙箱中运行：

```yaml
services:
  computer-use:
    image: ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
    ports:
      - "8501:8501"
      - "6080:6080"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```
:::

## Code Agent 架构与工具设计

### 四类核心工具

```python
"""code_tools.py — Code Agent 的四类核心工具"""

import os
import subprocess
import glob

def grep_search(pattern: str, path: str = ".", file_type: str = "") -> str:
    """在代码库中搜索文本模式"""
    cmd = ["rg", "--no-heading", "-n", "--max-count", "50"]
    if file_type:
        cmd.extend(["--type", file_type])
    cmd.extend([pattern, path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout[:5000] or f"未找到匹配 '{pattern}' 的内容"
    except subprocess.TimeoutExpired:
        return "搜索超时"

def read_file(file_path: str) -> str:
    """读取文件内容（带行号）"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        numbered = [f"{i+1}\t{line}" for i, line in enumerate(lines)]
        content = "".join(numbered)
        if len(content) > 10000:
            return content[:10000] + f"\n... (截断，共 {len(lines)} 行)"
        return content
    except FileNotFoundError:
        return f"文件不存在: {file_path}"

def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """精确替换文件中的代码段

    关键设计：替换而非重写。
    必须精确匹配 old_string，如果匹配到多处则报错。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if old_string not in content:
            return "错误：未找到要替换的文本。请确认精确匹配。"

        count = content.count(old_string)
        if count > 1:
            return f"错误：找到 {count} 处匹配。请提供更多上下文。"

        new_content = content.replace(old_string, new_string, 1)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return f"文件已更新: {file_path}"
    except Exception as e:
        return f"编辑失败: {e}"

def run_command(command: str) -> str:
    """执行终端命令"""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=30,
        )
        output = result.stdout[:5000]
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr[:2000]}"
        output += f"\n[退出码: {result.returncode}]"
        return output
    except subprocess.TimeoutExpired:
        return "命令执行超时（30秒限制）"
```

### "编辑而非重写"的设计哲学

为什么 `edit_file` 用字符串替换而不是重写整个文件？

```python
# 方案A：重写整个文件 —— 危险
# AI 需要输出完整文件内容，可能遗漏其他代码、引入格式错误
def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)  # 100 行的文件全部重写

# 方案B：精确替换 —— 安全
# AI 只需要指定要改的那几行，其余代码保持不变
def edit_file(path, old_string, new_string):
    content = open(path).read()
    content.replace(old_string, new_string, 1)  # 只改需要改的
```

方案 B 的优势：改动范围小、不容易引入新 Bug、Token 消耗少。

### 测试驱动的 Code Agent

```python
"""tdd_agent.py — 测试驱动的 Code Agent"""

import asyncio
import anthropic

async def test_driven_code_agent(task: str, project_path: str):
    """工作流程：理解 → 修改 → 测试 → 如失败则重试"""

    system_prompt = f"""你是专业的 Code Agent，遵循测试驱动开发。

工作目录: {project_path}

工作流程：
1. 用 grep_search 搜索相关代码
2. 用 read_file 阅读代码，理解结构
3. 如果没有测试，先写测试
4. 用 edit_file 修改代码（精确替换，不要重写）
5. 用 run_command 运行 pytest 验证
6. 测试失败则分析原因并修复

编辑原则：
- 先 read_file 确认要修改的内容
- 使用 edit_file 做精确替换
- 保持代码风格一致"""

    client = anthropic.AsyncAnthropic()
    tools = [
        {"name": "grep_search", "description": "搜索代码模式",
         "input_schema": {"type": "object", "properties": {
             "pattern": {"type": "string"},
             "path": {"type": "string", "default": "."},
         }, "required": ["pattern"]}},
        {"name": "read_file", "description": "读取文件",
         "input_schema": {"type": "object", "properties": {
             "file_path": {"type": "string"},
         }, "required": ["file_path"]}},
        {"name": "edit_file", "description": "编辑文件",
         "input_schema": {"type": "object", "properties": {
             "file_path": {"type": "string"},
             "old_string": {"type": "string"},
             "new_string": {"type": "string"},
         }, "required": ["file_path", "old_string", "new_string"]}},
        {"name": "run_command", "description": "执行命令",
         "input_schema": {"type": "object", "properties": {
             "command": {"type": "string"},
         }, "required": ["command"]}},
    ]

    messages = [{"role": "user", "content": task}]
    tool_handlers = {
        "grep_search": grep_search,
        "read_file": read_file,
        "edit_file": edit_file,
        "run_command": run_command,
    }

    for _ in range(30):
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if hasattr(b, "text"))

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                handler = tool_handlers[block.name]
                result = handler(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

    return "达到最大迭代次数"
```

## Computer Use 的应用场景

### UI 自动化测试

```python
# 用自然语言描述测试步骤，不需要写 CSS selector
test_cases = [
    "打开登录页面，输入用户名 testuser，密码 Test123!，"
    "点击登录按钮，验证是否跳转到首页",

    "不输入任何内容直接点击提交按钮，"
    "检查是否显示了必填字段的错误提示",
]

for case in test_cases:
    result = run_computer_use_task(f"执行 UI 测试：{case}")
```

### RPA：操作没有 API 的系统

```python
# 跨系统数据录入
run_computer_use_task("""
1. 打开桌面上的 data.xlsx 文件
2. 读取 A 列到 D 列的数据
3. 打开 Chrome，访问 http://erp.internal.com
4. 登录并进入数据录入页面
5. 把 Excel 数据逐行填入表单并提交
""")
```

## 小结

两大前沿方向的实现要点：

1. **Computer Use**：截图-分析-操作循环，Anthropic 提供 computer/bash/text_editor 三个内置工具，务必在沙箱中运行
2. **Code Agent**：搜索-读取-编辑-执行四类工具，"精确替换而非重写" 是核心设计原则，测试驱动确保修改质量

## 练习

1. 用 Anthropic Computer Use Docker Demo 完成一个任务：打开浏览器搜索 "Python tutorial"
2. 实现完整的 `edit_file` 工具，测试：正常替换、未找到匹配、多处匹配的三种情况
3. 写一个简单的 Code Agent，给它一个有 Bug 的 Python 文件和测试文件，让它自动修复

## 参考资源

- [Anthropic Computer Use 文档](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) -- 官方 API 文档
- [Anthropic Computer Use Demo](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo) -- Docker 示例
- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) -- CLI 编程工具
- [SWE-bench](https://www.swebench.com/) -- Code Agent 基准测试
- [OpenHands](https://github.com/All-Hands-AI/OpenHands) -- 开源 Code Agent 平台
- [Aider](https://github.com/paul-gauthier/aider) -- AI 结对编程工具
