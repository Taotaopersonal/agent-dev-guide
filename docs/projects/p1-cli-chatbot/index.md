# P1: 命令行聊天机器人

::: info 项目信息
**难度**: 入门 | **代码量**: ~200 行 | **预计时间**: 3-4 小时
**对应章节**: 初级篇第 1-4 章（Python 基础、LLM 原理、Prompt Engineering、LLM API 实战）
:::

## 项目目标

构建一个功能完备的命令行聊天机器人，支持多轮对话、流式输出、角色切换、对话历史保存。这是你的第一个 AI 应用，虽然是命令行界面，但核心交互逻辑和生产级聊天产品完全一致。

### 功能清单

- [x] 基础单轮问答
- [x] 多轮对话（自动管理消息历史）
- [x] Streaming 流式逐字输出
- [x] System Prompt 角色切换（翻译官、代码助手、写作助手等）
- [x] 对话历史保存与加载（JSON 格式）
- [x] 命令系统（`/help`、`/clear`、`/save`、`/load`、`/role`、`/exit`）
- [x] Token 用量统计
- [x] 终端美化输出（语法高亮、Markdown 渲染）

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| LLM SDK | `anthropic` | Anthropic 官方 SDK，API 设计清晰 |
| 终端美化 | `rich` | Python 最强终端 UI 库，支持 Markdown 渲染 |
| 环境变量 | `python-dotenv` | 安全管理 API Key |
| 数据存储 | JSON 文件 | 简单可靠，无需数据库 |

## 项目结构

```
cli-chatbot/
├── chatbot.py          # 主程序（全部逻辑）
├── roles.py            # 预定义角色
├── .env                # API Key（不提交到 Git）
├── .gitignore
└── history/            # 对话历史存储目录
    └── chat_2026-04-16_143022.json
```

## 架构设计

```mermaid
graph LR
    A[用户输入] --> B{是命令?}
    B -->|是| C[命令处理器]
    C --> D[/help /clear /save /load /role /exit]
    B -->|否| E[消息管理器]
    E --> F[添加到历史]
    F --> G[调用 Claude API]
    G --> H[流式输出]
    H --> I[追加 AI 回复到历史]
    I --> A
```

## 分步实现

### 第 1 步：基础单轮对话

先搭建最简骨架，能发消息、收回复就行：

```python
# chatbot.py - Step 1: 基础单轮对话
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

def chat(user_message: str) -> str:
    """发送单条消息并获取回复"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text

if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ("exit", "quit"):
            break
        reply = chat(user_input)
        print(f"\nAssistant: {reply}")
```

::: tip 验证点
运行后输入 "你好"，应该收到 Claude 的回复。但此时没有上下文记忆——每次提问都是全新的对话。
:::

### 第 2 步：多轮对话

加入消息历史管理，让 AI 能"记住"之前的对话：

```python
# Step 2: 多轮对话 - 消息历史管理
class ChatSession:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.messages: list[dict] = []
        self.model = "claude-sonnet-4-20250514"
        self.max_tokens = 2048

    def chat(self, user_message: str) -> str:
        # 添加用户消息到历史
        self.messages.append({"role": "user", "content": user_message})

        # 发送完整历史给 API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self.messages,
        )

        assistant_message = response.content[0].text
        # 添加 AI 回复到历史
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def clear(self):
        """清空对话历史"""
        self.messages = []
```

### 第 3 步：Streaming 流式输出

让回复逐字显示，而不是等全部生成完才输出：

```python
# Step 3: 流式输出
def chat_stream(self, user_message: str) -> str:
    self.messages.append({"role": "user", "content": user_message})

    full_response = ""
    with self.client.messages.stream(
        model=self.model,
        max_tokens=self.max_tokens,
        system=self.system_prompt,
        messages=self.messages,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    print()  # 换行
    self.messages.append({"role": "assistant", "content": full_response})
    return full_response
```

### 第 4 步：System Prompt 角色切换

定义多个预设角色，支持动态切换：

```python
# roles.py - 预定义角色
ROLES = {
    "default": {
        "name": "通用助手",
        "prompt": "你是一个友好、专业的 AI 助手。回复简洁有用。",
    },
    "translator": {
        "name": "翻译官",
        "prompt": (
            "你是一位精通中英日三语的翻译专家。"
            "用户发中文你翻译成英文，发英文你翻译成中文。"
            "翻译要自然流畅，不要逐字直译。"
        ),
    },
    "coder": {
        "name": "代码助手",
        "prompt": (
            "你是一位资深的全栈开发工程师。"
            "回答编程问题时给出完整可运行的代码，并解释关键设计决策。"
            "代码遵循最佳实践，包含必要的错误处理。"
        ),
    },
    "writer": {
        "name": "写作助手",
        "prompt": (
            "你是一位专业的中文写作教练。"
            "帮助用户改进文章结构、润色文字、修正语法。"
            "给出修改建议时解释原因。"
        ),
    },
}
```

### 第 5 步：对话历史保存/加载

```python
# Step 5: 历史持久化
import json
from datetime import datetime
from pathlib import Path

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)

def save_history(self, filename: str | None = None) -> str:
    if not filename:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"chat_{timestamp}.json"

    filepath = HISTORY_DIR / filename
    data = {
        "model": self.model,
        "role": self.current_role,
        "system_prompt": self.system_prompt,
        "messages": self.messages,
        "saved_at": datetime.now().isoformat(),
    }
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return str(filepath)

def load_history(self, filename: str) -> bool:
    filepath = HISTORY_DIR / filename
    if not filepath.exists():
        return False

    data = json.loads(filepath.read_text())
    self.messages = data["messages"]
    self.current_role = data.get("role", "default")
    self.system_prompt = data.get("system_prompt", "")
    return True
```

### 第 6 步：命令系统

```python
# Step 6: 命令处理
COMMANDS = {
    "/help":  "显示帮助信息",
    "/clear": "清空当前对话历史",
    "/save":  "保存对话历史到文件",
    "/load":  "加载历史对话 (用法: /load <文件名>)",
    "/role":  "切换角色 (用法: /role <角色名>)",
    "/roles": "列出所有可用角色",
    "/tokens":"显示当前 Token 用量统计",
    "/exit":  "退出程序",
}

def handle_command(self, command: str) -> bool:
    """处理命令。返回 True 表示继续，False 表示退出。"""
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/help":
        for c, desc in COMMANDS.items():
            self.console.print(f"  [bold cyan]{c:<10}[/] {desc}")

    elif cmd == "/clear":
        self.clear()
        self.console.print("[green]对话历史已清空。[/]")

    elif cmd == "/save":
        path = self.save_history(arg if arg else None)
        self.console.print(f"[green]对话已保存到: {path}[/]")

    elif cmd == "/load":
        if not arg:
            # 列出可用文件
            files = sorted(HISTORY_DIR.glob("*.json"))
            for f in files:
                self.console.print(f"  {f.name}")
            return True
        if self.load_history(arg):
            self.console.print(f"[green]已加载: {arg}[/]")
        else:
            self.console.print(f"[red]文件不存在: {arg}[/]")

    elif cmd == "/role":
        if not arg:
            self.console.print(f"当前角色: [bold]{self.current_role}[/]")
            return True
        if arg in ROLES:
            self.set_role(arg)
            self.console.print(f"[green]已切换到: {ROLES[arg]['name']}[/]")
        else:
            self.console.print(f"[red]未知角色: {arg}[/]")

    elif cmd == "/roles":
        for key, role in ROLES.items():
            marker = " <-- 当前" if key == self.current_role else ""
            self.console.print(f"  [cyan]{key:<12}[/] {role['name']}{marker}")

    elif cmd == "/tokens":
        self.show_token_stats()

    elif cmd == "/exit":
        return False

    return True
```

## 完整源码

将以上所有步骤整合为完整的可运行程序：

```python
#!/usr/bin/env python3
"""CLI Chatbot - 命令行聊天机器人
一个功能完备的终端聊天应用，支持多轮对话、流式输出、角色切换。
"""

import json
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

load_dotenv()

# ============================================================
# 角色定义
# ============================================================
ROLES = {
    "default": {
        "name": "通用助手",
        "prompt": "你是一个友好、专业的 AI 助手。回复简洁、有用、结构清晰。",
    },
    "translator": {
        "name": "翻译官",
        "prompt": "你是一位精通中英日三语的翻译专家。用户发中文你翻译成英文，发英文你翻译成中文。翻译要自然流畅。",
    },
    "coder": {
        "name": "代码助手",
        "prompt": "你是一位资深全栈工程师。给出完整可运行代码，解释关键设计。遵循最佳实践，包含错误处理。",
    },
    "writer": {
        "name": "写作助手",
        "prompt": "你是一位专业的中文写作教练。帮助改进文章结构、润色文字、修正语法，并解释修改原因。",
    },
}

# ============================================================
# 命令定义
# ============================================================
COMMANDS = {
    "/help": "显示帮助信息",
    "/clear": "清空当前对话历史",
    "/save": "保存对话历史 (用法: /save [文件名])",
    "/load": "加载对话历史 (用法: /load <文件名>)",
    "/role": "切换角色 (用法: /role <角色名>)",
    "/roles": "列出所有可用角色",
    "/tokens": "显示 Token 用量统计",
    "/exit": "退出程序",
}

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)


class ChatBot:
    """命令行聊天机器人核心类"""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.console = Console(theme=Theme({"user": "bold green", "ai": "bold blue"}))
        self.model = model
        self.max_tokens = 2048
        self.messages: list[dict] = []
        self.current_role = "default"
        self.system_prompt = ROLES["default"]["prompt"]
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # ----------------------------------------------------------
    # 核心对话
    # ----------------------------------------------------------
    def chat_stream(self, user_message: str) -> str:
        """发送消息并流式接收回复"""
        self.messages.append({"role": "user", "content": user_message})

        full_response = ""
        input_tokens = 0
        output_tokens = 0

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=self.messages,
        ) as stream:
            for text in stream.text_stream:
                self.console.print(text, end="", highlight=False)
                full_response += text

            # 获取最终的 usage 信息
            final_message = stream.get_final_message()
            input_tokens = final_message.usage.input_tokens
            output_tokens = final_message.usage.output_tokens

        self.console.print()  # 换行

        self.messages.append({"role": "assistant", "content": full_response})
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return full_response

    # ----------------------------------------------------------
    # 角色管理
    # ----------------------------------------------------------
    def set_role(self, role_key: str):
        if role_key in ROLES:
            self.current_role = role_key
            self.system_prompt = ROLES[role_key]["prompt"]
            self.clear()  # 切换角色时清空历史

    # ----------------------------------------------------------
    # 历史管理
    # ----------------------------------------------------------
    def clear(self):
        self.messages = []

    def save_history(self, filename: str | None = None) -> str:
        if not filename:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"chat_{timestamp}.json"
        if not filename.endswith(".json"):
            filename += ".json"

        filepath = HISTORY_DIR / filename
        data = {
            "model": self.model,
            "role": self.current_role,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "token_usage": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
            },
            "saved_at": datetime.now().isoformat(),
        }
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return str(filepath)

    def load_history(self, filename: str) -> bool:
        if not filename.endswith(".json"):
            filename += ".json"
        filepath = HISTORY_DIR / filename
        if not filepath.exists():
            return False
        data = json.loads(filepath.read_text())
        self.messages = data["messages"]
        self.current_role = data.get("role", "default")
        self.system_prompt = data.get("system_prompt", ROLES["default"]["prompt"])
        return True

    # ----------------------------------------------------------
    # Token 统计
    # ----------------------------------------------------------
    def show_token_stats(self):
        self.console.print(
            Panel(
                f"模型: {self.model}\n"
                f"对话轮次: {len(self.messages) // 2}\n"
                f"输入 Tokens: {self.total_input_tokens:,}\n"
                f"输出 Tokens: {self.total_output_tokens:,}\n"
                f"总计 Tokens: {self.total_input_tokens + self.total_output_tokens:,}",
                title="Token 用量统计",
                border_style="cyan",
            )
        )

    # ----------------------------------------------------------
    # 命令处理
    # ----------------------------------------------------------
    def handle_command(self, command: str) -> bool:
        """处理斜杠命令。返回 False 表示退出程序。"""
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            for c, desc in COMMANDS.items():
                self.console.print(f"  [bold cyan]{c:<10}[/] {desc}")

        elif cmd == "/clear":
            self.clear()
            self.console.print("[green]对话历史已清空。[/]")

        elif cmd == "/save":
            path = self.save_history(arg if arg else None)
            self.console.print(f"[green]已保存到: {path}[/]")

        elif cmd == "/load":
            if not arg:
                files = sorted(HISTORY_DIR.glob("*.json"))
                if files:
                    for f in files:
                        self.console.print(f"  {f.name}")
                else:
                    self.console.print("  [dim]暂无保存的对话[/]")
                return True
            if self.load_history(arg):
                self.console.print(f"[green]已加载: {arg} ({len(self.messages)} 条消息)[/]")
            else:
                self.console.print(f"[red]文件不存在: {arg}[/]")

        elif cmd == "/role":
            if not arg:
                name = ROLES[self.current_role]["name"]
                self.console.print(f"当前角色: [bold]{name}[/] ({self.current_role})")
                return True
            if arg in ROLES:
                self.set_role(arg)
                self.console.print(f"[green]已切换到: {ROLES[arg]['name']}[/]")
            else:
                self.console.print(f"[red]未知角色: {arg}。输入 /roles 查看可用角色[/]")

        elif cmd == "/roles":
            for key, role in ROLES.items():
                marker = " [bold yellow]<-- 当前[/]" if key == self.current_role else ""
                self.console.print(f"  [cyan]{key:<12}[/] {role['name']}{marker}")

        elif cmd == "/tokens":
            self.show_token_stats()

        elif cmd == "/exit":
            return False

        else:
            self.console.print(f"[red]未知命令: {cmd}。输入 /help 查看帮助[/]")

        return True

    # ----------------------------------------------------------
    # 主循环
    # ----------------------------------------------------------
    def run(self):
        """启动聊天主循环"""
        self.console.print(
            Panel(
                "[bold]CLI Chatbot[/] - 命令行聊天机器人\n"
                f"模型: {self.model} | 角色: {ROLES[self.current_role]['name']}\n"
                "输入 [bold cyan]/help[/] 查看命令列表，输入 [bold cyan]/exit[/] 退出",
                border_style="blue",
            )
        )

        while True:
            try:
                user_input = self.console.input("\n[user]You:[/] ").strip()

                if not user_input:
                    continue

                # 处理命令
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        self.console.print("[dim]再见！[/]")
                        break
                    continue

                # 正常对话
                self.console.print("\n[ai]Assistant:[/]")
                self.chat_stream(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[dim]使用 /exit 退出，或 /save 保存对话[/]")
            except anthropic.APIError as e:
                self.console.print(f"\n[red]API 错误: {e.message}[/]")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    bot = ChatBot()
    bot.run()
```

## 运行和测试

### 安装依赖

```bash
# 创建项目
mkdir cli-chatbot && cd cli-chatbot
uv init
uv add anthropic rich python-dotenv

# 配置 API Key
echo "ANTHROPIC_API_KEY=sk-ant-xxx" > .env
```

### 运行

```bash
uv run python chatbot.py
```

### 测试用例

| 测试场景 | 操作 | 预期结果 |
|---------|------|---------|
| 单轮对话 | 输入 "你好" | 收到 Claude 的问候回复 |
| 多轮记忆 | 先说 "我叫小明"，再问 "我叫什么" | 回复中包含 "小明" |
| 流式输出 | 输入一个长问题 | 文字逐字显示，不是一次性输出 |
| 角色切换 | `/role translator`，然后输入中文 | 回复为英文翻译 |
| 历史保存 | `/save test` | 生成 `history/test.json` |
| 历史加载 | `/load test` | 恢复之前的对话上下文 |
| 清空历史 | `/clear`，然后问 "我之前说了什么" | AI 说不知道之前的对话 |
| Token 统计 | `/tokens` | 显示累计的 Token 用量 |

## 扩展建议

完成基础版后，可以尝试以下扩展：

1. **自定义角色文件** -- 支持从 YAML/JSON 文件加载角色定义，用户可以自己创建角色
2. **Markdown 渲染** -- 使用 `rich.markdown.Markdown` 将 AI 回复渲染为格式化的 Markdown
3. **对话导出** -- 支持导出为 Markdown 或 HTML 格式，方便分享
4. **上下文窗口管理** -- 当消息历史超过模型上下文窗口时，自动截断或摘要化
5. **多模型支持** -- 通过命令切换不同的模型（Haiku/Sonnet/Opus）
6. **费用计算** -- 根据 Token 用量实时计算 API 费用

::: tip 进阶提示
这个项目的消息管理逻辑（追加 user/assistant 消息对）是所有 AI 应用的基石。在后续的 P2-P6 项目中，你会反复用到这个模式。确保你完全理解了 `messages` 列表的结构和作用。
:::

## 参考资源

- [Anthropic Messages API 文档](https://docs.anthropic.com/en/api/messages) -- 完整的 API 参数说明
- [Anthropic Streaming 文档](https://docs.anthropic.com/en/api/messages-streaming) -- 流式输出的详细说明
- [Rich 库文档](https://rich.readthedocs.io/) -- 终端美化库的完整功能
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) -- SDK 源码和示例
- [Prompt Engineering Guide (Anthropic)](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) -- System Prompt 最佳实践
