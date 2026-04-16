# 记忆系统入门

> **学习目标**：理解 Agent 为什么需要记忆系统，掌握对话历史管理的基本策略，实现滑动窗口和 Token 截断，学会用 SQLite 做简单的持久化存储。

学完本节，你将能够：
- 解释 LLM 的"无状态"问题以及记忆系统的价值
- 实现滑动窗口消息管理
- 实现基于 Token 计数的精确截断
- 用 SQLite 持久化保存对话历史
- 理解 Tool Use 消息的特殊处理需求

## 为什么 Agent 需要记忆

你跟 ChatGPT 聊了一下午，说了自己的名字、工作、偏好。关掉窗口再回来——它完全不记得你是谁。

这不是产品设计问题，而是 LLM 的根本特性：**每次 API 调用都是无状态的**。模型看到的只有你这次发送的 messages 列表。上次对话的内容？它根本不知道。

这对聊天机器人可能还行（每轮对话把历史带上就好），但对 Agent 来说是致命问题：

- **Agent 需要记住任务进度**。"上次你帮我分析了 A 文件，现在继续分析 B 文件"——如果 Agent 不记得分析过 A 文件，它就无法"继续"。
- **Agent 需要记住用户偏好**。"我说过了我喜欢简洁的代码风格"——如果 Agent 每次都忘记，用户体验极差。
- **Agent 需要从经验中学习**。上次尝试方案 A 失败了，这次应该试方案 B——如果不记得上次的失败，就会重蹈覆辙。

记忆系统就是解决这些问题的。最简单的形式是管理好发给 LLM 的 messages 列表；更高级的形式是用外部存储保存和检索重要信息。

## 对话历史就是最简单的"记忆"

当你把所有历史消息都放进 messages 列表发给 LLM 时，LLM 就"记住"了之前的对话。这就是最朴素的记忆方案。

```python
messages = [
    {"role": "user", "content": "你好，我叫小明"},
    {"role": "assistant", "content": "你好小明！有什么可以帮你的？"},
    {"role": "user", "content": "帮我写一个排序算法"},
    {"role": "assistant", "content": "好的，以下是冒泡排序的实现..."},
    {"role": "user", "content": "能优化一下吗？"},
    # LLM 看到前面的对话，知道"优化"指的是排序算法
]
```

问题来了：对话越来越长，messages 列表也越来越大。LLM 有**上下文窗口限制**（比如 Claude 是 200K tokens），而且消息越多，API 调用的**成本越高**、**速度越慢**。

核心矛盾：**对话越长上下文越丰富，但 token 消耗越多。** 记忆管理就是在这两者之间找平衡。

## 滑动窗口策略

最简单的解决方案：**只保留最近的 N 条消息**。

```python
class SlidingWindowMemory:
    """滑动窗口记忆：只保留最近 N 条消息"""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages: list[dict] = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def get_messages(self) -> list[dict]:
        return self.messages.copy()

    def clear(self):
        self.messages = []


# 使用示例
memory = SlidingWindowMemory(window_size=6)
memory.add("user", "你好，我叫小明")
memory.add("assistant", "你好小明！")
memory.add("user", "帮我写一个 Python 函数")
memory.add("assistant", "请问函数的功能是什么？")
memory.add("user", "计算斐波那契数列")
memory.add("assistant", "好的，代码如下...")
memory.add("user", "能优化一下性能吗？")  # 第 7 条，最早的被挤出

print(f"当前消息数: {len(memory.get_messages())}")  # 6
# "你好，我叫小明" 已被移除！Agent "忘记"了用户的名字
```

::: warning 滑动窗口的陷阱
窗口移除是"硬删除"——不管信息重不重要，超出窗口就丢弃。用户在第一句说的名字、核心需求可能被删掉。对于需要记住早期信息的场景，滑动窗口太粗暴了。但对于简单的短对话（< 10 轮），它足够好。
:::

## Token 计数与截断

按消息条数管理不够精确——一条消息可能是 10 个字，也可能是 1000 个字。更精确的方式是**按 Token 数量控制**。

```python
class TokenLimitMemory:
    """基于 Token 计数的记忆管理"""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.messages: list[dict] = []

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数（中文约 1 字 = 1.5 token，英文约 4 字符 = 1 token）"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars / 4)

    def _total_tokens(self) -> int:
        return sum(
            self._estimate_tokens(m["content"]) + 4  # 每条消息有格式开销
            for m in self.messages
        )

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # 超出限制时从最早的消息开始移除
        while self._total_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)

    def get_messages(self) -> list[dict]:
        return self.messages.copy()

    def remaining_tokens(self) -> int:
        """剩余可用的 token 数"""
        return self.max_tokens - self._total_tokens()
```

::: tip 精确 vs 估算
上面的 `_estimate_tokens` 是粗略估算。如果需要精确计数，可以用 `tiktoken` 库：
```python
import tiktoken
encoder = tiktoken.encoding_for_model("gpt-4")
tokens = len(encoder.encode(text))
```
对于 Claude 模型，官方没有提供公开的 tokenizer，但 API 响应中会返回 `usage.input_tokens`，可以据此调整。
:::

## Tool Use 消息的特殊处理

如果你的 Agent 使用了工具，消息历史中会有 `tool_use` 和 `tool_result` 消息。这些消息在截断时需要特别注意。

```python
class AgentMemoryManager:
    """Agent 专用的记忆管理（处理 Tool Use 消息）"""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages

    def trim(self, messages: list[dict]) -> list[dict]:
        """安全地裁剪消息列表"""
        if len(messages) <= self.max_messages:
            return messages

        trimmed = messages[-self.max_messages:]

        # 关键：修复边界的孤儿消息
        trimmed = self._fix_boundaries(trimmed)
        return trimmed

    def _fix_boundaries(self, messages: list[dict]) -> list[dict]:
        """确保 tool_use 和 tool_result 成对出现"""
        # 如果第一条是 tool_result（user 角色中的数组），它的 tool_use 可能已被裁掉
        while messages:
            first = messages[0]
            content = first.get("content", "")

            # 检查是否是 tool_result 消息
            if isinstance(content, list) and content:
                if content[0].get("type") == "tool_result":
                    messages = messages[1:]  # 移除孤儿 tool_result
                    continue

            # 检查是否是包含 tool_use 的 assistant 消息（后面没有对应的 tool_result）
            if first.get("role") == "assistant" and isinstance(content, list):
                has_tool_use = any(
                    getattr(block, "type", None) == "tool_use" or
                    (isinstance(block, dict) and block.get("type") == "tool_use")
                    for block in content
                )
                if has_tool_use and len(messages) < 2:
                    messages = messages[1:]
                    continue

            break

        return messages
```

::: danger Tool Use 消息配对规则
- assistant 的 `tool_use` 必须有对应的 user `tool_result`
- `tool_result` 的 `tool_use_id` 必须匹配 `tool_use` 的 `id`
- 裁剪时如果切断了配对，API 会报错
- 宁可多保留几条消息，也不要破坏配对关系
:::

## 用 SQLite 持久化对话

到目前为止，我们的记忆都在内存中——程序一关就丢了。用 SQLite 可以零依赖地把对话持久化到磁盘。

```python
import sqlite3
import json
from datetime import datetime

class PersistentMemory:
    """SQLite 持久化记忆"""

    def __init__(self, db_path: str = "./agent_memory.db", session_id: str = "default"):
        self.session_id = session_id
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TEXT
            )
        """)
        self.conn.commit()

    def add(self, role: str, content):
        """保存一条消息（content 可以是字符串或列表/字典）"""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        self.conn.execute(
            "INSERT INTO conversations (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (self.session_id, role, content, datetime.now().isoformat())
        )
        self.conn.commit()

    def get_messages(self, limit: int = 50) -> list[dict]:
        """获取最近的 N 条消息"""
        cursor = self.conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (self.session_id, limit)
        )
        rows = list(cursor)[::-1]  # 反转为时间正序
        messages = []
        for role, content in rows:
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                pass
            messages.append({"role": role, "content": content})
        return messages

    def get_sessions(self) -> list[str]:
        """列出所有会话"""
        cursor = self.conn.execute(
            "SELECT DISTINCT session_id FROM conversations"
        )
        return [row[0] for row in cursor]

    def clear_session(self):
        """清除当前会话"""
        self.conn.execute(
            "DELETE FROM conversations WHERE session_id = ?",
            (self.session_id,)
        )
        self.conn.commit()


# 使用示例
memory = PersistentMemory(session_id="user_123_session_1")
memory.add("user", "你好，我是产品经理")
memory.add("assistant", "你好！有什么可以帮你的？")

# 程序重启后，数据还在
memory2 = PersistentMemory(session_id="user_123_session_1")
messages = memory2.get_messages()
print(f"恢复了 {len(messages)} 条消息")
for msg in messages:
    print(f"  [{msg['role']}] {msg['content']}")
```

## 把记忆管理集成到 Agent

把滑动窗口和持久化组合起来，集成到 Agent 中：

```python
import anthropic
import json

client = anthropic.Anthropic()

class MemoryAgent:
    """带记忆管理的 Agent"""

    def __init__(self, system_prompt: str = "", max_messages: int = 20):
        self.system_prompt = system_prompt
        self.window = SlidingWindowMemory(window_size=max_messages)
        self.storage = PersistentMemory()

    def chat(self, user_input: str) -> str:
        # 1. 记录用户消息
        self.window.add("user", user_input)
        self.storage.add("user", user_input)

        # 2. 调用 LLM
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=self.system_prompt,
            messages=self.window.get_messages(),
        )
        reply = response.content[0].text

        # 3. 记录助手回复
        self.window.add("assistant", reply)
        self.storage.add("assistant", reply)

        return reply

    def load_history(self, limit: int = 20):
        """从持久化存储加载历史"""
        messages = self.storage.get_messages(limit)
        self.window.messages = messages
        print(f"已加载 {len(messages)} 条历史消息")


# 使用
agent = MemoryAgent(system_prompt="你是一个友好的助手，会记住用户的信息。")
agent.load_history()  # 加载上次的对话

print(agent.chat("你还记得我上次说什么吗？"))
```

## 各策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 滑动窗口 | 最简单，零额外成本 | 硬删除，丢失早期信息 | 短对话（< 10 轮） |
| Token 截断 | 精确控制成本 | 同滑动窗口 | 成本敏感场景 |
| 持久化存储 | 跨会话保持 | 需要管理存储 | 需要对话连续性 |
| 窗口 + 持久化 | 兼顾速度和持久性 | 实现稍复杂 | 大多数生产场景 |

## 小结

- **LLM 无状态**：每次 API 调用独立，记忆需要你来管理
- **滑动窗口**：最简单的策略，只保留最近 N 条，但可能丢失重要信息
- **Token 截断**：按 token 数控制，比按条数更精确
- **Tool Use 处理**：裁剪时必须保持 tool_use 和 tool_result 的配对关系
- **持久化**：用 SQLite 零依赖实现跨会话的对话保存

## 练习

1. **动手做**：实现一个 `TokenLimitMemory`，设置 max_tokens=2000，模拟一段 20 轮对话，观察何时开始截断。
2. **安全裁剪**：给 `AgentMemoryManager` 添加测试用例，验证当 tool_use/tool_result 配对被切断时，修复逻辑是否正确。
3. **持久化增强**：给 `PersistentMemory` 添加按时间范围查询的功能（"查看今天的对话"）。
4. **思考题**：滑动窗口会删掉用户第一句说的名字。如何在不增加太多复杂度的前提下，让 Agent 不忘记"重要的早期信息"？

## 参考资源

- [Anthropic: Long Context Window Tips](https://docs.anthropic.com/en/docs/build-with-claude/context-windows) -- Anthropic 长上下文使用建议
- [LangChain: Conversation Memory](https://python.langchain.com/docs/concepts/memory/) -- LangChain 记忆类型文档
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) -- 在线 Token 计数工具
- [tiktoken Library](https://github.com/openai/tiktoken) -- OpenAI 官方 token 计数库
