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

```typescript
const messages = [
  { role: "user", content: "你好，我叫小明" },
  { role: "assistant", content: "你好小明！有什么可以帮你的？" },
  { role: "user", content: "帮我写一个排序算法" },
  { role: "assistant", content: "好的，以下是冒泡排序的实现..." },
  { role: "user", content: "能优化一下吗？" },
  // LLM 看到前面的对话，知道"优化"指的是排序算法
];
```

问题来了：对话越来越长，messages 列表也越来越大。LLM 有**上下文窗口限制**（比如 Claude 是 200K tokens），而且消息越多，API 调用的**成本越高**、**速度越慢**。

核心矛盾：**对话越长上下文越丰富，但 token 消耗越多。** 记忆管理就是在这两者之间找平衡。

## 滑动窗口策略

最简单的解决方案：**只保留最近的 N 条消息**。

```typescript
interface Message {
  role: string;
  content: string;
}

class SlidingWindowMemory {
  /** 滑动窗口记忆：只保留最近 N 条消息 */
  private windowSize: number;
  private messages: Message[] = [];

  constructor(windowSize: number = 10) {
    this.windowSize = windowSize;
  }

  add(role: string, content: string): void {
    this.messages.push({ role, content });
    if (this.messages.length > this.windowSize) {
      this.messages = this.messages.slice(-this.windowSize);
    }
  }

  getMessages(): Message[] {
    return [...this.messages];
  }

  clear(): void {
    this.messages = [];
  }
}

// 使用示例
const memory = new SlidingWindowMemory(6);
memory.add("user", "你好，我叫小明");
memory.add("assistant", "你好小明！");
memory.add("user", "帮我写一个 Python 函数");
memory.add("assistant", "请问函数的功能是什么？");
memory.add("user", "计算斐波那契数列");
memory.add("assistant", "好的，代码如下...");
memory.add("user", "能优化一下性能吗？"); // 第 7 条，最早的被挤出

console.log(`当前消息数: ${memory.getMessages().length}`); // 6
// "你好，我叫小明" 已被移除！Agent "忘记"了用户的名字
```

::: warning 滑动窗口的陷阱
窗口移除是"硬删除"——不管信息重不重要，超出窗口就丢弃。用户在第一句说的名字、核心需求可能被删掉。对于需要记住早期信息的场景，滑动窗口太粗暴了。但对于简单的短对话（< 10 轮），它足够好。
:::

## Token 计数与截断

按消息条数管理不够精确——一条消息可能是 10 个字，也可能是 1000 个字。更精确的方式是**按 Token 数量控制**。

```typescript
class TokenLimitMemory {
  /** 基于 Token 计数的记忆管理 */
  private maxTokens: number;
  private messages: Message[] = [];

  constructor(maxTokens: number = 4000) {
    this.maxTokens = maxTokens;
  }

  private estimateTokens(text: string): number {
    /** 粗略估算 token 数（中文约 1 字 = 1.5 token，英文约 4 字符 = 1 token） */
    let chineseChars = 0;
    for (const c of text) {
      if (c >= "\u4e00" && c <= "\u9fff") chineseChars++;
    }
    const otherChars = text.length - chineseChars;
    return Math.floor(chineseChars * 1.5 + otherChars / 4);
  }

  private totalTokens(): number {
    return this.messages.reduce(
      (sum, m) => sum + this.estimateTokens(m.content) + 4, // 每条消息有格式开销
      0
    );
  }

  add(role: string, content: string): void {
    this.messages.push({ role, content });
    // 超出限制时从最早的消息开始移除
    while (this.totalTokens() > this.maxTokens && this.messages.length > 1) {
      this.messages.shift();
    }
  }

  getMessages(): Message[] {
    return [...this.messages];
  }

  remainingTokens(): number {
    /** 剩余可用的 token 数 */
    return this.maxTokens - this.totalTokens();
  }
}
```

::: tip 精确 vs 估算
上面的 `_estimate_tokens` 是粗略估算。如果需要精确计数，可以用 `tiktoken` 库：
```typescript
import { encoding_for_model } from "tiktoken";
const encoder = encoding_for_model("gpt-4");
const tokens = encoder.encode(text).length;
```
对于 Claude 模型，官方没有提供公开的 tokenizer，但 API 响应中会返回 `usage.input_tokens`，可以据此调整。
:::

## Tool Use 消息的特殊处理

如果你的 Agent 使用了工具，消息历史中会有 `tool_use` 和 `tool_result` 消息。这些消息在截断时需要特别注意。

```typescript
class AgentMemoryManager {
  /** Agent 专用的记忆管理（处理 Tool Use 消息） */
  private maxMessages: number;

  constructor(maxMessages: number = 20) {
    this.maxMessages = maxMessages;
  }

  trim(messages: Record<string, any>[]): Record<string, any>[] {
    /** 安全地裁剪消息列表 */
    if (messages.length <= this.maxMessages) {
      return messages;
    }

    let trimmed = messages.slice(-this.maxMessages);

    // 关键：修复边界的孤儿消息
    trimmed = this.fixBoundaries(trimmed);
    return trimmed;
  }

  private fixBoundaries(messages: Record<string, any>[]): Record<string, any>[] {
    /** 确保 tool_use 和 tool_result 成对出现 */
    // 如果第一条是 tool_result（user 角色中的数组），它的 tool_use 可能已被裁掉
    while (messages.length > 0) {
      const first = messages[0];
      const content = first.content ?? "";

      // 检查是否是 tool_result 消息
      if (Array.isArray(content) && content.length > 0) {
        if (content[0]?.type === "tool_result") {
          messages = messages.slice(1); // 移除孤儿 tool_result
          continue;
        }
      }

      // 检查是否是包含 tool_use 的 assistant 消息（后面没有对应的 tool_result）
      if (first.role === "assistant" && Array.isArray(content)) {
        const hasToolUse = content.some(
          (block: any) => block?.type === "tool_use"
        );
        if (hasToolUse && messages.length < 2) {
          messages = messages.slice(1);
          continue;
        }
      }

      break;
    }

    return messages;
  }
}
```

::: danger Tool Use 消息配对规则
- assistant 的 `tool_use` 必须有对应的 user `tool_result`
- `tool_result` 的 `tool_use_id` 必须匹配 `tool_use` 的 `id`
- 裁剪时如果切断了配对，API 会报错
- 宁可多保留几条消息，也不要破坏配对关系
:::

## 用 SQLite 持久化对话

到目前为止，我们的记忆都在内存中——程序一关就丢了。用 SQLite 可以零依赖地把对话持久化到磁盘。

```typescript
import Database from "better-sqlite3";

class PersistentMemory {
  /** SQLite 持久化记忆 */
  private sessionId: string;
  private db: Database.Database;

  constructor(dbPath: string = "./agent_memory.db", sessionId: string = "default") {
    this.sessionId = sessionId;
    this.db = new Database(dbPath);
    this.createTables();
  }

  private createTables(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        created_at TEXT
      )
    `);
  }

  add(role: string, content: string | any): void {
    /** 保存一条消息（content 可以是字符串或对象/数组） */
    const serialized = typeof content === "string" ? content : JSON.stringify(content);
    this.db
      .prepare(
        "INSERT INTO conversations (session_id, role, content, created_at) VALUES (?, ?, ?, ?)"
      )
      .run(this.sessionId, role, serialized, new Date().toISOString());
  }

  getMessages(limit: number = 50): Record<string, any>[] {
    /** 获取最近的 N 条消息 */
    const rows = this.db
      .prepare(
        "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id DESC LIMIT ?"
      )
      .all(this.sessionId, limit) as { role: string; content: string }[];

    return rows.reverse().map(({ role, content }) => {
      let parsed: any = content;
      try {
        parsed = JSON.parse(content);
      } catch {
        // 保持原始字符串
      }
      return { role, content: parsed };
    });
  }

  getSessions(): string[] {
    /** 列出所有会话 */
    const rows = this.db
      .prepare("SELECT DISTINCT session_id FROM conversations")
      .all() as { session_id: string }[];
    return rows.map((row) => row.session_id);
  }

  clearSession(): void {
    /** 清除当前会话 */
    this.db
      .prepare("DELETE FROM conversations WHERE session_id = ?")
      .run(this.sessionId);
  }
}

// 使用示例
const memory = new PersistentMemory("./agent_memory.db", "user_123_session_1");
memory.add("user", "你好，我是产品经理");
memory.add("assistant", "你好！有什么可以帮你的？");

// 程序重启后，数据还在
const memory2 = new PersistentMemory("./agent_memory.db", "user_123_session_1");
const messages = memory2.getMessages();
console.log(`恢复了 ${messages.length} 条消息`);
for (const msg of messages) {
  console.log(`  [${msg.role}] ${msg.content}`);
}
```

## 把记忆管理集成到 Agent

把滑动窗口和持久化组合起来，集成到 Agent 中：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

class MemoryAgent {
  /** 带记忆管理的 Agent */
  private systemPrompt: string;
  private window: SlidingWindowMemory;
  private storage: PersistentMemory;

  constructor(systemPrompt: string = "", maxMessages: number = 20) {
    this.systemPrompt = systemPrompt;
    this.window = new SlidingWindowMemory(maxMessages);
    this.storage = new PersistentMemory();
  }

  async chat(userInput: string): Promise<string> {
    // 1. 记录用户消息
    this.window.add("user", userInput);
    this.storage.add("user", userInput);

    // 2. 调用 LLM
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      system: this.systemPrompt,
      messages: this.window.getMessages(),
    });
    const reply = (response.content[0] as { text: string }).text;

    // 3. 记录助手回复
    this.window.add("assistant", reply);
    this.storage.add("assistant", reply);

    return reply;
  }

  loadHistory(limit: number = 20): void {
    /** 从持久化存储加载历史 */
    const messages = this.storage.getMessages(limit);
    (this.window as any).messages = messages;
    console.log(`已加载 ${messages.length} 条历史消息`);
  }
}

// 使用
const agent = new MemoryAgent("你是一个友好的助手，会记住用户的信息。");
agent.loadHistory(); // 加载上次的对话

console.log(await agent.chat("你还记得我上次说什么吗？"));
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
