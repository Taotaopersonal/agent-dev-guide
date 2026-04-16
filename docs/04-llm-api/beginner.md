# LLM API 调用 · 初级

::: info 学习目标
- 完成 Anthropic 和 OpenAI 的环境配置
- 深入理解 Messages API / Chat Completions API 的请求和响应结构
- 掌握两套 SDK 的使用方法和关键差异
- 实现多轮对话管理器
- 学完能独立调用 LLM API 完成各种任务

预计学习时间：2-3 小时
:::

## 准备工作

### 安装与配置

```bash
# 安装两个主要的 LLM SDK
npm install @anthropic-ai/sdk openai

# 设置环境变量（推荐方式）
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export OPENAI_API_KEY="sk-..."
```

```typescript
// 或者使用 .env 文件（添加到 .gitignore）
// npm install dotenv
import "dotenv/config"; // 自动加载 .env 文件中的环境变量
```

::: warning 安全提醒
- API Key 只会显示一次，丢失后需重新创建
- 不要将 Key 硬编码在代码中或提交到 Git
- 使用环境变量或 `.env` 文件管理 Key
:::

## Claude Messages API

Messages API 是与 Claude 交互的核心接口。

### 请求结构

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic(); // 自动读取 ANTHROPIC_API_KEY

const response = await client.messages.create({
  // 必填参数
  model: "claude-sonnet-4-20250514",  // 模型名称
  max_tokens: 1024,                    // 最大输出 token 数（必填）
  messages: [                          // 对话消息列表
    { role: "user", content: "你好，介绍一下你自己" },
  ],

  // 可选参数
  system: "你是一位友好的 AI 助手。",  // System Prompt（独立参数）
  temperature: 0.7,                    // 随机性 0-1
  stop_sequences: ["END"],             // 停止序列
});
```

**参数速查：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | 模型标识符 |
| `max_tokens` | number | 是 | 最大输出 token 数 |
| `messages` | array | 是 | 对话消息列表 |
| `system` | string | 否 | System Prompt |
| `temperature` | number | 否 | 0-1，控制随机性 |
| `top_p` | number | 否 | 核采样，与 temperature 二选一 |
| `stop_sequences` | string[] | 否 | 自定义停止序列 |

### 响应结构

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const response = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 1024,
  messages: [{ role: "user", content: "用一句话介绍 TypeScript" }],
});

// 提取回复文本（最常用）
const answer =
  response.content[0].type === "text" ? response.content[0].text : "";
console.log(`回复: ${answer}`);

// 完整响应信息
console.log(`模型: ${response.model}`);
console.log(`停止原因: ${response.stop_reason}`); // end_turn / max_tokens / tool_use
console.log(`输入 Token: ${response.usage.input_tokens}`);
console.log(`输出 Token: ${response.usage.output_tokens}`);
```

**stop_reason 含义：**

| 值 | 含义 | 处理方式 |
|------|------|---------|
| `end_turn` | 模型自然结束 | 正常处理 |
| `max_tokens` | 达到上限被截断 | 可能需要继续生成 |
| `stop_sequence` | 遇到停止序列 | 按业务逻辑处理 |
| `tool_use` | 模型请求调用工具 | 执行工具并返回结果 |

### 消息角色

```typescript
// user：用户的输入
// assistant：模型的回复（多轮对话中需传入历史回复）
// system：通过独立的 system 参数传入（不在 messages 中）

import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const messages: Anthropic.MessageParam[] = [
  { role: "user", content: "帮我写一个快速排序" },
  { role: "assistant", content: "好的，以下是 TypeScript 实现..." },
  { role: "user", content: "能改成非递归版本吗？" },
];

const response = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 1024,
  system: "你是一位擅长算法的工程师。",
  messages,
});
```

::: tip 消息顺序规则
- messages 中 `user` 和 `assistant` 必须交替出现
- 列表必须以 `user` 消息开始
- `system` 不放在 messages 里，通过独立参数传入
:::

## OpenAI Chat Completions API

OpenAI 的接口是行业事实标准，许多第三方模型（DeepSeek、Mistral）也兼容它。

### 基础调用

```typescript
import OpenAI from "openai";

const client = new OpenAI(); // 自动读取 OPENAI_API_KEY

const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [
    { role: "system", content: "你是一位 TypeScript 开发者。" }, // system 在 messages 中
    { role: "user", content: "用一句话解释 GIL" },
  ],
  max_tokens: 256,
  temperature: 0.7,
});

// 提取回复
console.log(response.choices[0].message.content);

// Token 统计
console.log(`输入: ${response.usage?.prompt_tokens}`);
console.log(`输出: ${response.usage?.completion_tokens}`);
```

### 使用 OpenAI 兼容接口

许多第三方模型服务兼容 OpenAI 接口，只需修改 base_url：

```typescript
import OpenAI from "openai";

// DeepSeek（兼容 OpenAI 接口）
const deepseek = new OpenAI({
  apiKey: "your-deepseek-key",
  baseURL: "https://api.deepseek.com",
});
const response = await deepseek.chat.completions.create({
  model: "deepseek-chat",
  messages: [{ role: "user", content: "你好" }],
  max_tokens: 256,
});
console.log(response.choices[0].message.content);

// 本地模型（如 Ollama）
const ollama = new OpenAI({
  apiKey: "ollama", // Ollama 不需要真实 key
  baseURL: "http://localhost:11434/v1",
});
const localResponse = await ollama.chat.completions.create({
  model: "llama3",
  messages: [{ role: "user", content: "你好" }],
});
```

## Claude vs OpenAI：关键差异

两套 API 功能高度相似，但细节上有不少差异。掌握这些对写统一适配层至关重要。

| 维度 | Claude (Anthropic) | OpenAI |
|------|-------------------|--------|
| **System Prompt** | 独立 `system` 参数 | `messages` 中 role="system" |
| **响应内容** | `response.content[0].text` | `response.choices[0].message.content` |
| **Token 统计** | `usage.input_tokens` / `output_tokens` | `usage.prompt_tokens` / `completion_tokens` |
| **停止原因** | `stop_reason`: end_turn / max_tokens | `finish_reason`: stop / length |
| **必填 max_tokens** | 是 | 否（有默认值） |
| **JSON 模式** | Tool Use 或 Prompt 约束 | `response_format` 参数 |
| **客户端类** | `new Anthropic()` | `new OpenAI()` |

```typescript
// Claude 的写法
import Anthropic from "@anthropic-ai/sdk";
const claude = new Anthropic();
const resp = await claude.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 256,
  system: "你是助手",                       // 独立参数
  messages: [{ role: "user", content: "你好" }],
});
const text =
  resp.content[0].type === "text" ? resp.content[0].text : "";   // 取内容
const tokens = resp.usage.input_tokens;                          // 取 token

// OpenAI 的写法
import OpenAI from "openai";
const gpt = new OpenAI();
const resp2 = await gpt.chat.completions.create({
  model: "gpt-4o",
  messages: [
    { role: "system", content: "你是助手" },  // 在 messages 中
    { role: "user", content: "你好" },
  ],
});
const text2 = resp2.choices[0].message.content;      // 取内容
const tokens2 = resp2.usage?.prompt_tokens;           // 取 token
```

## 多轮对话管理

在 Agent 开发中，多轮对话是核心能力。关键是每次调用时传入完整的对话历史。

```typescript
import Anthropic from "@anthropic-ai/sdk";

interface ConversationStats {
  turns: number;
  total_input_tokens: number;
  total_output_tokens: number;
}

class Conversation {
  /** 多轮对话管理器 */
  private client: Anthropic;
  private model: string;
  private system: string;
  private maxHistory: number;
  private messages: Anthropic.MessageParam[] = [];
  private totalInputTokens = 0;
  private totalOutputTokens = 0;

  constructor(
    model = "claude-sonnet-4-20250514",
    system = "",
    maxHistory = 20
  ) {
    this.client = new Anthropic();
    this.model = model;
    this.system = system;
    this.maxHistory = maxHistory;
  }

  /** 发送消息并获取回复 */
  async chat(userInput: string): Promise<string> {
    this.messages.push({ role: "user", content: userInput });

    // 截断历史（保留最近 N 轮）
    if (this.messages.length > this.maxHistory * 2) {
      this.messages = this.messages.slice(-(this.maxHistory * 2));
    }

    const params: Anthropic.MessageCreateParams = {
      model: this.model,
      max_tokens: 1024,
      messages: this.messages,
    };
    if (this.system) {
      params.system = this.system;
    }

    const response = await this.client.messages.create(params);
    const assistantMsg =
      response.content[0].type === "text" ? response.content[0].text : "";

    this.messages.push({ role: "assistant", content: assistantMsg });
    this.totalInputTokens += response.usage.input_tokens;
    this.totalOutputTokens += response.usage.output_tokens;

    return assistantMsg;
  }

  reset(): void {
    this.messages = [];
  }

  getStats(): ConversationStats {
    return {
      turns: Math.floor(this.messages.length / 2),
      total_input_tokens: this.totalInputTokens,
      total_output_tokens: this.totalOutputTokens,
    };
  }
}

// 使用
const conv = new Conversation(
  "claude-sonnet-4-20250514",
  "你是一位全栈开发导师。每次只教一个步骤。"
);

const questions = [
  "我想做一个 TODO 应用，从哪里开始？",
  "好，我选 Vue 3。项目怎么创建？",
  "创建好了，接下来呢？",
];

for (const q of questions) {
  console.log(`\n学生: ${q}`);
  const answer = await conv.chat(q);
  console.log(`导师: ${answer.slice(0, 200)}...`);
}

const stats = conv.getStats();
console.log(`\n对话轮数: ${stats.turns}`);
console.log(
  `总 Token: ${stats.total_input_tokens} in / ${stats.total_output_tokens} out`
);
```

::: warning 多轮对话的 Token 成本
每次 API 调用都传入完整历史，input token 随对话长度线性增长。长对话需要：
1. 设置 `max_history` 限制历史长度
2. 或实现摘要机制，用摘要替代早期对话
3. 监控 token 使用量，避免意外高额费用
:::

## 异步调用

Agent 系统中经常需要并发调用多个 LLM 请求：

```typescript
import Anthropic from "@anthropic-ai/sdk";

async function asyncChat(): Promise<string> {
  /** 异步 API 调用 */
  const client = new Anthropic();
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 256,
    messages: [{ role: "user", content: "用一句话解释异步编程" }],
  });
  return response.content[0].type === "text" ? response.content[0].text : "";
}

async function parallelCalls(): Promise<void> {
  /** 并行调用多个请求 */
  const client = new Anthropic();
  const topics = ["递归", "闭包", "协程"];

  const tasks = topics.map((topic) =>
    client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 128,
      messages: [{ role: "user", content: `用一句话解释${topic}` }],
    })
  );

  const responses = await Promise.all(tasks);
  for (const resp of responses) {
    const text =
      resp.content[0].type === "text" ? resp.content[0].text : "";
    console.log(`- ${text}`);
  }
}

// 运行
await parallelCalls();
```

## 小结

1. **Claude Messages API**：system 是独立参数，max_tokens 必填，响应在 `content[0].text`
2. **OpenAI Chat Completions API**：system 放在 messages 中，是行业事实标准接口
3. **两套 API 的核心差异**在于 System Prompt 位置、响应结构和 Token 字段名
4. **多轮对话**通过传入完整历史实现上下文记忆，需注意 Token 成本控制
5. **异步调用**使用 `Promise.all` 实现并发，TypeScript 中客户端本身即支持异步

## 练习

1. **基础练习**：创建一个命令行聊天程序，支持多轮对话，输入 "quit" 退出，退出时打印总 Token 使用量和估算费用。

2. **对比练习**：分别用 Claude 和 OpenAI API 实现一个"翻译助手"（中译英），对比翻译质量和 Token 用量差异。

3. **进阶练习**：实现一个 `SmartConversation` 类，当对话超过 10 轮时，自动调用 LLM 对前 5 轮做摘要，用摘要替代原始历史，控制 Token 用量。

4. **Temperature 实验**：对同一个问题，用 `temperature=0`、`0.5`、`1.0` 各调用 3 次，记录输出差异。

## 参考资源

- [Anthropic Claude API 文档](https://docs.anthropic.com/en/api/messages)
- [OpenAI Chat Completions 文档](https://platform.openai.com/docs/api-reference/chat)
- [Anthropic TypeScript SDK](https://github.com/anthropics/anthropic-sdk-typescript)
- [OpenAI TypeScript SDK](https://github.com/openai/openai-node)
