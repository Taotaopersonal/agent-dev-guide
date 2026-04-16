# LLM API 调用 · 中级

::: info 学习目标
- 理解流式输出的原理（SSE 协议）
- 实现 Claude 和 OpenAI 的流式调用
- 掌握 Express 流式端点 + 前端消费的完整链路
- 学会图片、PDF 等多模态输入的处理方法
- 设计统一的 LLM 适配层，支持多模型切换
- 学完能构建有流式打字效果的 AI 对话应用

预计学习时间：3-4 小时
:::

## 流式输出（Streaming）

### 为什么需要流式输出

LLM 生成文本是逐 token 进行的，一个完整回复可能需要几秒到几十秒。非流式调用让用户干等到全部生成完毕，而流式输出让文字像打字一样逐步呈现。

两大价值：
1. **用户体验**：感知延迟从"生成总时间"降低到"首 token 时间"（通常 <1 秒）
2. **长响应处理**：用户可以在生成过程中就开始阅读，甚至提前终止

### SSE 协议原理

LLM 的流式 API 基于 SSE（Server-Sent Events）协议——一种服务器向客户端单向推送数据的 HTTP 协议。

```
客户端                     服务器
  |--- HTTP Request -------> |
  |                          |（保持连接不断）
  |<--- data: {"token":"你"} |
  |<--- data: {"token":"好"} |
  |<--- data: [DONE]         |
  |--- 连接关闭 -------------|
```

关键特征：
- 使用标准 HTTP（不是 WebSocket）
- 响应 Content-Type 为 `text/event-stream`
- 数据格式：`data: <JSON>\n\n`

### Claude 流式实现

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function streamBasic(userInput: string): Promise<void> {
  /** 基础流式输出 */
  process.stdout.write("Claude: ");

  const stream = client.messages.stream({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [{ role: "user", content: userInput }],
  });

  stream.on("text", (text) => {
    process.stdout.write(text);
  });

  await stream.finalMessage();
  console.log();
}

streamBasic("用 TypeScript 写一个二分查找，加注释");
```

### 处理完整的流式事件

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function streamWithStats(userInput: string): Promise<string> {
  /** 流式输出并统计 Token */
  let fullText = "";
  let inputTokens = 0;
  let outputTokens = 0;

  const stream = client.messages.stream({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    system: "你是一位简洁的技术助手。",
    messages: [{ role: "user", content: userInput }],
  });

  stream.on("text", (text) => {
    fullText += text;
    process.stdout.write(text);
  });

  stream.on("message", (message) => {
    inputTokens = message.usage.input_tokens;
    outputTokens = message.usage.output_tokens;
  });

  await stream.finalMessage();

  console.log(`\n[Tokens: ${inputTokens} in / ${outputTokens} out]`);
  return fullText;
}

streamWithStats("什么是事件循环？");
```

### OpenAI 流式实现

```typescript
import OpenAI from "openai";

const client = new OpenAI();

async function openaiStream(userInput: string): Promise<string> {
  /** OpenAI 流式输出 */
  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: userInput }],
    max_tokens: 1024,
    stream: true,
    stream_options: { include_usage: true }, // 请求返回用量
  });

  let fullText = "";
  for await (const chunk of stream) {
    if (chunk.choices?.[0]?.delta?.content) {
      const text = chunk.choices[0].delta.content;
      fullText += text;
      process.stdout.write(text);
    }

    if (chunk.usage) {
      console.log(
        `\n[Tokens: ${chunk.usage.prompt_tokens} in / ` +
          `${chunk.usage.completion_tokens} out]`
      );
    }
  }

  return fullText;
}

openaiStream("解释 TypeScript 的类型系统");
```

### Express 流式端点

在 Agent 应用中，前端需要消费后端转发的流式数据：

```typescript
/**
 * 后端服务：npx tsx server.ts
 * 或配合 express: npx ts-node server.ts
 */

import express from "express";
import cors from "cors";
import Anthropic from "@anthropic-ai/sdk";

const app = express();
app.use(cors());
app.use(express.json());

const client = new Anthropic();

interface ChatRequest {
  message: string;
  system?: string;
}

app.post("/chat/stream", async (req, res) => {
  const { message, system = "你是一位友好的 AI 助手。" } = req.body as ChatRequest;

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const stream = client.messages.stream({
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    system,
    messages: [{ role: "user", content: message }],
  });

  stream.on("text", (text) => {
    const data = JSON.stringify({ type: "text", content: text });
    res.write(`data: ${data}\n\n`);
  });

  stream.on("finalMessage", () => {
    res.write(`data: ${JSON.stringify({ type: "done" })}\n\n`);
    res.end();
  });

  stream.on("error", (err) => {
    console.error("Stream error:", err);
    res.end();
  });
});

app.listen(8000, () => {
  console.log("Server running on http://localhost:8000");
});
```

### 前端消费流（JavaScript）

```javascript
async function streamChat(message, onChunk, onDone) {
  const response = await fetch('http://localhost:8000/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop(); // 最后一段可能不完整

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (data.type === 'text') onChunk(data.content);
        else if (data.type === 'done') { onDone(); return; }
      }
    }
  }
}

// 使用
let fullText = '';
streamChat('用 JavaScript 实现防抖函数',
  (chunk) => { fullText += chunk; console.log(chunk); },
  () => console.log('完成:', fullText)
);
```

### Vue 3 流式组件

```vue
<script setup>
import { ref } from 'vue'

const message = ref('')
const response = ref('')
const loading = ref(false)

async function sendMessage() {
  if (!message.value.trim() || loading.value) return
  loading.value = true
  response.value = ''

  try {
    const res = await fetch('http://localhost:8000/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: message.value }),
    })

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop()
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6))
          if (data.type === 'text') response.value += data.content
        }
      }
    }
  } catch (err) {
    response.value = `错误: ${err.message}`
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="chat">
    <div class="response">{{ response || '等待输入...' }}</div>
    <input v-model="message" @keyup.enter="sendMessage" placeholder="输入消息..." />
    <button @click="sendMessage" :disabled="loading">
      {{ loading ? '生成中...' : '发送' }}
    </button>
  </div>
</template>
```

::: warning 流式输出注意事项
1. **错误处理**：传输中途可能断开（网络波动），需捕获异常提示用户
2. **取消机制**：用户可能想取消生成，前端用 AbortController 实现中断
3. **Token 统计**：流式模式下统计信息在最后一个事件中返回
4. **并发限制**：同一用户的多个流式请求可能触发 Rate Limit，前端建议加锁
:::

## 多模态输入

### 图片输入（Vision）

Claude 支持两种方式传入图片：base64 编码和 URL。

```typescript
import Anthropic from "@anthropic-ai/sdk";
import * as fs from "fs";
import * as path from "path";

const client = new Anthropic();

// 方式一：Base64（本地图片）
async function analyzeLocalImage(imagePath: string, question: string): Promise<string> {
  const imageData = fs.readFileSync(imagePath).toString("base64");

  const ext = path.extname(imagePath).slice(1).toLowerCase();
  const mediaTypes: Record<string, string> = {
    jpg: "image/jpeg", jpeg: "image/jpeg",
    png: "image/png", gif: "image/gif", webp: "image/webp",
  };

  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [{
      role: "user",
      content: [
        { type: "image", source: {
          type: "base64",
          media_type: (mediaTypes[ext] || "image/png") as "image/jpeg" | "image/png" | "image/gif" | "image/webp",
          data: imageData,
        }},
        { type: "text", text: question },
      ],
    }],
  });

  return response.content[0].type === "text" ? response.content[0].text : "";
}

// 方式二：URL（网络图片）
async function analyzeUrlImage(url: string, question: string): Promise<string> {
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [{
      role: "user",
      content: [
        { type: "image", source: { type: "url", url } },
        { type: "text", text: question },
      ],
    }],
  });

  return response.content[0].type === "text" ? response.content[0].text : "";
}

// 使用
const result = await analyzeUrlImage(
  "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/200px-Python-logo-notext.svg.png",
  "这个 logo 属于哪个编程语言？"
);
console.log(result);
```

### 多图片对比

```typescript
import Anthropic from "@anthropic-ai/sdk";
import * as fs from "fs";
import * as path from "path";

const client = new Anthropic();

type ImageMediaType = "image/jpeg" | "image/png" | "image/gif" | "image/webp";

async function compareImages(path1: string, path2: string, question: string): Promise<string> {
  function load(filePath: string): Anthropic.ImageBlockParam {
    const data = fs.readFileSync(filePath).toString("base64");
    const ext = path.extname(filePath).slice(1).toLowerCase();
    return {
      type: "image",
      source: {
        type: "base64",
        media_type: (ext === "png" ? "image/png" : "image/jpeg") as ImageMediaType,
        data,
      },
    };
  }

  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    messages: [{
      role: "user",
      content: [
        { type: "text", text: "图片 1：" },
        load(path1),
        { type: "text", text: "图片 2：" },
        load(path2),
        { type: "text", text: question },
      ],
    }],
  });

  return response.content[0].type === "text" ? response.content[0].text : "";
}

// compareImages("v1.png", "v2.png", "对比两个版本的差异和改进点。");
```

### PDF 文档输入

```typescript
import Anthropic from "@anthropic-ai/sdk";
import * as fs from "fs";

const client = new Anthropic();

async function analyzePdf(pdfPath: string, question: string): Promise<string> {
  const pdfData = fs.readFileSync(pdfPath).toString("base64");

  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 4096,
    messages: [{
      role: "user",
      content: [
        { type: "document", source: {
          type: "base64",
          media_type: "application/pdf",
          data: pdfData,
        }},
        { type: "text", text: question },
      ],
    }],
  });

  return response.content[0].type === "text" ? response.content[0].text : "";
}

// const result = await analyzePdf("report.pdf", "总结这份报告的核心要点。");
```

### 图片分析的 Prompt 技巧

```typescript
// 差的 Prompt——太笼统
const bad = "分析这张图片";

// 好的 Prompt——指定分析维度
const good = `分析这张 UI 截图，从以下维度评估：
1. 布局合理性：元素排列是否符合阅读习惯
2. 视觉层次：信息优先级是否通过大小、颜色清晰传达
3. 交互可发现性：可点击元素是否一目了然
4. 一致性：字体、颜色、间距是否统一

每个维度打分（1-5）并给出改进建议。`;

// 提供上下文信息
const contextual = `这是一个电商 App 的商品详情页截图。
目标用户：25-35 岁年轻消费者。
请评估设计是否符合目标用户审美，指出可能降低转化率的问题。`;
```

::: tip 多模态注意事项
1. **图片大小**：支持最大约 20MB，建议压缩到合理大小
2. **支持格式**：JPEG、PNG、GIF、WebP
3. **Token 消耗**：图片会消耗 Token，分辨率越高消耗越多
4. **多图上限**：单次请求最多约 20 张
5. **PDF 页数**：过长文档建议分段处理
:::

## 统一适配层设计

在实际项目中，你往往需要同时支持多个 LLM 提供商。以下是一个简洁的统一适配层：

```typescript
import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";

enum Role {
  SYSTEM = "system",
  USER = "user",
  ASSISTANT = "assistant",
}

interface Message {
  role: Role;
  content: string;
}

interface LLMResponse {
  content: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  stopReason: string; // "complete" / "max_tokens" / "tool_use"
}

interface ChatOptions {
  maxTokens?: number;
  temperature?: number;
}

abstract class LLMProvider {
  abstract chat(messages: Message[], options?: ChatOptions): Promise<LLMResponse>;
}

class ClaudeProvider extends LLMProvider {
  private client: Anthropic;
  private model: string;

  constructor(model = "claude-sonnet-4-20250514") {
    super();
    this.client = new Anthropic();
    this.model = model;
  }

  async chat(messages: Message[], options: ChatOptions = {}): Promise<LLMResponse> {
    let system = "";
    const apiMsgs: { role: string; content: string }[] = [];
    for (const msg of messages) {
      if (msg.role === Role.SYSTEM) {
        system = msg.content;
      } else {
        apiMsgs.push({ role: msg.role, content: msg.content });
      }
    }

    const req: Record<string, unknown> = {
      model: this.model,
      max_tokens: options.maxTokens ?? 1024,
      messages: apiMsgs,
    };
    if (system) req.system = system;
    if (options.temperature !== undefined) req.temperature = options.temperature;

    const resp = await this.client.messages.create(req as Anthropic.MessageCreateParams);
    const stopMap: Record<string, string> = {
      end_turn: "complete",
      max_tokens: "max_tokens",
      tool_use: "tool_use",
    };
    return {
      content: resp.content[0]?.type === "text" ? resp.content[0].text : "",
      model: resp.model,
      inputTokens: resp.usage.input_tokens,
      outputTokens: resp.usage.output_tokens,
      stopReason: stopMap[resp.stop_reason] ?? resp.stop_reason,
    };
  }
}

class OpenAIProvider extends LLMProvider {
  private client: OpenAI;
  private model: string;

  constructor(model = "gpt-4o") {
    super();
    this.client = new OpenAI();
    this.model = model;
  }

  async chat(messages: Message[], options: ChatOptions = {}): Promise<LLMResponse> {
    const apiMsgs = messages.map((m) => ({ role: m.role as string, content: m.content }));
    const req: Record<string, unknown> = { model: this.model, messages: apiMsgs };
    if (options.maxTokens !== undefined) req.max_tokens = options.maxTokens;
    if (options.temperature !== undefined) req.temperature = options.temperature;

    const resp = await this.client.chat.completions.create(
      req as OpenAI.ChatCompletionCreateParamsNonStreaming
    );
    const choice = resp.choices[0];
    const stopMap: Record<string, string> = {
      stop: "complete",
      length: "max_tokens",
      tool_calls: "tool_use",
    };
    return {
      content: choice.message.content ?? "",
      model: resp.model,
      inputTokens: resp.usage?.prompt_tokens ?? 0,
      outputTokens: resp.usage?.completion_tokens ?? 0,
      stopReason: stopMap[choice.finish_reason ?? ""] ?? choice.finish_reason ?? "",
    };
  }
}

function createProvider(provider: string = "claude", model?: string): LLMProvider {
  const providers: Record<string, new (model?: string) => LLMProvider> = {
    claude: ClaudeProvider,
    openai: OpenAIProvider,
  };
  if (!(provider in providers)) {
    throw new Error(`不支持: ${provider}。可选: ${Object.keys(providers).join(", ")}`);
  }
  return new providers[provider](model);
}

// 使用——切换模型只需改一行
const messages: Message[] = [
  { role: Role.SYSTEM, content: "你是简洁的技术助手。" },
  { role: Role.USER, content: "什么是 RESTful API？一句话。" },
];

const claude = createProvider("claude");
const res = await claude.chat(messages, { maxTokens: 128 });
console.log(`[Claude] ${res.content}`);

// const openaiLlm = createProvider("openai");
// const res2 = await openaiLlm.chat(messages, { maxTokens: 128 });
// console.log(`[OpenAI] ${res2.content}`);
```

::: tip 适配层的价值
1. **切换无感知**：业务代码只依赖统一接口，换模型只改一行配置
2. **对比方便**：同一任务轻松在不同模型间对比效果
3. **降低锁定**：避免深度绑定单一供应商
4. **扩展容易**：新增模型只需添加一个 Provider class
:::

## 小结

1. **流式输出**基于 SSE 协议，将等待变为逐字呈现，大幅改善用户体验
2. **Claude** 使用 `client.messages.stream()` 配合事件监听，**OpenAI** 使用 `stream: true` 配合 `for await...of`
3. **前端消费**通过 `fetch + ReadableStream` 实现，需手动解析 SSE 格式
4. **多模态输入**支持图片（base64/URL）和 PDF，Prompt 技巧同样重要
5. **统一适配层**通过抽象类 + 适配器模式，让业务代码与 LLM 提供商解耦

## 练习

1. **流式练习**：实现一个流式"代码解释器"，用户粘贴代码，模型逐行解释。完成后统计 Token 和耗时。

2. **前后端练习**：用 Express + Vue 3 构建流式聊天应用，支持多轮对话、实时打字效果、Token 统计和"停止生成"按钮（提示：用 AbortController）。

3. **多模态练习**：构建一个"文档问答 Agent"，用户上传 PDF 后可以提问。支持多轮对话，首次分析后缓存摘要避免重复传入整个 PDF。

4. **适配层练习**：在统一适配层基础上，添加 `DeepSeekProvider`（基于 OpenAI 兼容接口），通过 `createProvider("deepseek")` 创建。

## 参考资源

- [Claude Streaming 文档](https://docs.anthropic.com/en/api/messages-streaming)
- [Claude Vision 文档](https://docs.anthropic.com/en/docs/build-with-claude/vision)
- [OpenAI Streaming 文档](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream)
- [MDN - Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
