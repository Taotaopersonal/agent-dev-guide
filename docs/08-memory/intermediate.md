# 记忆系统进阶

> **学习目标**：掌握短期记忆的摘要压缩策略，理解长期记忆的向量存储方案，实现用户画像渐进构建，设计多层记忆架构。

学完本节，你将能够：
- 实现对话摘要压缩（SummaryMemory）
- 实现 Buffer + Summary 混合记忆策略
- 用向量数据库构建长期语义记忆
- 用 KV 存储管理结构化的用户画像
- 设计短期 + 长期的多层记忆系统

## 对话摘要压缩

入门篇的滑动窗口会直接丢弃旧消息。更智能的做法是：**用 LLM 把旧消息压缩成摘要**，这样既节省了 token，又保留了关键信息。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface Message {
  role: string;
  content: string;
}

class SummaryMemory {
  /** 摘要记忆：压缩早期对话 */
  private maxMessages: number;
  private summaryThreshold: number;
  private messages: Message[] = [];
  private summary: string = "";

  constructor(maxMessages: number = 10, summaryThreshold: number = 8) {
    this.maxMessages = maxMessages;
    this.summaryThreshold = summaryThreshold;
  }

  add(role: string, content: string): void {
    this.messages.push({ role, content });
    if (this.messages.length > this.summaryThreshold) {
      this.compress();
    }
  }

  private async compress(): Promise<void> {
    /** 将前半部分消息压缩为摘要 */
    const nToCompress = Math.floor(this.messages.length / 2);
    const toCompress = this.messages.slice(0, nToCompress);
    this.messages = this.messages.slice(nToCompress);

    const conversation = toCompress
      .map((m) => `${m.role}: ${m.content}`)
      .join("\n");

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 512,
      messages: [
        {
          role: "user",
          content: `请将以下对话历史压缩为简洁的摘要。
保留关键信息：用户身份、核心需求、已达成的共识、重要决策。

${this.summary ? `之前的摘要：${this.summary}` : ""}

新对话：
${conversation}

请输出更新后的摘要：`,
        },
      ],
    });
    this.summary = (response.content[0] as { text: string }).text;
    console.log(`[摘要已更新] ${this.summary.slice(0, 100)}...`);
  }

  getMessages(): Message[] {
    /** 返回包含摘要的完整上下文 */
    const result: Message[] = [];
    if (this.summary) {
      result.push({
        role: "user",
        content: `[对话历史摘要] ${this.summary}`,
      });
      result.push({
        role: "assistant",
        content: "好的，我已了解之前的对话背景。请继续。",
      });
    }
    result.push(...this.messages);
    return result;
  }
}
```

摘要的好处很明显：10 轮对话可能需要 2000 tokens，压缩成摘要可能只需要 200 tokens，节省 90%。而且用户的名字、核心需求等关键信息都保留了。

::: warning 摘要的代价
每次压缩需要一次额外的 LLM API 调用。如果对话不长（< 10 轮），摘要压缩的收益可能不如成本。建议只在对话确实较长时才启用。
:::

## Buffer + Summary 混合策略

实际中最好用的方案：**最近的消息完整保留（Buffer），更早的消息压缩成摘要（Summary）**。

```typescript
class BufferSummaryMemory {
  /** Buffer + Summary 混合记忆 */
  private bufferSize: number;
  private maxTokens: number;
  private buffer: Message[] = []; // 最近的消息（完整保留）
  private summary: string = "";   // 早期消息的摘要
  private totalMessages: number = 0;

  constructor(bufferSize: number = 6, maxTokens: number = 2000) {
    this.bufferSize = bufferSize;
    this.maxTokens = maxTokens;
  }

  async add(role: string, content: string): Promise<void> {
    this.buffer.push({ role, content });
    this.totalMessages++;

    if (this.buffer.length > this.bufferSize) {
      const overflow = this.buffer.slice(0, -this.bufferSize);
      this.buffer = this.buffer.slice(-this.bufferSize);
      await this.updateSummary(overflow);
    }
  }

  private async updateSummary(messages: Message[]): Promise<void> {
    /** 将溢出的消息合并到摘要中 */
    const conversation = messages
      .map((m) => `${m.role}: ${m.content}`)
      .join("\n");

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 512,
      messages: [
        {
          role: "user",
          content: `将以下新对话内容整合到现有摘要中。

现有摘要：${this.summary || "（空）"}

新内容：
${conversation}

输出更新后的摘要（保持简洁，只保留重要信息）：`,
        },
      ],
    });
    this.summary = (response.content[0] as { text: string }).text;
  }

  getMessages(): Message[] {
    const result: Message[] = [];
    if (this.summary) {
      result.push({
        role: "user",
        content: `[对话摘要，共 ${this.totalMessages - this.buffer.length} 条消息] ${this.summary}`,
      });
      result.push({
        role: "assistant",
        content: "好的，我了解了之前的对话背景。",
      });
    }
    result.push(...this.buffer);
    return result;
  }

  stats(): Record<string, any> {
    return {
      totalMessages: this.totalMessages,
      bufferMessages: this.buffer.length,
      hasSummary: Boolean(this.summary),
    };
  }
}
```

使用示例：

```typescript
const memory = new BufferSummaryMemory(6);

const conversations: [string, string][] = [
  ["user", "你好，我是产品经理小王"],
  ["assistant", "你好小王！有什么可以帮你的？"],
  ["user", "我们在做一个电商推荐系统"],
  ["assistant", "好的，请告诉我具体需求"],
  ["user", "需要基于用户行为数据做个性化推荐"],
  ["assistant", "明白，这需要协同过滤或深度学习方法"],
  ["user", "预算有限，不想用太复杂的模型"],
  ["assistant", "那可以考虑基于物品的协同过滤"],
  // 到这里早期消息开始被压缩
  ["user", "我们目前有 10 万条数据"],
  ["assistant", "10 万条足够了，可以开始搭建"],
];

for (const [role, content] of conversations) {
  await memory.add(role, content);
}

console.log(`统计: ${JSON.stringify(memory.stats())}`);
// 早期消息已压缩为摘要，"小王"、"电商推荐"、"协同过滤"等关键信息都保留了
```

## 长期记忆：向量数据库存储

短期记忆只在单次对话中有效。用户关掉对话再回来，一切归零。长期记忆让 Agent 能够**跨会话**记住重要信息。

最常用的方案是向量数据库——把记忆编码为向量存储，检索时用语义匹配。

```typescript
/** 简单的内存向量存储（替代 chromadb） */

interface MemoryEntry {
  id: string;
  document: string;
  metadata: Record<string, any>;
  embedding: number[];
}

/** 简单的文本转向量（基于字符频率，仅用于演示） */
function simpleEmbed(text: string): number[] {
  const vec = new Array(128).fill(0);
  for (const ch of text) {
    vec[ch.charCodeAt(0) % 128] += 1;
  }
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
  return vec.map((v) => v / norm);
}

/** 余弦相似度 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

class VectorLongTermMemory {
  /** 基于内存向量存储的长期记忆 */
  private userId: string;
  private memories: MemoryEntry[] = [];
  private counter: number = 0;

  constructor(userId: string) {
    this.userId = userId;
  }

  store(content: string, memoryType: string = "general", importance: number = 0.5): void {
    /** 存储一条记忆 */
    this.counter++;
    this.memories.push({
      id: `mem_${this.counter}`,
      document: content,
      metadata: {
        type: memoryType,
        importance,
        timestamp: new Date().toISOString(),
        userId: this.userId,
      },
      embedding: simpleEmbed(content),
    });
  }

  recall(
    query: string,
    topK: number = 5,
    memoryType?: string
  ): Record<string, any>[] {
    /** 检索相关记忆 */
    const queryEmbedding = simpleEmbed(query);

    let candidates = this.memories.filter(
      (m) => m.metadata.userId === this.userId
    );
    if (memoryType) {
      candidates = candidates.filter((m) => m.metadata.type === memoryType);
    }

    const scored = candidates.map((m) => ({
      content: m.document,
      type: m.metadata.type,
      importance: m.metadata.importance,
      timestamp: m.metadata.timestamp,
      relevance: cosineSimilarity(queryEmbedding, m.embedding),
    }));

    scored.sort((a, b) => b.relevance - a.relevance);
    return scored.slice(0, topK);
  }

  count(): number {
    return this.memories.length;
  }
}
```

## 结构化记忆：KV 存储

对于确定性信息（用户名、偏好设置），用键值存储比向量数据库更合适——检索精确，不需要语义匹配。

```typescript
import * as fs from "fs";
import * as path from "path";

class KVMemory {
  /** 键值对长期记忆 */
  private userId: string;
  private filePath: string;
  private data: Record<string, any>;

  constructor(userId: string, storagePath: string = "./kv_memory") {
    this.userId = userId;
    this.filePath = path.join(storagePath, `${userId}.json`);
    this.data = this.load();
  }

  private load(): Record<string, any> {
    if (fs.existsSync(this.filePath)) {
      return JSON.parse(fs.readFileSync(this.filePath, "utf-8"));
    }
    return {};
  }

  private save(): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(this.data, null, 2));
  }

  set(key: string, value: any): void {
    this.data[key] = {
      value,
      updated_at: new Date().toISOString(),
    };
    this.save();
  }

  get(key: string, defaultValue: any = null): any {
    const entry = this.data[key];
    return entry ? entry.value : defaultValue;
  }

  getAll(): Record<string, any> {
    return Object.fromEntries(
      Object.entries(this.data).map(([k, v]) => [k, (v as any).value])
    );
  }
}
```

## 用户画像渐进构建

Agent 可以在每次对话中提取用户的新信息，逐步构建用户画像。

```typescript
interface ProfileData {
  name: string | null;
  occupation: string | null;
  interests: string[];
  preferences: Record<string, any>;
  interaction_count: number;
}

class UserProfile {
  /** 渐进式用户画像 */
  private kv: KVMemory;
  private profile: ProfileData;

  constructor(userId: string) {
    this.kv = new KVMemory(userId, "./user_profiles");
    this.profile = this.kv.get("profile", {
      name: null,
      occupation: null,
      interests: [],
      preferences: {},
      interaction_count: 0,
    });
  }

  async updateFromConversation(conversation: Message[]): Promise<void> {
    /** 从对话中更新用户画像 */
    const convText = conversation
      .map((m) => `${m.role}: ${m.content}`)
      .join("\n");

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 512,
      messages: [
        {
          role: "user",
          content: `基于以下对话，提取用户的个人信息来更新画像。

当前画像：${JSON.stringify(this.profile)}

新对话：
${convText}

只返回需要更新的字段（JSON），没有新信息则返回 {}。
可更新的字段：name, occupation, interests(数组), preferences(对象)`,
        },
      ],
    });

    const updates = JSON.parse(
      (response.content[0] as { text: string }).text
    );
    for (const [key, value] of Object.entries(updates)) {
      if (key === "interests" && Array.isArray(value)) {
        const existing = this.profile.interests || [];
        this.profile.interests = [...new Set([...existing, ...value])];
      } else if (key === "preferences" && typeof value === "object" && value !== null) {
        this.profile.preferences = { ...this.profile.preferences, ...(value as Record<string, any>) };
      } else {
        (this.profile as any)[key] = value;
      }
    }

    this.profile.interaction_count = (this.profile.interaction_count || 0) + 1;
    this.kv.set("profile", this.profile);
  }

  getSummary(): string {
    /** 生成用户画像摘要 */
    const p = this.profile;
    const parts: string[] = [];
    if (p.name) parts.push(`用户名: ${p.name}`);
    if (p.occupation) parts.push(`职业: ${p.occupation}`);
    if (p.interests?.length) parts.push(`兴趣: ${p.interests.join(", ")}`);
    if (p.preferences && Object.keys(p.preferences).length > 0) {
      const prefs = Object.entries(p.preferences)
        .map(([k, v]) => `${k}=${v}`)
        .join("; ");
      parts.push(`偏好: ${prefs}`);
    }
    parts.push(`交互次数: ${p.interaction_count || 0}`);
    return parts.length > 0 ? parts.join(" | ") : "新用户，暂无画像";
  }
}
```

## 重要信息提取器

不是所有对话内容都值得存入长期记忆。需要一个"重要性筛选器"。

```typescript
class MemoryExtractor {
  /** 从对话中提取值得记忆的信息 */

  async extract(conversation: Message[]): Promise<Record<string, any>[]> {
    const convText = conversation
      .map((m) => `${m.role}: ${m.content}`)
      .join("\n");

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      messages: [
        {
          role: "user",
          content: `从以下对话中提取值得长期记忆的信息。

对话：
${convText}

识别以下类型：
1. 用户个人信息（名字、职业、偏好）
2. 重要决策和结论
3. 用户的具体需求和目标
4. 有价值的反馈

返回 JSON 数组：
[{"content": "记忆内容", "type": "personal/decision/need/feedback", "importance": 0.1-1.0}]

没有值得记忆的信息就返回 []。`,
        },
      ],
    });
    return JSON.parse((response.content[0] as { text: string }).text);
  }
}
```

## 多层记忆系统

把短期记忆和长期记忆组合成一个完整的系统：

```typescript
class MultiLayerMemory {
  /** 多层记忆系统：短期 + 长期 */
  private shortTerm: BufferSummaryMemory;
  private longTerm: VectorLongTermMemory;
  private profile: UserProfile;
  private extractor: MemoryExtractor;

  constructor(userId: string) {
    this.shortTerm = new BufferSummaryMemory(8);
    this.longTerm = new VectorLongTermMemory(userId);
    this.profile = new UserProfile(userId);
    this.extractor = new MemoryExtractor();
  }

  beforeResponse(userInput: string): Record<string, any> {
    /** 在生成回复前，收集所有相关记忆 */
    // 从长期记忆中检索相关信息
    const relevantMemories = this.longTerm.recall(userInput, 3);
    const memoryContext = relevantMemories
      .map((m) => m.content)
      .join("\n");

    return {
      messages: this.shortTerm.getMessages(),
      userProfile: this.profile.getSummary(),
      relevantMemories: memoryContext,
    };
  }

  async afterResponse(userInput: string, assistantReply: string): Promise<void> {
    /** 在生成回复后，更新所有记忆层 */
    // 更新短期记忆
    await this.shortTerm.add("user", userInput);
    await this.shortTerm.add("assistant", assistantReply);

    // 提取重要信息存入长期记忆
    const recent: Message[] = [
      { role: "user", content: userInput },
      { role: "assistant", content: assistantReply },
    ];
    const memories = await this.extractor.extract(recent);
    for (const mem of memories) {
      this.longTerm.store(
        mem.content,
        mem.type,
        mem.importance
      );
    }

    // 更新用户画像
    await this.profile.updateFromConversation(recent);
  }
}
```

::: tip 各层记忆的职责
| 层级 | 存储内容 | 检索方式 | 生命周期 |
|------|---------|---------|---------|
| 短期（Buffer） | 最近几条消息 | 直接包含在 prompt 中 | 当前对话 |
| 短期（Summary） | 早期对话的摘要 | 作为系统提示 | 当前对话 |
| 长期（向量） | 重要的语义记忆 | 按语义相似度检索 | 永久 |
| 长期（KV） | 用户画像、偏好 | 按键精确查找 | 永久 |
:::

## 小结

- **摘要压缩**：用 LLM 把旧消息压缩为摘要，节省 token 同时保留关键信息
- **Buffer + Summary**：最近消息完整保留，更早的压缩为摘要，是实践中最好用的方案
- **向量长期记忆**：语义记忆用 ChromaDB 存储，按相似度检索
- **KV 长期记忆**：结构化信息用 JSON 文件存储，精确查找
- **用户画像**：每次对话提取新信息，增量更新
- **多层架构**：短期 + 长期组合，before_response 收集、after_response 更新

## 练习

1. **Buffer + Summary 实验**：模拟 30 轮对话，观察摘要的压缩质量。用户在第 1 轮说的名字是否在第 30 轮还能被"记住"？
2. **长期记忆**：实现一个跨会话的 Agent——第一次对话时用户说"我是 Python 工程师"，第二次对话时 Agent 能主动提起这个信息。
3. **记忆整合**：当长期记忆超过 100 条时，实现一个自动合并相似记忆的功能（提示：用向量相似度找相似记忆，用 LLM 合并）。

## 参考资源

- [LangChain: Memory Types](https://python.langchain.com/docs/concepts/memory/) -- LangChain 记忆类型文档
- [MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) -- MemGPT 虚拟内存管理论文
- [Mem0](https://github.com/mem0ai/mem0) -- 开源的 Agent 记忆层
- [Generative Agents (arXiv:2304.03442)](https://arxiv.org/abs/2304.03442) -- Stanford 生成式 Agent 记忆系统
