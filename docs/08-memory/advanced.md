# 记忆高级：前沿记忆架构

> **学习目标**：理解 MemGPT 架构、时间衰减机制、记忆整合与冲突解决，学习 Generative Agents 的记忆设计，构建自主记忆管理 Agent。

学完本节，你将能够：
- 理解 MemGPT 的虚拟内存管理思想
- 实现时间衰减的记忆权重机制
- 处理记忆之间的冲突和矛盾
- 理解 Generative Agents 的反思和记忆检索机制
- 实现记忆蒸馏和压缩
- 构建一个能自主管理记忆的 Agent

## MemGPT 架构

MemGPT 的核心灵感来自操作系统的虚拟内存：就像 OS 把不常用的数据从内存换到磁盘，MemGPT 把不常用的上下文从 LLM 的上下文窗口"换出"到外部存储。

```
传统 LLM:  [固定的上下文窗口] -- 装满了就没办法了
MemGPT:    [主上下文] <-> [外部存储] -- LLM 自己决定换入换出什么
```

关键创新：**LLM 自己管理自己的记忆**。它有特殊的"记忆管理工具"，可以主动把信息存入外部、从外部检索信息、修改已存储的信息。

```typescript
/** MemGPT 风格的记忆管理 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

/** 简单的内存向量存储（替代 chromadb） */
interface VectorDoc {
  id: string;
  document: string;
  metadata: Record<string, any>;
  embedding: number[];
}

function simpleEmbed(text: string): number[] {
  const vec = new Array(128).fill(0);
  for (const ch of text) {
    vec[ch.charCodeAt(0) % 128] += 1;
  }
  const norm = Math.sqrt(vec.reduce((s: number, v: number) => s + v * v, 0)) || 1;
  return vec.map((v: number) => v / norm);
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

class SimpleVectorStore {
  private docs: VectorDoc[] = [];

  add(documents: string[], ids: string[], metadatas: Record<string, any>[]): void {
    for (let i = 0; i < documents.length; i++) {
      this.docs.push({
        id: ids[i],
        document: documents[i],
        metadata: metadatas[i],
        embedding: simpleEmbed(documents[i]),
      });
    }
  }

  query(queryTexts: string[], nResults: number = 3): { documents: string[][]; metadatas: Record<string, any>[][] } {
    const queryEmb = simpleEmbed(queryTexts[0]);
    const scored = this.docs
      .map((doc) => ({ doc, score: cosineSimilarity(queryEmb, doc.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, nResults);
    return {
      documents: [scored.map((s) => s.doc.document)],
      metadatas: [scored.map((s) => s.doc.metadata)],
    };
  }
}


class MemGPTAgent {
  /** MemGPT 风格的 Agent：LLM 自主管理记忆 */

  // 核心记忆（始终在上下文中）
  private coreMemory: Record<string, string> = {
    user_info: "",    // 用户基本信息
    agent_info: "我是一个有长期记忆的AI助手。",
    preferences: "",  // 用户偏好
  };
  // 归档记忆（外部存储，按需检索）
  private archive = new SimpleVectorStore();
  private archiveCount = 0;

  // 对话缓冲（短期记忆，有限大小）
  private buffer: Record<string, any>[] = [];
  private bufferLimit = 10;

  // 记忆管理工具
  private tools = [
    {
      name: "core_memory_update",
      description:
        "更新核心记忆中的信息。核心记忆始终可见。" +
        "当用户透露重要的个人信息、偏好变化时使用。",
      input_schema: {
        type: "object" as const,
        properties: {
          section: {
            type: "string" as const,
            enum: ["user_info", "preferences"],
            description: "要更新的记忆区域",
          },
          content: {
            type: "string" as const,
            description: "新的内容（会追加到现有内容）",
          },
        },
        required: ["section", "content"],
      },
    },
    {
      name: "archive_memory_store",
      description:
        "将信息存入归档记忆。归档记忆容量大但不在上下文中。" +
        "存储对话细节、过去的讨论结论等。",
      input_schema: {
        type: "object" as const,
        properties: {
          content: {
            type: "string" as const,
            description: "要归档的内容",
          },
          tags: {
            type: "string" as const,
            description: "标签，用逗号分隔",
          },
        },
        required: ["content"],
      },
    },
    {
      name: "archive_memory_search",
      description: "搜索归档记忆。当需要回忆过去的对话或信息时使用。",
      input_schema: {
        type: "object" as const,
        properties: {
          query: {
            type: "string" as const,
            description: "搜索关键词",
          },
        },
        required: ["query"],
      },
    },
  ];

  private buildSystemPrompt(): string {
    /** 构建包含核心记忆的 system prompt */
    return `你是一个有记忆能力的AI助手。

=== 核心记忆（始终可见）===
用户信息：${this.coreMemory.user_info || "暂无"}
用户偏好：${this.coreMemory.preferences || "暂无"}
关于你自己：${this.coreMemory.agent_info}

=== 规则 ===
1. 当用户透露新的个人信息时，用 core_memory_update 保存
2. 当讨论产生重要结论时，用 archive_memory_store 归档
3. 当需要回忆过去的对话时，用 archive_memory_search 搜索
4. 自然地使用你对用户的了解来个性化回答`;
  }

  private executeTool(name: string, params: Record<string, any>): string {
    if (name === "core_memory_update") {
      const section = params.section as string;
      const content = params.content as string;
      if (this.coreMemory[section]) {
        this.coreMemory[section] += `\n${content}`;
      } else {
        this.coreMemory[section] = content;
      }
      return JSON.stringify({ status: "updated", section });
    }

    if (name === "archive_memory_store") {
      this.archiveCount++;
      this.archive.add(
        [params.content],
        [`archive_${this.archiveCount}`],
        [{ tags: params.tags ?? "", timestamp: new Date().toISOString() }]
      );
      return JSON.stringify({ status: "archived", id: this.archiveCount });
    }

    if (name === "archive_memory_search") {
      const results = this.archive.query([params.query], 3);
      const memories = results.documents[0] ?? [];
      return JSON.stringify({ results: memories });
    }

    return JSON.stringify({ error: "unknown tool" });
  }

  async chat(userInput: string): Promise<string> {
    /** 对话（LLM 会自主管理记忆） */
    this.buffer.push({ role: "user", content: userInput });

    // 缓冲区满时裁剪
    if (this.buffer.length > this.bufferLimit) {
      this.buffer = this.buffer.slice(-this.bufferLimit);
      while (this.buffer.length > 0 && this.buffer[0].role !== "user") {
        this.buffer.shift();
      }
    }

    for (let i = 0; i < 5; i++) {
      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        system: this.buildSystemPrompt(),
        tools: this.tools,
        messages: this.buffer,
      });

      if (response.stop_reason === "end_turn") {
        const text = response.content
          .filter((b: any) => b.type === "text")
          .map((b: any) => b.text)
          .join("");
        this.buffer.push({ role: "assistant", content: response.content });
        return text;
      }

      if (response.stop_reason === "tool_use") {
        this.buffer.push({ role: "assistant", content: response.content });
        const results: any[] = [];
        for (const block of response.content) {
          if (block.type === "tool_use") {
            const result = this.executeTool(block.name, block.input as Record<string, any>);
            console.log(`  [记忆操作] ${block.name}: ${JSON.stringify(block.input).slice(0, 60)}`);
            results.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: result,
            });
          }
        }
        this.buffer.push({ role: "user", content: results });
      }
    }

    return "处理超时";
  }
}
```

## 时间衰减机制

人会遗忘 -- 越久远的记忆越模糊。给 Agent 加上类似的机制：

```typescript
/** 时间衰减记忆 */

interface DecayingMemoryEntry {
  content: string;
  baseImportance: number;   // 初始重要性
  createdAt: Date;
  accessCount: number;
  lastAccessed: Date;
}

class DecayingMemory {
  /** 带时间衰减的记忆系统 */
  private memories: DecayingMemoryEntry[] = [];
  private halfLife: number;

  /**
   * @param halfLifeHours 记忆"半衰期"，单位小时。
   * 24小时后重要性减半，48小时后减为1/4...
   */
  constructor(halfLifeHours: number = 24.0) {
    this.halfLife = halfLifeHours;
  }

  store(content: string, importance: number = 1.0): void {
    /** 存入记忆 */
    const now = new Date();
    this.memories.push({
      content,
      baseImportance: importance,
      createdAt: now,
      accessCount: 0,
      lastAccessed: now,
    });
  }

  private calculateWeight(memory: DecayingMemoryEntry): number {
    /** 计算记忆的当前权重（考虑时间衰减） */
    const hoursPassed =
      (new Date().getTime() - memory.createdAt.getTime()) / (1000 * 3600);

    // 指数衰减
    const decay = Math.pow(0.5, hoursPassed / this.halfLife);

    // 被访问次数的加成（每次访问延缓衰减）
    const accessBoost = 1 + 0.1 * memory.accessCount;

    return memory.baseImportance * decay * accessBoost;
  }

  recall(query?: string, topK: number = 5): (DecayingMemoryEntry & { currentWeight: number; finalScore: number })[] {
    /** 检索记忆（按权重排序） */
    const weighted: (DecayingMemoryEntry & { currentWeight: number; finalScore: number })[] = [];

    for (const mem of this.memories) {
      const weight = this.calculateWeight(mem);
      // 简单的关键词匹配加成
      let relevance = 1.0;
      if (query) {
        const queryWords = new Set(query.toLowerCase().split(/\s+/));
        const memWords = new Set(mem.content.toLowerCase().split(/\s+/));
        let overlap = 0;
        for (const w of queryWords) {
          if (memWords.has(w)) overlap++;
        }
        relevance = 1.0 + overlap * 0.5;
      }

      const finalScore = weight * relevance;
      weighted.push({ ...mem, currentWeight: weight, finalScore });
    }

    // 按分数排序
    weighted.sort((a, b) => b.finalScore - a.finalScore);

    // 更新访问次数
    for (const mem of weighted.slice(0, topK)) {
      for (const original of this.memories) {
        if (original.content === mem.content) {
          original.accessCount++;
          original.lastAccessed = new Date();
        }
      }
    }

    return weighted.slice(0, topK);
  }

  forget(threshold: number = 0.01): void {
    /** 遗忘权重低于阈值的记忆 */
    const before = this.memories.length;
    this.memories = this.memories.filter(
      (m) => this.calculateWeight(m) >= threshold
    );
    const forgotten = before - this.memories.length;
    if (forgotten) {
      console.log(`[遗忘] 清除了 ${forgotten} 条低权重记忆`);
    }
  }
}

// 使用
const memory = new DecayingMemory(24);
memory.store("用户名叫小明", 2.0);  // 重要信息，高权重
memory.store("今天讨论了天气", 0.5);  // 不重要的闲聊
memory.store("用户决定使用 FastAPI", 1.5);

for (const mem of memory.recall(undefined, 3)) {
  console.log(`  [${mem.currentWeight.toFixed(2)}] ${mem.content}`);
}
```

## 记忆整合与冲突解决

当新记忆和旧记忆矛盾时（用户说"我改用 Rust 了"，但之前记的是"用户用 Java"），需要处理冲突：

```typescript
/** 记忆冲突检测和解决 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();


class MemoryConflictResolver {
  /** 记忆冲突检测和解决 */

  async detectConflict(
    newMemory: string,
    existingMemories: string[]
  ): Promise<Record<string, any>> {
    /** 检测新记忆是否与已有记忆冲突 */
    const existingText = existingMemories.map((m) => `- ${m}`).join("\n");

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 200,
      messages: [
        {
          role: "user",
          content: `检测新信息是否与已有信息冲突。

已有信息：
${existingText}

新信息：${newMemory}

返回 JSON：
{
    "has_conflict": true/false,
    "conflicting_with": "冲突的已有信息",
    "resolution": "update"(用新的替换旧的) / "merge"(合并) / "keep_both"(都保留)
    "resolved_content": "解决后的内容"
}`,
        },
      ],
    });

    let text = (response.content[0] as { text: string }).text.trim();
    try {
      if (text.includes("```")) {
        text = text.split("```")[1].replace("json", "").trim();
      }
      return JSON.parse(text);
    } catch {
      return { has_conflict: false };
    }
  }

  async resolve(
    newMemory: string,
    existingMemories: string[]
  ): Promise<string[]> {
    /** 解决冲突并返回更新后的记忆列表 */
    const result = await this.detectConflict(newMemory, existingMemories);

    if (!result.has_conflict) {
      return [...existingMemories, newMemory];
    }

    console.log(`[冲突检测] 发现冲突: ${result.conflicting_with ?? ""}`);
    console.log(`[解决策略] ${result.resolution ?? "keep_both"}`);

    const resolution = result.resolution ?? "keep_both";
    const resolved = result.resolved_content ?? newMemory;

    if (resolution === "update") {
      // 替换冲突的旧记忆
      return existingMemories.map((mem) =>
        mem === result.conflicting_with ? resolved : mem
      );
    }

    if (resolution === "merge") {
      // 合并为一条
      const updated = existingMemories.filter(
        (m) => m !== result.conflicting_with
      );
      updated.push(resolved);
      return updated;
    }

    // keep_both
    return [...existingMemories, newMemory];
  }
}

// 使用
const resolver = new MemoryConflictResolver();
const memories = ["用户主要使用 Java 编程", "用户在杭州工作"];
const newInfo = "用户最近转向了 Rust 开发";

const updated = await resolver.resolve(newInfo, memories);
console.log("更新后的记忆:");
for (const m of updated) {
  console.log(`  - ${m}`);
}
```

## 记忆蒸馏

当长期记忆积累太多时，把大量具体记忆"蒸馏"成少数高级概括：

```typescript
/** 记忆蒸馏：把具体记忆压缩成高级概括 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();


async function distillMemories(
  memories: string[],
  maxOutput: number = 5
): Promise<string[]> {
  /** 将大量具体记忆蒸馏为少量高级概括 */
  const memoryText = memories.map((m) => `- ${m}`).join("\n");

  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 500,
    messages: [
      {
        role: "user",
        content: `将以下 ${memories.length} 条具体记忆蒸馏为 ${maxOutput} 条高级概括。

要求：
1. 合并相似的记忆
2. 保留最重要的信息
3. 用概括性的语言，不是简单拼接
4. 标注信息的时间特征（如"长期偏好" vs "最近变化"）

具体记忆：
${memoryText}

返回 JSON 数组。`,
      },
    ],
  });

  let text = (response.content[0] as { text: string }).text.trim();
  try {
    if (text.includes("```")) {
      text = text.split("```")[1].replace("json", "").trim();
    }
    return JSON.parse(text);
  } catch {
    return memories.slice(0, maxOutput);
  }
}

// 使用
const rawMemories = [
  "2024-03-15: 用户问了 Python 装饰器的用法",
  "2024-03-16: 用户在写一个 Flask 项目",
  "2024-03-20: 用户遇到了 CORS 跨域问题",
  "2024-04-01: 用户开始学习 FastAPI",
  "2024-04-05: 用户把 Flask 项目迁移到了 FastAPI",
  "2024-04-10: 用户问了 Pydantic 的 validator 用法",
  "2024-04-15: 用户在部署 FastAPI 到 Docker",
  "2024-04-20: 用户配置了 Nginx 反向代理",
];

const distilled = await distillMemories(rawMemories, 3);
console.log("蒸馏后的记忆:");
for (const m of distilled) {
  console.log(`  - ${m}`);
}
```

## Generative Agents 记忆架构

### 论文简介

2023 年，斯坦福大学和 Google Research 发表了论文 *"Generative Agents: Interactive Simulacra of Human Behavior"*（arXiv:2304.03442）。研究者在一个类似"The Sims"的虚拟小镇中放入 25 个由 LLM 驱动的 AI 居民，让它们自主生活、社交、工作。这些 Agent 能记住过去的经历、形成对彼此的看法、协调计划（比如自发组织一场派对），展现出了惊人的"类人"行为。

这一切的基础，是论文提出的**三层记忆架构**：

```
┌───────────────────────────────────────────────┐
│                  规划 (Planning)                │  ← 基于记忆和反思生成行动计划
├───────────────────────────────────────────────┤
│                反思 (Reflection)                │  ← 定期从记忆中提炼高层洞察
├───────────────────────────────────────────────┤
│              记忆流 (Memory Stream)             │  ← 按时间顺序记录所有经历
└───────────────────────────────────────────────┘
```

### 三层架构详解

**1. 记忆流（Memory Stream）**

记忆流是最底层的数据结构，按时间顺序记录 Agent 的所有经历——观察到的事件、自己的行为、与他人的对话等。每条记忆包含：

- `description`：自然语言描述（如"John 在咖啡馆和 Maria 聊了关于画展的话题"）
- `created_at`：创建时间戳
- `importance`：重要性评分（1-10，由 LLM 评估）
- `last_accessed`：最近一次被检索的时间

**2. 反思（Reflection）**

当记忆积累到一定量（论文中以重要性分数之和超过阈值为触发条件），Agent 会进行"反思"——从近期的具体记忆中抽象出高层次洞察。例如：

- 具体记忆："Maria 在画展上花了3小时"、"Maria 跟我说她最近在学油画"、"Maria 的房间里挂满了画"
- 反思结论："Maria 对艺术非常热情，尤其是绘画"

反思本身也会存入记忆流（标记为 reflection 类型），可以被后续检索引用，甚至可以基于反思再反思，形成越来越抽象的认知。

**3. 规划（Planning）**

Agent 根据记忆流和反思结果，生成从粗到细的行动计划。先生成一天的大致安排（"上午去咖啡馆工作，下午参加画展"），再递归细化到具体动作（"9:00 出门，9:15 到达咖啡馆，点一杯美式……"）。计划执行过程中如果遇到意外事件（比如路上碰到朋友），Agent 会根据记忆和反思决定是否调整计划。

### 记忆检索：三维评分

Generative Agents 检索记忆时，不是简单地按时间排序或关键词匹配，而是综合三个维度打分：

$$\text{score} = \alpha \cdot \text{recency} + \beta \cdot \text{importance} + \gamma \cdot \text{relevance}$$

- **时效性（Recency）**：指数衰减函数，最近的记忆得分更高。论文使用衰减因子 0.995^{小时数}
- **重要性（Importance）**：记忆创建时由 LLM 评分（1-10），"吃了早饭"得 1 分，"求婚被接受"得 10 分
- **相关性（Relevance）**：当前情境与记忆内容的语义相似度，通过 embedding 余弦相似度计算

三个维度分别归一化到 [0, 1] 后加权求和，论文中 alpha = beta = gamma = 1（等权重）。

### 代码实现

下面是一个教学级别的简化实现，包含记忆流、三维评分检索和反思机制：

```typescript
/**
 * Generative Agents 记忆架构简化实现
 * 参考论文：Generative Agents: Interactive Simulacra of Human Behavior (2304.03442)
 *
 * 实现了三个核心机制：
 * 1. 记忆流 (Memory Stream) — 按时间记录所有经历
 * 2. 三维评分检索 — Recency x Importance x Relevance
 * 3. 反思 (Reflection) — 从具体记忆中抽象出高层洞察
 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();


// ─── 数据结构 ───────────────────────────────────────────

interface MemoryRecord {
  /** 记忆流中的单条记忆 */
  description: string;                  // 自然语言描述
  createdAt: Date;                      // 创建时间
  importance: number;                   // 重要性评分 (1-10)
  lastAccessed: Date;                   // 最近访问时间
  memoryType: "observation" | "reflection" | "plan";  // 记忆类型
}


// ─── 记忆流核心类 ──────────────────────────────────────

class MemoryStream {
  /**
   * Generative Agents 的记忆流实现。
   *
   * 核心能力：
   * - 存储所有经历（observations）
   * - 三维评分检索（recency x importance x relevance）
   * - 触发反思（reflection）生成高层洞察
   */
  private memories: MemoryRecord[] = [];
  private recencyDecay: number;
  private reflectionThreshold: number;
  private alpha: number;
  private beta: number;
  private gamma: number;

  // 自上次反思以来的重要性累积值
  private importanceAccumulator: number = 0;

  constructor(
    recencyDecay: number = 0.995,       // 时效性衰减因子（每小时）
    reflectionThreshold: number = 50.0, // 触发反思的重要性累积阈值
    alpha: number = 1.0,                // 时效性权重
    beta: number = 1.0,                 // 重要性权重
    gamma: number = 1.0,               // 相关性权重
  ) {
    this.recencyDecay = recencyDecay;
    this.reflectionThreshold = reflectionThreshold;
    this.alpha = alpha;
    this.beta = beta;
    this.gamma = gamma;
  }

  // ─── 重要性评分（由 LLM 打分）───────────────────

  private async rateImportance(description: string): Promise<number> {
    /** 让 LLM 对一条记忆的重要性评分 (1-10) */
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 20,
      messages: [
        {
          role: "user",
          content:
            `请对以下事件的重要性评分（1分=吃早饭等日常琐事，` +
            `10分=求婚、重大人生转折）。只返回一个整数。\n\n` +
            `事件：${description}`,
        },
      ],
    });
    const text = (response.content[0] as { text: string }).text.trim();
    try {
      const score = parseFloat(text);
      return Math.max(1.0, Math.min(10.0, score));
    } catch {
      return 5.0; // 解析失败时给默认中等评分
    }
  }

  // ─── 存入记忆 ──────────────────────────────────

  async addObservation(
    description: string,
    timestamp?: Date
  ): Promise<MemoryRecord> {
    /**
     * 向记忆流中添加一条观察记忆。
     * 添加后检查是否需要触发反思。
     */
    const ts = timestamp ?? new Date();
    const importance = await this.rateImportance(description);

    const record: MemoryRecord = {
      description,
      createdAt: ts,
      importance,
      lastAccessed: ts,
      memoryType: "observation",
    };
    this.memories.push(record);
    this.importanceAccumulator += importance;

    console.log(`  [记忆+] (重要性=${importance.toFixed(0)}) ${description}`);

    // 累积重要性超过阈值时触发反思
    if (this.importanceAccumulator >= this.reflectionThreshold) {
      await this.reflect();
    }

    return record;
  }

  // ─── 三维评分检索 ──────────────────────────────

  retrieve(
    query: string,
    topK: number = 5,
    now?: Date
  ): { memory: MemoryRecord; score: number; recency: number; importance: number; relevance: number }[] {
    /**
     * 三维评分检索：
     * score = alpha * recency + beta * importance + gamma * relevance
     *
     * 各维度先归一化到 [0, 1]，再加权求和。
     */
    if (this.memories.length === 0) return [];

    const currentTime = now ?? new Date();

    // --- 1. 计算各维度原始分 ---
    const rawRecency: number[] = [];
    const rawImportance: number[] = [];
    const rawRelevance: number[] = [];

    for (const mem of this.memories) {
      // 时效性：指数衰减
      const hoursPassed =
        (currentTime.getTime() - mem.createdAt.getTime()) / (1000 * 3600);
      rawRecency.push(Math.pow(this.recencyDecay, hoursPassed));

      // 重要性：直接用评分
      rawImportance.push(mem.importance);

      // 相关性：简化实现 — 用关键词重叠度模拟语义相似度
      // 生产环境应使用 embedding 余弦相似度
      rawRelevance.push(this.computeRelevance(query, mem.description));
    }

    // --- 2. Min-Max 归一化到 [0, 1] ---
    function normalize(values: number[]): number[] {
      const minV = Math.min(...values);
      const maxV = Math.max(...values);
      if (maxV - minV < 1e-9) return values.map(() => 1.0);
      return values.map((v) => (v - minV) / (maxV - minV));
    }

    const normRecency = normalize(rawRecency);
    const normImportance = normalize(rawImportance);
    const normRelevance = normalize(rawRelevance);

    // --- 3. 加权求和 ---
    const scored = this.memories.map((mem, i) => {
      const score =
        this.alpha * normRecency[i] +
        this.beta * normImportance[i] +
        this.gamma * normRelevance[i];
      return {
        memory: mem,
        score,
        recency: normRecency[i],
        importance: normImportance[i],
        relevance: normRelevance[i],
      };
    });

    // 按总分降序排序
    scored.sort((a, b) => b.score - a.score);

    // 更新被检索记忆的 lastAccessed
    for (const item of scored.slice(0, topK)) {
      item.memory.lastAccessed = currentTime;
    }

    return scored.slice(0, topK);
  }

  private computeRelevance(query: string, description: string): number {
    /**
     * 计算查询与记忆的相关性（简化版：关键词重叠）。
     *
     * 生产环境建议替换为：
     *   const queryEmb = getEmbedding(query);
     *   const memEmb = getEmbedding(description);
     *   return cosineSimilarity(queryEmb, memEmb);
     */
    const tokenize = (s: string) =>
      new Set(s.toLowerCase().replace(/[，。]/g, " ").split(/\s+/).filter(Boolean));
    const queryTokens = tokenize(query);
    const memTokens = tokenize(description);
    if (queryTokens.size === 0) return 0.0;
    let overlap = 0;
    for (const t of queryTokens) {
      if (memTokens.has(t)) overlap++;
    }
    return overlap / queryTokens.size;
  }

  // ─── 反思机制 ──────────────────────────────────

  private async reflect(): Promise<void> {
    /**
     * 反思：从近期记忆中提炼高层次洞察。
     *
     * 流程：
     * 1. 收集最近的记忆（按重要性排序取 top-20）
     * 2. 让 LLM 从中抽象出 2-3 条高层洞察
     * 3. 将洞察作为 reflection 类型存入记忆流
     */
    console.log("\n  === 触发反思 ===");

    // 取最近的记忆，按重要性排序
    const recent = [...this.memories]
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 20);

    const memoryText = recent
      .map((m) => {
        const timeStr = `${m.createdAt.getHours().toString().padStart(2, "0")}:${m.createdAt.getMinutes().toString().padStart(2, "0")}`;
        return `  [${timeStr}] (重要性=${m.importance.toFixed(0)}) ${m.description}`;
      })
      .join("\n");

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 500,
      messages: [
        {
          role: "user",
          content: `基于以下记忆，提炼出 2-3 条高层次洞察。
洞察应该是对具体事件的抽象概括，揭示模式、关系或性格特征。

记忆列表：
${memoryText}

以 JSON 数组返回，每条洞察是一个字符串。示例：
["Maria 对艺术充满热情，尤其是绘画", "John 和 Maria 正在发展友谊"]`,
        },
      ],
    });

    let text = (response.content[0] as { text: string }).text.trim();
    let insights: string[];
    try {
      if (text.includes("```")) {
        text = text.split("```")[1].replace("json", "").trim();
      }
      insights = JSON.parse(text);
    } catch {
      insights = [text];
    }

    // 将反思结果存入记忆流（类型标记为 reflection）
    for (const insight of insights) {
      const record: MemoryRecord = {
        description: `[反思] ${insight}`,
        createdAt: new Date(),
        importance: 8.0, // 反思结果通常比具体记忆更重要
        lastAccessed: new Date(),
        memoryType: "reflection",
      };
      this.memories.push(record);
      console.log(`  [反思结果] ${insight}`);
    }

    // 重置累积器
    this.importanceAccumulator = 0;
    console.log("  === 反思完成 ===\n");
  }
}


// ─── 使用示例 ──────────────────────────────────────────

async function main() {
  const stream = new MemoryStream(0.995, 30.0);

  // 模拟一个 Agent 一天的经历（手动指定时间戳以演示时效性）
  const baseTime = new Date(2026, 3, 16, 8, 0, 0); // 月份从 0 开始

  const observations: [number, string][] = [
    [0, "早上在公园散步，遇到了邻居 Maria"],
    [1, "Maria 提到她正在准备下周的画展"],
    [2, "去咖啡馆工作，写了两个小时的报告"],
    [3, "午饭在食堂吃了意面"],
    [4, "下午和同事 John 讨论了新项目方案，他对 AI 方向很感兴趣"],
    [5, "John 邀请我周六去他家的烧烤派对"],
    [6, "收到邮件，下个月公司要重组部门"],
    [8, "晚上去了 Maria 的画室，看到她最新的油画作品非常出色"],
    [9, "Maria 说她考虑辞职全职画画"],
    [10, "回家路上想到 John 和 Maria 都对创意工作感兴趣，也许可以介绍他们认识"],
  ];

  // 依次添加观察记忆
  for (const [hourOffset, desc] of observations) {
    const ts = new Date(baseTime.getTime() + hourOffset * 3600 * 1000);
    await stream.addObservation(desc, ts);
  }

  // --- 三维评分检索演示 ---
  console.log("\n" + "=".repeat(60));
  console.log("检索: 'Maria 的兴趣爱好'\n");

  const results = stream.retrieve(
    "Maria 的兴趣爱好",
    5,
    new Date(baseTime.getTime() + 12 * 3600 * 1000)
  );

  for (let i = 0; i < results.length; i++) {
    const item = results[i];
    const mem = item.memory;
    console.log(
      `  #${i + 1} [总分=${item.score.toFixed(2)}] ` +
      `(时效=${item.recency.toFixed(2)} ` +
      `重要=${item.importance.toFixed(2)} ` +
      `相关=${item.relevance.toFixed(2)}) ` +
      `${mem.memoryType === "reflection" ? "[反思]" : ""}` +
      `${mem.description}`
    );
  }
}

main();
```

### 与 MemGPT 的对比

Generative Agents 和 MemGPT 都是 2023 年提出的经典记忆架构，但设计哲学不同：

| 维度 | Generative Agents | MemGPT |
|------|-------------------|--------|
| **核心隐喻** | 人类认知心理学（记忆、反思、计划） | 操作系统虚拟内存（换页、缓存） |
| **记忆组织** | 扁平的记忆流 + 反思层级 | 分层存储（核心/归档/对话缓冲） |
| **检索机制** | 三维评分（时效 x 重要 x 相关） | LLM 自主决定何时搜索 |
| **写入机制** | 自动记录所有观察 | LLM 自主决定何时存储 |
| **记忆演化** | 反思机制生成高层洞察 | 无内置抽象机制 |
| **适用场景** | 多 Agent 仿真、NPC、社交模拟 | 长对话助手、个人 AI 伴侣 |
| **上下文管理** | 检索相关记忆注入 prompt | 主动换入换出上下文 |

**选型建议**：

- 如果你在构建**交互式 NPC** 或**多 Agent 模拟**，Agent 需要自主生活、形成社交关系，Generative Agents 的反思 + 规划架构更合适
- 如果你在构建**长期对话助手**，需要在有限上下文窗口内管理大量历史信息，MemGPT 的主动记忆管理更实用
- 两者可以**组合使用**：用 MemGPT 的分层存储 + 工具化管理作为基础设施，在其上叠加 Generative Agents 的三维检索和反思机制

## 实战：自主记忆管理 Agent

综合所有高级概念，构建一个能自主管理记忆的 Agent：

```typescript
/** 自主记忆管理 Agent */
import Anthropic from "@anthropic-ai/sdk";
import * as readline from "readline";

const client = new Anthropic();


class AutonomousMemoryAgent {
  /** 自主记忆管理 Agent -- 综合 MemGPT + 时间衰减 + 冲突解决 */

  private core: Record<string, string> = {
    user_profile: "",
    preferences: "",
    context: "",
  };
  private workingMemory: Record<string, any>[] = [];
  private longTerm: { content: string; importance: number; timestamp: string }[] = [];
  private maxWorking = 15;

  // Agent 有记忆管理工具
  private tools = [
    {
      name: "remember",
      description: "将重要信息存入长期记忆。用户偏好、决定、关键事实应该被记住。",
      input_schema: {
        type: "object" as const,
        properties: {
          content: { type: "string" as const, description: "要记住的内容" },
          importance: { type: "number" as const, description: "重要性 1-5" },
        },
        required: ["content"],
      },
    },
    {
      name: "recall",
      description: "从长期记忆中回忆信息。需要引用过去的对话或事实时使用。",
      input_schema: {
        type: "object" as const,
        properties: {
          query: { type: "string" as const, description: "想回忆什么" },
        },
        required: ["query"],
      },
    },
    {
      name: "update_profile",
      description: "更新对用户的认知。当了解到用户新的信息或偏好变化时使用。",
      input_schema: {
        type: "object" as const,
        properties: {
          field: {
            type: "string" as const,
            enum: ["user_profile", "preferences"],
          },
          content: { type: "string" as const },
        },
        required: ["field", "content"],
      },
    },
  ];

  private executeTool(name: string, params: Record<string, any>): string {
    if (name === "remember") {
      this.longTerm.push({
        content: params.content,
        importance: params.importance ?? 3,
        timestamp: new Date().toISOString(),
      });
      return JSON.stringify({ status: "remembered" });
    }

    if (name === "recall") {
      const query = (params.query as string).toLowerCase();
      const queryWords = query.split(/\s+/);
      const relevant = this.longTerm
        .filter((m) =>
          queryWords.some((w) => m.content.toLowerCase().includes(w))
        )
        .sort((a, b) => b.importance - a.importance);
      return JSON.stringify({
        memories: relevant.slice(0, 5).map((m) => m.content),
      });
    }

    if (name === "update_profile") {
      this.core[params.field] = params.content;
      return JSON.stringify({ status: "profile updated" });
    }

    return JSON.stringify({ error: "unknown" });
  }

  private buildSystem(): string {
    const profile = this.core.user_profile || "暂无";
    const prefs = this.core.preferences || "暂无";
    return `你是一个有记忆能力的AI助手。

=== 你对用户的了解 ===
用户画像：${profile}
用户偏好：${prefs}

=== 记忆管理规则 ===
1. 用户透露个人信息时 -> update_profile
2. 讨论产生重要结论时 -> remember
3. 需要引用过去的事时 -> recall
4. 自然地使用你的记忆，不要每次都刻意提及`;
  }

  async chat(userInput: string): Promise<string> {
    this.workingMemory.push({ role: "user", content: userInput });
    if (this.workingMemory.length > this.maxWorking) {
      this.workingMemory = this.workingMemory.slice(-this.maxWorking);
    }

    for (let i = 0; i < 5; i++) {
      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        system: this.buildSystem(),
        tools: this.tools,
        messages: this.workingMemory,
      });

      if (response.stop_reason === "end_turn") {
        const text = response.content
          .filter((b: any) => b.type === "text")
          .map((b: any) => b.text)
          .join("");
        this.workingMemory.push({ role: "assistant", content: response.content });
        return text;
      }

      if (response.stop_reason === "tool_use") {
        this.workingMemory.push({ role: "assistant", content: response.content });
        const results: any[] = [];
        for (const block of response.content) {
          if (block.type === "tool_use") {
            const result = this.executeTool(block.name, block.input as Record<string, any>);
            results.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: result,
            });
          }
        }
        this.workingMemory.push({ role: "user", content: results });
      }
    }

    return "处理超时";
  }
}

async function main() {
  const agent = new AutonomousMemoryAgent();
  console.log("自主记忆 Agent（输入 quit 退出）\n");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const prompt = (): Promise<string> =>
    new Promise((resolve) => rl.question("你: ", resolve));

  while (true) {
    const user = (await prompt()).trim();
    if (user === "quit") break;
    console.log(`助手: ${await agent.chat(user)}\n`);
  }

  rl.close();
}

main();
```

## 小结

- **MemGPT 架构**：LLM 通过工具自主管理记忆的换入换出，像操作系统管理虚拟内存
- **时间衰减**：越久远的记忆权重越低，模拟自然遗忘。被频繁访问的记忆衰减更慢
- **记忆冲突**：新旧信息矛盾时，检测冲突并选择替换/合并/共存策略
- **记忆蒸馏**：大量具体记忆压缩为少量高级概括，减少存储和检索成本
- **Generative Agents**：三层架构（记忆流 + 反思 + 规划）+ 三维评分检索（时效 x 重要 x 相关），让 Agent 像人一样积累经验并形成认知
- **自主管理**：Agent 自己决定什么值得记住、什么时候检索、如何更新

## 练习

1. **MemGPT 对话**：用 MemGPTAgent 进行 20 轮对话，观察它何时主动存储和检索记忆。
2. **时间衰减调参**：调整 half_life_hours 从 1 到 72，观察记忆保留情况的变化。
3. **冲突测试**：存入互相矛盾的信息（"用户用Mac" -> "用户换了Windows"），验证冲突解决是否合理。
4. **蒸馏效果**：存入 30 条细节记忆，蒸馏为 5 条，评估信息保留的完整性。
5. **Generative Agents 反思**：向 MemoryStream 中添加 20+ 条观察记忆（降低 reflection_threshold 到 20），观察反思机制生成的洞察质量。尝试调整三维权重（alpha/beta/gamma），对比不同权重配比下的检索结果差异。

## 参考资源

- [MemGPT 论文 (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) -- MemGPT: Towards LLMs as Operating Systems
- [Generative Agents (arXiv:2304.03442)](https://arxiv.org/abs/2304.03442) -- Stanford 生成式 Agent，记忆系统设计的里程碑
- [Generative Agents 项目主页](https://reverie.herokuapp.com/arXiv_Demo/) -- 在线体验 25 个 AI 居民的虚拟小镇
- [Generative Agents 源码 (GitHub)](https://github.com/joonspk-research/generative_agents) -- 论文官方开源实现
- [Letta (MemGPT) 文档](https://docs.letta.com/) -- MemGPT 的开源实现
- [Memory for Agents 综述 (arXiv:2404.13501)](https://arxiv.org/abs/2404.13501) -- Agent 记忆系统综述
- [Harrison Chase: Memory for Agents (YouTube)](https://www.youtube.com/watch?v=bBuFGmyjDJ8) -- LangChain 创始人讲解记忆
