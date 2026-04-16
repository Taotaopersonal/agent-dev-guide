# RAG 入门

> **学习目标**：理解 RAG 的核心思想和价值，掌握 Embedding 原理，用 ChromaDB 构建第一个知识库，搭建完整的文档问答管道。

学完本节，你将能够：
- 解释 RAG 是什么、为什么需要它
- 理解 Embedding（向量化）的基本原理
- 用 ChromaDB 构建和检索知识库
- 搭建一个完整的"加载 -> 分块 -> 向量化 -> 检索 -> 生成"管道
- 构建一个能回答文档相关问题的问答系统

## 为什么需要 RAG

LLM 有三个根本性的局限：

**知识有截止日期。** GPT-4 的训练数据截止到某个时间点，你公司上周发布的新产品规范它完全不知道。

**不知道你的私有数据。** 你公司的内部文档、客户信息、业务规则——这些从来没出现在 LLM 的训练数据里。

**会编造答案。** 当 LLM 不知道答案时，它不会说"我不知道"，而是自信满满地编一个看起来很合理的答案。这就是"幻觉"（Hallucination）。

RAG（Retrieval-Augmented Generation，检索增强生成）的解决思路非常直接：**先搜索，再回答**。用户提问时，先从你的知识库里找出相关内容，然后把这些内容和问题一起交给 LLM，让它"带着参考资料"回答。

就像开卷考试——有了参考资料，回答既准确又有据可查。

```
用户提问: "我们的退款政策是什么？"
     ↓
[检索] 从知识库中找到相关文档片段：
  "退款政策：购买后7天内可无条件退款..."
     ↓
[生成] 将文档片段 + 问题一起发给 LLM
     ↓
LLM回答: "根据公司政策，购买后7天内可以无条件退款..."
```

::: tip RAG vs 微调
你可能听说过微调（Fine-tuning）也能让 LLM 学习新知识。两者的区别：
- **RAG**：知识存在外部，随时可更新，不需要重新训练模型。适合频繁变化的知识。
- **微调**：知识嵌入模型参数，更新需要重新训练。适合稳定的领域知识和风格调整。
- 大多数场景推荐**先用 RAG**，效果不够再考虑微调。两者也可以结合使用。
:::

## Embedding：把文本变成向量

RAG 的第一个核心技术是 Embedding——把文本转换成一串数字（向量）。

为什么要这么做？因为计算机不擅长直接理解"语义相似性"。"苹果"和"水果"在字符层面完全不同，但语义上很接近。Embedding 模型的作用就是把文本映射到一个高维空间，使得**语义相近的文本，向量也相近**。

```typescript
/**
 * Embedding 基础演示
 * 运行前：npm install openai
 */
import OpenAI from "openai";

const openaiClient = new OpenAI();

async function getEmbedding(
  text: string,
  model: string = "text-embedding-3-small"
): Promise<number[]> {
  /** 获取文本的 Embedding 向量 */
  const response = await openaiClient.embeddings.create({
    model,
    input: text,
  });
  return response.data[0].embedding;
}

// 计算余弦相似度
function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

// 试试看
const vec1 = await getEmbedding("Python 是一门编程语言");
const vec2 = await getEmbedding("Java 是一种编程语言");
const vec3 = await getEmbedding("今天天气真不错");

console.log(`向量维度: ${vec1.length}`); // text-embedding-3-small 输出 1536 维

console.log(`编程语言 vs 编程语言: ${cosineSimilarity(vec1, vec2).toFixed(4)}`); // 高相似度
console.log(`编程语言 vs 天气: ${cosineSimilarity(vec1, vec3).toFixed(4)}`);     // 低相似度
```

关键概念：
- **向量维度**：text-embedding-3-small 输出 1536 维向量，每个维度是一个浮点数
- **余弦相似度**：衡量两个向量的方向是否一致，范围 [-1, 1]，越接近 1 越相似
- **语义匹配**：不需要关键词完全相同，"如何入门机器学习"和"ML 初学者指南"的向量也会很接近

### 常用 Embedding 模型

| 模型 | 维度 | 特点 | 适用场景 |
|------|------|------|---------|
| text-embedding-3-small | 1536 | OpenAI，性价比高 | 通用场景，推荐入门使用 |
| text-embedding-3-large | 3072 | OpenAI，精度更高 | 对精度要求高的场景 |
| BAAI/bge-large-zh-v1.5 | 1024 | 开源，中文效果好 | 中文场景，本地部署 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 开源，速度快 | 资源受限，快速原型 |

## ChromaDB：最简单的向量数据库

有了向量，需要一个地方存起来，还能高效地搜索。这就是向量数据库的作用。ChromaDB 是最适合入门的选择——安装简单，零配置，API 直观。

```typescript
/**
 * ChromaDB 入门：构建和检索知识库
 * 运行前：npm install chromadb
 */
import { ChromaClient, OpenAIEmbeddingFunction } from "chromadb";

// 初始化 ChromaDB（持久化模式，数据存到磁盘）
const client = new ChromaClient({ path: "./my_knowledge_base" });

// 使用 OpenAI embedding（也可以用默认的免费 embedding）
const openaiEf = new OpenAIEmbeddingFunction({
  openai_model_name: "text-embedding-3-small",
});

// 创建一个集合（类似数据库的表）
const collection = await client.getOrCreateCollection({
  name: "company_docs",
  embeddingFunction: openaiEf,
  metadata: { "hnsw:space": "cosine" }, // 使用余弦距离
});

// === 添加文档 ===
const documents = [
  "退款政策：购买后7天内可以无条件退款，需提供订单号。",
  "会员等级：消费满1000元升级为银卡会员，享受9折优惠。",
  "配送说明：普通快递3-5天送达，顺丰次日达（需加10元）。",
  "售后服务：产品质量问题30天内免费换货，人为损坏不在保修范围内。",
  "积分规则：每消费1元获得1积分，积分可在下次购物时抵扣（100积分=1元）。",
  "营业时间：线下门店周一至周五 9:00-18:00，周末 10:00-17:00。",
];

await collection.add({
  documents,
  ids: documents.map((_, i) => `doc_${i}`),
  metadatas: [
    { category: "退款" },
    { category: "会员" },
    { category: "配送" },
    { category: "售后" },
    { category: "积分" },
    { category: "门店" },
  ],
});

console.log(`知识库中共 ${await collection.count()} 条文档`);

// === 检索 ===
const results = await collection.query({
  queryTexts: ["我买的东西想退货怎么办"],
  nResults: 3,
});

console.log("\n检索结果：");
results.documents[0].forEach((doc, i) => {
  const dist = results.distances![0][i];
  console.log(`  [${dist.toFixed(4)}] ${doc}`);
});

// === 带元数据过滤的检索 ===
const filteredResults = await collection.query({
  queryTexts: ["会员有什么优惠"],
  nResults: 3,
  where: { category: "会员" }, // 只在"会员"分类中搜索
});

console.log("\n过滤检索（仅会员类）：");
filteredResults.documents[0].forEach((doc) => {
  console.log(`  ${doc}`);
});
```

ChromaDB 的核心操作就四个：
- `collection.add()` -- 添加文档（自动向量化并存储）
- `collection.query()` -- 检索（自动将查询向量化，找最相似的）
- `collection.update()` -- 更新文档
- `collection.delete()` -- 删除文档

::: info 为什么不用普通数据库？
普通数据库用 SQL 做精确匹配：`WHERE content LIKE '%退货%'`。但用户可能说"想把东西退了"、"不想要了怎么办"——措辞千变万化，关键词匹配根本覆盖不了。向量检索做的是**语义匹配**，不管用户怎么措辞，只要意思接近就能找到。
:::

## 完整的 RAG 管道

现在把所有组件串起来，搭建一个完整的文档问答系统。

```typescript
/**
 * 完整的 RAG 文档问答系统
 * 运行前：npm install @anthropic-ai/sdk chromadb openai
 */
import Anthropic from "@anthropic-ai/sdk";
import { ChromaClient, OpenAIEmbeddingFunction } from "chromadb";

const anthropicClient = new Anthropic();

class SimpleRAG {
  /** 简单但完整的 RAG 系统 */
  private chroma: ChromaClient;
  private ef: OpenAIEmbeddingFunction;
  private collection!: Awaited<ReturnType<ChromaClient["getOrCreateCollection"]>>;

  constructor() {
    this.chroma = new ChromaClient({ path: "./rag_db" });
    this.ef = new OpenAIEmbeddingFunction({
      openai_model_name: "text-embedding-3-small",
    });
  }

  async init(collectionName: string = "knowledge_base") {
    this.collection = await this.chroma.getOrCreateCollection({
      name: collectionName,
      embeddingFunction: this.ef,
    });
  }

  // === Step 1: 加载文档 ===
  async loadDocuments(
    documents: string[],
    metadatas?: Record<string, any>[]
  ) {
    /** 将文档加入知识库 */
    const count = await this.collection.count();
    const ids = documents.map((_, i) => `doc_${count + i}`);
    await this.collection.add({
      documents,
      ids,
      metadatas: metadatas || documents.map(() => ({})),
    });
    const newCount = await this.collection.count();
    console.log(`已加载 ${documents.length} 条文档，总计 ${newCount} 条`);
  }

  // === Step 2: 分块（简单版：按段落分割） ===
  static chunkText(
    text: string,
    chunkSize: number = 500,
    overlap: number = 50
  ): string[] {
    /** 将长文本分成小块 */
    const chunks: string[] = [];
    let start = 0;
    while (start < text.length) {
      const end = start + chunkSize;
      const chunk = text.slice(start, end).trim();
      if (chunk) chunks.push(chunk);
      start = end - overlap; // 有重叠，避免在边界丢失信息
    }
    return chunks;
  }

  // === Step 3 & 4: 检索 ===
  async retrieve(query: string, topK: number = 3): Promise<string[]> {
    /** 检索最相关的文档片段 */
    const results = await this.collection.query({
      queryTexts: [query],
      nResults: topK,
    });
    return results.documents[0] as string[];
  }

  // === Step 5: 生成回答 ===
  async answer(question: string, topK: number = 3): Promise<string> {
    /** 检索 + 生成：完整的 RAG 流程 */
    // 检索相关文档
    const relevantDocs = await this.retrieve(question, topK);

    if (!relevantDocs.length) {
      return "知识库中没有找到相关信息。";
    }

    // 构建上下文
    const context = relevantDocs.join("\n---\n");

    // 生成回答
    const response = await anthropicClient.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      messages: [{
        role: "user",
        content: `基于以下参考资料回答用户的问题。
如果参考资料中没有相关信息，如实告知用户你不确定，不要编造。
回答时引用信息来源。

参考资料：
${context}

用户问题：${question}`,
      }],
    });
    return response.content[0].type === "text" ? response.content[0].text : "";
  }
}

// === 使用示例 ===
const rag = new SimpleRAG();
await rag.init();

// 加载公司知识库
const docs = [
  "退款政策：购买后7天内可以无条件退款，需提供订单号和支付凭证。超过7天但在30天内，"
  + "如果产品未拆封可以退款，需扣除10%手续费。超过30天不接受退款。",

  "会员体系：普通会员消费满1000元自动升级为银卡会员（9折），满5000元升级为金卡（8.5折），"
  + "满20000元升级为钻石卡（8折）。会员等级每年1月1日重新计算。",

  "配送方式：支持普通快递（免费，3-5个工作日）、顺丰快递（加10元，次日达）、"
  + "同城配送（加5元，当日达，仅限北上广深）。订单满199元免运费。",

  "积分规则：每消费1元获得1积分。积分可用于兑换优惠券或直接抵扣（100积分=1元）。"
  + "积分有效期为获得之日起12个月，过期清零。每笔订单积分抵扣不超过订单金额的20%。",
];
await rag.loadDocuments(docs);

// 问几个问题
const questions = [
  "我买了15天了还能退货吗？",
  "怎样才能成为金卡会员？有什么优惠？",
  "我在上海，最快什么时候能收到货？",
  "积分能抵多少钱？会过期吗？",
];

for (const q of questions) {
  console.log(`\n问：${q}`);
  console.log(`答：${await rag.answer(q)}`);
  console.log("-".repeat(60));
}
```

运行这段代码，你会看到 RAG 系统准确地根据知识库内容回答问题——而且不会编造不存在的信息。

## RAG 管道全景图

一个完整的 RAG 系统包含两个阶段：

**离线阶段（索引构建）：**
```
原始文档 → 文本提取 → 分块(Chunking) → 向量化(Embedding) → 存入向量数据库
```

**在线阶段（查询回答）：**
```
用户提问 → 查询向量化 → 向量检索(Top-K) → 构建上下文 → LLM 生成回答
```

::: warning 常见陷阱
1. **分块太大**：检索结果包含太多无关信息，LLM 被干扰
2. **分块太小**：上下文被切断，丢失关键信息
3. **不做重叠**：重要信息刚好在两个块的边界被切断
4. **忽略元数据**：不添加分类、来源等元数据，无法做过滤检索
5. **生成时不限制**：不告诉 LLM"只根据参考资料回答"，它还是会编造
:::

## 小结

- **RAG 的核心**：先搜索再回答，让 LLM 带着参考资料回答问题
- **Embedding**：把文本变成向量，语义相近的文本向量也相近
- **ChromaDB**：零配置的向量数据库，add/query/update/delete 四个核心操作
- **完整管道**：加载 -> 分块 -> 向量化 -> 检索 -> 生成
- **关键原则**：在 prompt 中明确告诉 LLM"基于参考资料回答，不要编造"

## 练习

1. **动手做**：用你自己的文档（比如一份产品说明书或技术文档）构建一个 RAG 系统，测试至少 10 个问题。
2. **分块实验**：把 `chunk_size` 分别设为 100、500、1000，对比检索效果。哪个 chunk_size 效果最好？为什么？
3. **添加文件加载**：扩展 SimpleRAG，让它能直接加载 `.txt` 和 `.md` 文件（读取文件内容，分块后加入知识库）。
4. **思考题**：如果知识库有 10 万条文档，直接用 ChromaDB 能撑住吗？什么时候需要换更强大的向量数据库？

## 参考资源

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (arXiv:2005.11401)](https://arxiv.org/abs/2005.11401) -- RAG 原始论文
- [ChromaDB Documentation](https://docs.trychroma.com/) -- ChromaDB 官方文档
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) -- OpenAI Embedding 使用指南
- [Anthropic: Contextual Retrieval](https://docs.anthropic.com/en/docs/build-with-claude/retrieval) -- Anthropic RAG 文档
- [James Briggs: RAG from Scratch (YouTube)](https://www.youtube.com/watch?v=sVcwVQRHIc8) -- 从零构建 RAG 的视频教程
