# RAG 高级：前沿 RAG 架构

> **学习目标**：掌握 Agentic RAG、Graph RAG、Self-RAG、CRAG 等前沿架构，学会评估 RAG 系统质量，构建自适应 RAG 系统。

学完本节，你将能够：
- 实现 Agentic RAG（Agent 自主决定何时检索）
- 实现 Graph RAG（知识图谱增强检索，支持多跳推理）
- 实现 Self-RAG（自我评估检索质量）
- 理解 CRAG（修正性 RAG）的工作机制
- 用评估框架量化 RAG 系统质量

## Agentic RAG

标准 RAG 是"无脑检索" -- 不管什么问题都先搜一遍。但有些问题 LLM 自己就能回答（如"1+1等于几"），有些需要多次检索（如"对比 A 和 B 的区别"）。

Agentic RAG 让 Agent 自己决定**要不要检索**、**检索什么**、**检索几次**：

```typescript
/** Agentic RAG -- Agent 自主决策的 RAG */
import Anthropic from "@anthropic-ai/sdk";
import { ChromaClient } from "chromadb";

const anthropicClient = new Anthropic();
const chromaClient = new ChromaClient();
const collection = await chromaClient.getOrCreateCollection({
  name: "agentic_rag",
});

// 预置知识库（实际中应该有大量文档）
await collection.add({
  documents: [
    "Python 3.12 引入了 type 语句，简化类型别名定义。",
    "FastAPI 使用 Pydantic 进行数据验证，支持自动生成 API 文档。",
    "Django ORM 支持惰性查询，只在真正需要数据时才执行 SQL。",
    "Docker 容器共享主机内核，比虚拟机更轻量。",
    "Kubernetes 通过 Pod 管理容器，支持自动扩缩容。",
  ],
  ids: Array.from({ length: 5 }, (_, i) => `doc_${i}`),
});

// 将检索定义为工具
const tools: Anthropic.Tool[] = [
  {
    name: "search_knowledge_base",
    description:
      "搜索内部知识库获取技术文档信息。"
      + "当用户的问题涉及特定技术细节、产品文档、内部规范时使用。"
      + "对于常识性问题、数学计算、简单推理，不需要使用此工具。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: {
          type: "string",
          description: "搜索关键词，尽量精确",
        },
        n_results: {
          type: "integer",
          description: "返回结果数量，默认3",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "search_with_filter",
    description: "带过滤条件搜索知识库。当需要限定搜索范围时使用。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: { type: "string" },
        category: {
          type: "string",
          enum: ["python", "web", "devops", "database"],
          description: "文档分类",
        },
      },
      required: ["query"],
    },
  },
];

async function searchKnowledgeBase(
  query: string,
  nResults: number = 3
): Promise<Record<string, any>> {
  const results = await collection.query({ queryTexts: [query], nResults });
  return {
    query,
    results: results.documents[0] || [],
    count: results.documents[0]?.length || 0,
  };
}

async function searchWithFilter(
  query: string,
  category?: string
): Promise<Record<string, any>> {
  const options: any = { queryTexts: [query], nResults: 3 };
  if (category) {
    options.where = { category };
  }
  const results = await collection.query(options);
  return { results: results.documents[0], filter: category };
}

const toolMap: Record<string, (args: any) => Promise<Record<string, any>>> = {
  search_knowledge_base: (args) =>
    searchKnowledgeBase(args.query, args.n_results),
  search_with_filter: (args) =>
    searchWithFilter(args.query, args.category),
};

async function agenticRag(question: string): Promise<string> {
  /** Agentic RAG：Agent 自主决定是否检索 */
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: question },
  ];
  const system =
    "你是一个技术助手。回答技术问题时，如果你确定答案就直接回答；"
    + "如果不确定或需要查询具体信息，使用搜索工具。"
    + "基于搜索结果回答时，明确引用来源。";

  for (let i = 0; i < 5; i++) {
    const response = await anthropicClient.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      system,
      tools,
      messages,
    });

    if (response.stop_reason === "end_turn") {
      return response.content
        .filter((b): b is Anthropic.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("");
    }

    if (response.stop_reason === "tool_use") {
      messages.push({ role: "assistant", content: response.content });
      const toolResults: Anthropic.ToolResultBlockParam[] = [];

      for (const block of response.content) {
        if (block.type === "tool_use") {
          const func = toolMap[block.name];
          const result = await func(block.input as any);
          console.log(
            `  [检索] ${block.name}: ${(block.input as any).query || ""}`
          );
          toolResults.push({
            type: "tool_result",
            tool_use_id: block.id,
            content: JSON.stringify(result),
          });
        }
      }
      messages.push({ role: "user", content: toolResults });
    }
  }

  return "处理超时";
}

// 测试：Agent 会自动判断是否需要检索
console.log("=== 不需要检索的问题 ===");
console.log(await agenticRag("1+1 等于几？"));

console.log("\n=== 需要检索的问题 ===");
console.log(await agenticRag("Python 3.12 的 type 语句怎么用？"));

console.log("\n=== 需要多次检索的问题 ===");
console.log(await agenticRag("对比 FastAPI 和 Django 在数据验证方面的差异"));
```

## Self-RAG：自我评估检索质量

Self-RAG 让模型在生成答案之前，先评估检索到的内容是否真的有用：

```typescript
/** Self-RAG -- 自我评估检索质量 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

class SelfRAG {
  /** 带自我评估的 RAG 系统 */
  private search: (query: string) => Promise<string[]>;

  constructor(searchFunc: (query: string) => Promise<string[]>) {
    this.search = searchFunc;
  }

  async evaluateRetrieval(
    query: string,
    documents: string[]
  ): Promise<Record<string, any>> {
    /** 评估检索结果的质量 */
    const docList = documents
      .map((d, i) => `[${i + 1}] ${d}`)
      .join("\n");

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 300,
      messages: [{
        role: "user",
        content: `评估以下检索结果对回答查询的有用程度。

查询：${query}

检索结果：
${docList}

返回 JSON：
{
    "overall_relevance": "high/medium/low/none",
    "useful_docs": [1, 3],
    "missing_info": "缺少什么信息",
    "should_retry": true/false,
    "retry_query": "如果需要重试，用什么查询"
}`,
      }],
    });

    let text =
      response.content[0].type === "text"
        ? response.content[0].text.trim()
        : "";
    try {
      if (text.includes("```")) {
        text = text.split("```")[1].replace("json", "").trim();
      }
      return JSON.parse(text);
    } catch {
      return { overall_relevance: "unknown", should_retry: false };
    }
  }

  async evaluateAnswer(
    query: string,
    answer: string,
    sources: string[]
  ): Promise<Record<string, any>> {
    /** 评估生成答案的质量（是否有幻觉） */
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 200,
      messages: [{
        role: "user",
        content: `评估答案是否完全基于参考来源。

问题：${query}
答案：${answer}
参考来源：${JSON.stringify(sources)}

返回 JSON：
{
    "is_supported": true/false,
    "has_hallucination": true/false,
    "confidence": 0.0-1.0,
    "unsupported_claims": ["不支持的说法"]
}`,
      }],
    });

    let text =
      response.content[0].type === "text"
        ? response.content[0].text.trim()
        : "";
    try {
      if (text.includes("```")) {
        text = text.split("```")[1].replace("json", "").trim();
      }
      return JSON.parse(text);
    } catch {
      return { is_supported: true, confidence: 0.5 };
    }
  }

  async query(
    question: string,
    maxRetries: number = 2
  ): Promise<Record<string, any>> {
    /** Self-RAG 完整流程 */
    let currentQuery = question;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      // 1. 检索
      const docs = await this.search(currentQuery);
      console.log(`[第${attempt + 1}轮检索] 查询: ${currentQuery}`);

      // 2. 评估检索质量
      const evalResult = await this.evaluateRetrieval(question, docs);
      console.log(`[评估] 相关性: ${evalResult.overall_relevance}`);

      if (
        evalResult.overall_relevance === "none" &&
        evalResult.should_retry
      ) {
        currentQuery = evalResult.retry_query || question;
        console.log(`[重试] 新查询: ${currentQuery}`);
        continue;
      }

      // 3. 过滤有用文档
      const usefulIndices: number[] =
        evalResult.useful_docs ||
        Array.from({ length: docs.length }, (_, i) => i + 1);
      const usefulDocs = usefulIndices
        .filter((i) => i > 0 && i <= docs.length)
        .map((i) => docs[i - 1]);

      // 4. 生成答案
      const context = usefulDocs.join("\n");
      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 500,
        messages: [{
          role: "user",
          content: `基于参考回答问题。只用参考中的信息。

参考：${context}
问题：${question}`,
        }],
      });
      const answer =
        response.content[0].type === "text" ? response.content[0].text : "";

      // 5. 评估答案质量
      const answerEval = await this.evaluateAnswer(
        question,
        answer,
        usefulDocs
      );
      console.log(
        `[答案评估] 置信度: ${answerEval.confidence ?? "N/A"}`
      );

      return {
        answer,
        sources: usefulDocs,
        retrieval_quality: evalResult.overall_relevance,
        answer_confidence: answerEval.confidence ?? 0,
        has_hallucination: answerEval.has_hallucination ?? false,
        attempts: attempt + 1,
      };
    }

    return {
      answer: "无法找到足够的信息来回答这个问题。",
      attempts: maxRetries + 1,
    };
  }
}
```

## CRAG：修正性 RAG

CRAG（Corrective RAG）在 Self-RAG 基础上更进一步：如果检索结果不够好，它不只是重试，而是**切换数据源**或**改变策略**：

```typescript
/** CRAG -- 修正性 RAG 概念实现 */

class CorrectiveRAG {
  /** 修正性 RAG：根据检索质量动态调整策略 */
  private primarySearch: (query: string) => Promise<string[]>;
  private webSearch: (query: string) => Promise<string[]>;
  private generate: (query: string, docs: string[]) => Promise<string>;

  constructor(
    primarySearch: (query: string) => Promise<string[]>,
    webSearch: (query: string) => Promise<string[]>,
    llmGenerate: (query: string, docs: string[]) => Promise<string>
  ) {
    this.primarySearch = primarySearch; // 主知识库检索
    this.webSearch = webSearch;         // 网络搜索（备用）
    this.generate = llmGenerate;        // LLM 生成
  }

  assessRelevance(query: string, doc: string): string {
    /** 评估单个文档的相关性 */
    // 实际中用 LLM 或分类器
    // 返回 "correct", "incorrect", "ambiguous"
    return "correct"; // 简化
  }

  async query(
    question: string
  ): Promise<{ answer: string; strategy: string; sources: string[] }> {
    // 1. 主检索
    const docs = await this.primarySearch(question);

    // 2. 逐个评估文档
    const correctDocs: string[] = [];
    for (const doc of docs) {
      const relevance = this.assessRelevance(question, doc);
      if (relevance === "correct") {
        correctDocs.push(doc);
      }
    }

    // 3. 根据评估结果决定策略
    let strategy: string;
    let finalDocs: string[];

    if (correctDocs.length >= 2) {
      // 策略A：足够多的相关文档，直接生成
      strategy = "direct";
      finalDocs = correctDocs;
    } else if (correctDocs.length === 1) {
      // 策略B：部分相关，补充网络搜索
      strategy = "augmented";
      const webDocs = await this.webSearch(question);
      finalDocs = [...correctDocs, ...webDocs.slice(0, 2)];
    } else {
      // 策略C：完全不相关，重写查询 + 全网搜索
      strategy = "web_fallback";
      finalDocs = await this.webSearch(question);
    }

    console.log(`[CRAG] 策略: ${strategy}, 文档数: ${finalDocs.length}`);

    // 4. 生成答案
    const answer = await this.generate(question, finalDocs);
    return { answer, strategy, sources: finalDocs };
  }
}
```

## Graph RAG：知识图谱增强检索

### 什么是 Graph RAG

Graph RAG 用**知识图谱**（Knowledge Graph）增强 RAG 的检索能力。传统向量 RAG 把文档切成块、做嵌入、按相似度检索——这对"某段文字提到了什么"的问题效果不错，但遇到需要**跨文档推理**、**多跳关系追踪**的问题就力不从心了。

Graph RAG 的核心思路：先从文档中提取**实体**和**关系**构建知识图谱，检索时不只是找相似文本，而是沿着图谱的边做**关系遍历**，把多跳推理链路上的信息一起拿出来给 LLM。

### 为什么需要 Graph RAG

向量检索有三个结构性局限：

1. **缺乏实体关系感知**：向量搜索只看语义相似度，不理解"张三是李四的经理"这种结构化关系
2. **无法多跳推理**：问"张三的经理的部门负责什么项目"，需要先找到张三的经理，再找经理的部门，再找部门的项目——向量检索一步到位搜不到
3. **全局摘要困难**：问"公司所有部门之间的协作关系是什么"，信息散落在几十个文档里，向量检索只能拿到局部碎片

Graph RAG 通过知识图谱的**显式关系结构**解决这些问题：实体是节点，关系是边，多跳推理变成图遍历。

### 核心流程

```
文档集合
  ↓ ① LLM 提取实体和关系
实体/关系三元组 (subject, predicate, object)
  ↓ ② 构建知识图谱
图结构 (节点 + 边)
  ↓ ③ 查询时：图检索（子图提取 / 路径遍历）
相关子图 + 原始文本片段
  ↓ ④ 拼装上下文，LLM 生成答案
最终回答
```

### 实现示例

以下是一个教学级 Graph RAG 实现，用邻接表构建知识图谱，用 Claude 做实体提取和答案生成：

```typescript
/**
 * Graph RAG -- 知识图谱增强的 RAG 系统
 *
 * 注：Python 的 networkx 在 TypeScript 中没有直接对等库。
 * 以下用简单的邻接表 + BFS 实现教学级知识图谱，功能等价于
 * networkx.MultiDiGraph 的子集。
 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// ============================================================
// 第一步：从文档中提取实体和关系（用 LLM）
// ============================================================

interface Triple {
  subject: string;
  predicate: string;
  object: string;
}

async function extractEntitiesAndRelations(
  text: string
): Promise<Triple[]> {
  /** 用 LLM 从文本中提取实体和关系三元组 */
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [{
      role: "user",
      content: `从以下文本中提取所有实体和它们之间的关系。

文本：
${text}

返回 JSON 数组，每个元素是一个三元组：
[
    {"subject": "实体A", "predicate": "关系", "object": "实体B"},
    ...
]

规则：
- subject 和 object 是具体的实体（人名、组织、技术、项目等）
- predicate 是关系描述（如"负责"、"使用"、"属于"、"依赖"等）
- 尽量提取所有能识别的关系
- 只返回 JSON 数组，不要其他内容`,
    }],
  });

  let textResult =
    response.content[0].type === "text"
      ? response.content[0].text.trim()
      : "";
  try {
    // 处理可能被 markdown 代码块包裹的情况
    if (textResult.includes("```")) {
      textResult = textResult.split("```")[1].replace("json", "").trim();
    }
    return JSON.parse(textResult);
  } catch {
    return [];
  }
}

// ============================================================
// 第二步：构建知识图谱
// ============================================================

interface Edge {
  from: string;
  relation: string;
  to: string;
}

interface Relation extends Edge {
  depth: number;
  is_path?: boolean;
}

class KnowledgeGraph {
  /** 基于邻接表的知识图谱（等价于 networkx.MultiDiGraph 子集） */
  private outEdges: Map<string, Edge[]> = new Map();
  private inEdges: Map<string, Edge[]> = new Map();
  private nodes: Set<string> = new Set();
  sourceTexts: Map<string, string> = new Map(); // 记录每个三元组的来源文本

  async addFromDocuments(documents: string[]) {
    /** 从文档列表批量构建图谱 */
    for (let i = 0; i < documents.length; i++) {
      console.log(`[构建图谱] 处理文档 ${i + 1}/${documents.length}...`);
      const triples = await extractEntitiesAndRelations(documents[i]);

      for (const triple of triples) {
        const { subject: subj, predicate: pred, object: obj } = triple;

        // 添加节点
        this.nodes.add(subj);
        this.nodes.add(obj);

        // 添加带标签的边
        const edge: Edge = { from: subj, relation: pred, to: obj };
        if (!this.outEdges.has(subj)) this.outEdges.set(subj, []);
        this.outEdges.get(subj)!.push(edge);
        if (!this.inEdges.has(obj)) this.inEdges.set(obj, []);
        this.inEdges.get(obj)!.push(edge);

        // 记录来源
        const key = `${subj}-${pred}-${obj}`;
        this.sourceTexts.set(key, documents[i]);
      }

      console.log(`  提取了 ${triples.length} 个三元组`);
    }

    console.log(
      `[图谱完成] ${this.nodes.size} 个节点, ` +
        `${[...this.outEdges.values()].reduce((s, e) => s + e.length, 0)} 条边`
    );
  }

  getNeighbors(entity: string, maxDepth: number = 2): Relation[] {
    /**
     * 获取实体的邻居信息（支持多跳）
     * @param entity  起始实体名称
     * @param maxDepth 最大遍历深度（1=直接关系，2=两跳关系）
     * @returns 关系三元组列表
     */
    if (!this.nodes.has(entity)) return [];

    const relations: Relation[] = [];
    const visited = new Set<string>();

    // BFS 遍历：从起始实体出发，逐层扩展
    const queue: [string, number][] = [[entity, 0]]; // [当前节点, 当前深度]
    visited.add(entity);

    while (queue.length > 0) {
      const [current, depth] = queue.shift()!;
      if (depth >= maxDepth) continue;

      // 出边：current -> neighbor
      for (const edge of this.outEdges.get(current) || []) {
        relations.push({
          from: current,
          relation: edge.relation,
          to: edge.to,
          depth: depth + 1,
        });
        if (!visited.has(edge.to)) {
          visited.add(edge.to);
          queue.push([edge.to, depth + 1]);
        }
      }

      // 入边：neighbor -> current
      for (const edge of this.inEdges.get(current) || []) {
        relations.push({
          from: edge.from,
          relation: edge.relation,
          to: current,
          depth: depth + 1,
        });
        if (!visited.has(edge.from)) {
          visited.add(edge.from);
          queue.push([edge.from, depth + 1]);
        }
      }
    }

    return relations;
  }

  findPath(entityA: string, entityB: string): Edge[] | null {
    /** 查找两个实体之间的关系路径（BFS 最短路径） */
    if (!this.nodes.has(entityA) || !this.nodes.has(entityB)) return null;

    // 在无向视图上做 BFS 找最短路径
    const visited = new Set<string>();
    const parent = new Map<string, string>();
    const queue: string[] = [entityA];
    visited.add(entityA);

    while (queue.length > 0) {
      const current = queue.shift()!;
      if (current === entityB) break;

      // 出边和入边都算邻居（无向遍历）
      const neighbors: string[] = [];
      for (const edge of this.outEdges.get(current) || []) {
        neighbors.push(edge.to);
      }
      for (const edge of this.inEdges.get(current) || []) {
        neighbors.push(edge.from);
      }

      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          parent.set(neighbor, current);
          queue.push(neighbor);
        }
      }
    }

    if (!parent.has(entityB) && entityA !== entityB) return null;

    // 还原路径
    const pathNodes: string[] = [];
    let node = entityB;
    while (node !== entityA) {
      pathNodes.unshift(node);
      node = parent.get(node)!;
    }
    pathNodes.unshift(entityA);

    // 还原路径上每条边的关系
    const pathRelations: Edge[] = [];
    for (let i = 0; i < pathNodes.length - 1; i++) {
      const src = pathNodes[i];
      const dst = pathNodes[i + 1];
      // 尝试正向边
      const outEdge = (this.outEdges.get(src) || []).find(
        (e) => e.to === dst
      );
      if (outEdge) {
        pathRelations.push({ from: src, relation: outEdge.relation, to: dst });
      } else {
        // 尝试反向边
        const inEdge = (this.outEdges.get(dst) || []).find(
          (e) => e.to === src
        );
        if (inEdge) {
          pathRelations.push({
            from: dst,
            relation: inEdge.relation,
            to: src,
          });
        }
      }
    }

    return pathRelations;
  }

  getAllNodes(): string[] {
    return [...this.nodes];
  }
}

// ============================================================
// 第三步：Graph RAG 检索 + 生成
// ============================================================

async function identifyEntitiesInQuery(
  query: string,
  knownEntities: string[]
): Promise<string[]> {
  /** 识别查询中提到的实体 */
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 200,
    messages: [{
      role: "user",
      content: `从查询中识别与已知实体匹配的实体名称。

查询：${query}
已知实体：${JSON.stringify(knownEntities)}

返回 JSON 数组，只包含在查询中出现或被提及的实体名。只返回 JSON 数组。`,
    }],
  });

  let text =
    response.content[0].type === "text"
      ? response.content[0].text.trim()
      : "";
  try {
    if (text.includes("```")) {
      text = text.split("```")[1].replace("json", "").trim();
    }
    return JSON.parse(text);
  } catch {
    return [];
  }
}

async function graphRagQuery(
  query: string,
  kg: KnowledgeGraph
): Promise<string> {
  /** Graph RAG 完整查询流程 */
  const allEntities = kg.getAllNodes();

  // 1. 识别查询中涉及的实体
  const queryEntities = await identifyEntitiesInQuery(query, allEntities);
  console.log(`[Graph RAG] 识别到实体: ${JSON.stringify(queryEntities)}`);

  // 2. 图检索：获取相关子图
  const allRelations: Relation[] = [];
  for (const entity of queryEntities) {
    // 获取每个实体的 2 跳邻居关系
    const neighbors = kg.getNeighbors(entity, 2);
    allRelations.push(...neighbors);
  }

  // 如果有多个实体，还查找它们之间的路径
  if (queryEntities.length >= 2) {
    for (let i = 0; i < queryEntities.length; i++) {
      for (let j = i + 1; j < queryEntities.length; j++) {
        const path = kg.findPath(queryEntities[i], queryEntities[j]);
        if (path) {
          console.log(
            `[Graph RAG] 找到路径: ${queryEntities[i]} -> ${queryEntities[j]}`
          );
          allRelations.push(
            ...path.map((r) => ({ ...r, depth: 0, is_path: true }))
          );
        }
      }
    }
  }

  // 3. 去重并格式化为上下文
  const seen = new Set<string>();
  const uniqueRelations: Relation[] = [];
  for (const r of allRelations) {
    const key = `${r.from}-${r.relation}-${r.to}`;
    if (!seen.has(key)) {
      seen.add(key);
      uniqueRelations.push(r);
    }
  }

  if (uniqueRelations.length === 0) {
    return "未找到相关的知识图谱信息。";
  }

  // 格式化关系为自然语言上下文
  const contextLines = uniqueRelations.map(
    (r) => `- ${r.from} --[${r.relation}]--> ${r.to}`
  );

  const context = contextLines.join("\n");
  console.log(`[Graph RAG] 检索到 ${uniqueRelations.length} 条关系`);

  // 4. 收集相关的原始文本片段
  const sourceSnippets = new Set<string>();
  for (const r of uniqueRelations) {
    const key = `${r.from}-${r.relation}-${r.to}`;
    const src = kg.sourceTexts.get(key);
    if (src) sourceSnippets.add(src);
  }

  let sourceContext = "";
  if (sourceSnippets.size > 0) {
    sourceContext =
      "\n\n原始文档片段：\n" + [...sourceSnippets].join("\n---\n");
  }

  // 5. LLM 生成最终答案
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 500,
    messages: [{
      role: "user",
      content: `基于以下知识图谱关系和原始文档回答问题。

知识图谱关系：
${context}
${sourceContext}

问题：${query}

请基于以上信息回答，注意利用关系链路进行推理。如果信息不足以回答，请说明。`,
    }],
  });

  return response.content[0].type === "text" ? response.content[0].text : "";
}

// ============================================================
// 运行示例
// ============================================================

// 示例文档（模拟一个公司的技术团队信息）
const documents = [
  "张伟是后端团队的技术负责人，他主导了订单系统的重构项目。后端团队使用 Python 和 FastAPI 框架。",
  "李娜是前端团队的负责人，前端团队使用 React 和 TypeScript。前端团队与后端团队共同协作开发电商平台。",
  "订单系统依赖用户服务和支付服务。支付服务由王强负责开发，使用了 Go 语言。",
  "电商平台是公司的核心产品，由产品部的赵敏负责规划。电商平台包含订单系统、用户系统和推荐系统。",
  "推荐系统使用机器学习技术，由数据团队的陈磊负责。推荐系统依赖用户行为数据。",
];

// 构建知识图谱
const kg = new KnowledgeGraph();
await kg.addFromDocuments(documents);

// 测试：单跳查询（直接关系）
console.log("\n=== 单跳查询 ===");
console.log(await graphRagQuery("张伟负责什么项目？", kg));

// 测试：多跳查询（需要推理链）
console.log("\n=== 多跳查询 ===");
console.log(
  await graphRagQuery("订单系统依赖的支付服务是谁开发的？用了什么语言？", kg)
);

// 测试：全局关联查询
console.log("\n=== 全局关联查询 ===");
console.log(await graphRagQuery("电商平台涉及哪些团队和技术栈？", kg));
```

### Graph RAG vs 向量 RAG 对比

| 维度 | 向量 RAG | Graph RAG |
|------|----------|-----------|
| **检索方式** | 语义相似度匹配 | 图遍历 + 关系路径查找 |
| **擅长场景** | "某段文字说了什么" | "A 和 B 之间什么关系" |
| **多跳推理** | 不支持（只能检索单个片段） | 原生支持（沿边遍历即可） |
| **全局摘要** | 困难（信息碎片化） | 较好（图结构天然聚合关系） |
| **构建成本** | 低（切块 + 嵌入） | 高（需要实体/关系提取） |
| **维护成本** | 低（新增文档直接入库） | 中（需要增量更新图谱） |
| **适合的数据** | 非结构化长文本 | 实体关系密集的文档 |
| **延迟** | 低（单次向量查询） | 中（图遍历 + 可能多次查询） |

> **实际中常常两者结合使用**：向量 RAG 负责语义层面的模糊检索，Graph RAG 负责结构化关系的精确推理。先用向量搜索初筛相关文档，再在知识图谱上做多跳推理，最后合并上下文交给 LLM 生成答案。

### 适用场景

Graph RAG 特别适合以下场景：

- **企业知识管理**：人员-部门-项目-技术栈之间的多层关系查询
- **医疗健康**：疾病-症状-药物-副作用的关联推理
- **金融风控**：公司-股东-关联方-交易的多跳追踪
- **学术研究**：论文-作者-机构-引用的关系网络分析
- **法律法规**：法规-条款-判例-适用场景的结构化检索

不适合的场景：文档本身缺乏明确实体关系（如散文、评论），或者查询只需要语义匹配（如"关于容器化的最佳实践"）。

## RAG 评估框架

构建 RAG 系统不难，**知道它好不好**才是难点。评估 RAG 需要从多个维度衡量：

```typescript
/** RAG 评估框架 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface EvalMetrics {
  relevancy?: number;
  faithfulness?: number;
  context_precision?: number;
  correctness?: number;
  overall: number;
}

interface TestCase {
  question: string;
  answer: string;
  retrieved_docs: string[];
  ground_truth?: string;
}

class RAGEvaluator {
  /** RAG 系统评估器 */

  async evaluateSingle(
    question: string,
    answer: string,
    retrievedDocs: string[],
    groundTruth?: string
  ): Promise<EvalMetrics> {
    /** 评估单个 QA 对 */
    const metrics: Record<string, number> = {};

    // 1. 相关性（Answer Relevancy）：答案是否回答了问题
    metrics.relevancy = await this.scoreRelevancy(question, answer);

    // 2. 忠实度（Faithfulness）：答案是否基于检索到的文档
    metrics.faithfulness = await this.scoreFaithfulness(answer, retrievedDocs);

    // 3. 上下文精度（Context Precision）：检索到的文档是否相关
    metrics.context_precision = await this.scoreContextPrecision(
      question,
      retrievedDocs
    );

    // 4. 如果有标准答案，计算正确性
    if (groundTruth) {
      metrics.correctness = await this.scoreCorrectness(answer, groundTruth);
    }

    // 总分
    const scores = Object.values(metrics).filter(
      (v) => typeof v === "number"
    );
    metrics.overall =
      scores.length > 0
        ? scores.reduce((a, b) => a + b, 0) / scores.length
        : 0;

    return metrics as unknown as EvalMetrics;
  }

  private async scoreRelevancy(
    question: string,
    answer: string
  ): Promise<number> {
    /** 答案是否切题 */
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 50,
      messages: [{
        role: "user",
        content: `评估答案与问题的相关性。
问题：${question}
答案：${answer}
返回 0.0-1.0 的分数，只返回数字。`,
      }],
    });
    try {
      const text =
        response.content[0].type === "text" ? response.content[0].text.trim() : "";
      return parseFloat(text);
    } catch {
      return 0.5;
    }
  }

  private async scoreFaithfulness(
    answer: string,
    docs: string[]
  ): Promise<number> {
    /** 答案是否忠实于来源文档 */
    const context = docs.join("\n");
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 50,
      messages: [{
        role: "user",
        content: `评估答案是否完全基于参考文档（没有编造信息）。
参考文档：${context}
答案：${answer}
返回 0.0-1.0 的分数，只返回数字。`,
      }],
    });
    try {
      const text =
        response.content[0].type === "text" ? response.content[0].text.trim() : "";
      return parseFloat(text);
    } catch {
      return 0.5;
    }
  }

  private async scoreContextPrecision(
    question: string,
    docs: string[]
  ): Promise<number> {
    /** 检索到的文档是否与问题相关 */
    const docList = docs.map((d, i) => `[${i + 1}] ${d}`).join("\n");
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 50,
      messages: [{
        role: "user",
        content: `评估检索到的文档与问题的相关程度。
问题：${question}
文档：${docList}
返回 0.0-1.0 的分数，只返回数字。`,
      }],
    });
    try {
      const text =
        response.content[0].type === "text" ? response.content[0].text.trim() : "";
      return parseFloat(text);
    } catch {
      return 0.5;
    }
  }

  private async scoreCorrectness(
    answer: string,
    groundTruth: string
  ): Promise<number> {
    /** 答案是否正确 */
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 50,
      messages: [{
        role: "user",
        content: `对比答案与标准答案，评估正确性。
标准答案：${groundTruth}
实际答案：${answer}
返回 0.0-1.0 的分数，只返回数字。`,
      }],
    });
    try {
      const text =
        response.content[0].type === "text" ? response.content[0].text.trim() : "";
      return parseFloat(text);
    } catch {
      return 0.5;
    }
  }

  async evaluateBatch(
    testCases: TestCase[]
  ): Promise<{ individual: EvalMetrics[]; aggregate: Record<string, number> }> {
    /** 批量评估 */
    const allResults: EvalMetrics[] = [];

    for (const testCase of testCases) {
      const result = await this.evaluateSingle(
        testCase.question,
        testCase.answer,
        testCase.retrieved_docs,
        testCase.ground_truth
      );
      allResults.push(result);
      console.log(
        `  Q: ${testCase.question.slice(0, 40)}... | 总分: ${result.overall.toFixed(2)}`
      );
    }

    // 聚合指标
    const avgMetrics: Record<string, number> = {};
    const keys = Object.keys(allResults[0]) as (keyof EvalMetrics)[];
    for (const key of keys) {
      const values = allResults
        .map((r) => r[key])
        .filter((v): v is number => typeof v === "number");
      if (values.length > 0) {
        avgMetrics[`avg_${key}`] =
          values.reduce((a, b) => a + b, 0) / values.length;
      }
    }

    return { individual: allResults, aggregate: avgMetrics };
  }
}
```

## 小结

- **Agentic RAG**：Agent 自主决定何时检索、检索什么，避免不必要的检索开销
- **Graph RAG**：用知识图谱捕获实体关系，支持多跳推理和全局关联查询
- **Self-RAG**：检索后自评质量，质量不够则重试或改写查询
- **CRAG**：根据检索质量切换策略（直接用 / 补充搜索 / 完全换源）
- **RAG 评估**：从相关性、忠实度、上下文精度、正确性四个维度衡量
- **核心趋势**：从固定流水线 -> Agent 动态编排，让 LLM 自己判断最佳检索策略

## 练习

1. **Agentic RAG**：扩展 agentic_rag 函数，添加"不需要检索时直接回答"的统计，观察不同类型问题的检索率。
2. **Graph RAG 扩展**：在 KnowledgeGraph 类中实现社区检测（提示：可以用连通分量算法或引入图分析库），将社区摘要作为全局查询的上下文。
3. **Self-RAG 实验**：构建一个测试集（10 个问题），分别用标准 RAG 和 Self-RAG 回答，对比质量。
4. **评估实践**：用 RAGEvaluator 评估你在入门篇构建的 RAG 系统，找到最薄弱的环节。

## 参考资源

- [Self-RAG 论文 (arXiv:2310.11511)](https://arxiv.org/abs/2310.11511) -- Self-Reflective RAG
- [CRAG 论文 (arXiv:2401.15884)](https://arxiv.org/abs/2401.15884) -- Corrective RAG
- [Graph RAG 论文 (arXiv:2404.16130)](https://arxiv.org/abs/2404.16130) -- From Local to Global: A Graph RAG Approach
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) -- 微软开源的 Graph RAG 实现
- [Ragas 评估框架](https://docs.ragas.io/) -- RAG 评估开源工具
- [LangChain: Agentic RAG](https://python.langchain.com/docs/tutorials/qa_chat_history/) -- Agentic RAG 教程
- [Jerry Liu: Building Production RAG](https://www.youtube.com/watch?v=TRjq7t2Ms5I) -- 生产级 RAG 实践
