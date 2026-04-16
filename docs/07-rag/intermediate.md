# RAG 进阶

> **学习目标**：掌握分块策略的选择，理解混合检索和 Reranking 的原理，学会用查询改写提升检索质量，构建一个完整的检索流水线。

学完本节，你将能够：
- 根据文档类型选择合适的分块策略
- 实现向量 + 关键词的混合检索
- 用 Reranking 精排提升检索精度
- 实现多查询检索和 HyDE 等查询改写技术
- 用 Recall@K、MRR 等指标评估检索质量

## 分块策略：RAG 效果的基石

入门篇用了一个简单的按长度分块。但实际场景中，分块策略对 RAG 效果影响巨大——分得好能大幅提升回答准确率，分得差会让系统几乎不可用。

### 策略一：固定大小分块

最简单的方式，按字符数切分。

```typescript
class FixedSizeChunker {
  /** 固定大小分块 */
  private chunkSize: number;
  private overlap: number;

  constructor(chunkSize: number = 500, overlap: number = 50) {
    this.chunkSize = chunkSize;
    this.overlap = overlap;
  }

  chunk(text: string): string[] {
    const chunks: string[] = [];
    let start = 0;
    while (start < text.length) {
      const end = start + this.chunkSize;
      const chunk = text.slice(start, end).trim();
      if (chunk) chunks.push(chunk);
      start = end - this.overlap;
    }
    return chunks;
  }
}
```

**优点**：简单、可预测。**缺点**：不尊重语义边界，可能在句子中间切断。

### 策略二：递归字符分块

LangChain 的默认策略。按多级分隔符递归分割：先尝试按段落分，段落太长就按句子分，句子太长就按字符分。

```typescript
class RecursiveChunker {
  /** 递归字符分块 */
  private chunkSize: number;
  private overlap: number;
  private separators: string[];

  constructor(
    chunkSize: number = 500,
    overlap: number = 50,
    separators?: string[]
  ) {
    this.chunkSize = chunkSize;
    this.overlap = overlap;
    this.separators = separators || ["\n\n", "\n", "。", "！", "？", ".", " ", ""];
  }

  chunk(text: string): string[] {
    return this.recursiveSplit(text, this.separators);
  }

  private recursiveSplit(text: string, separators: string[]): string[] {
    if (text.length <= this.chunkSize) {
      return text.trim() ? [text] : [];
    }

    const sep = separators.length > 0 ? separators[0] : "";
    const remainingSeps = separators.length > 1 ? separators.slice(1) : [""];

    if (!sep) {
      // 最后的兜底：按字符切
      const result: string[] = [];
      for (let i = 0; i < text.length; i += this.chunkSize - this.overlap) {
        result.push(text.slice(i, i + this.chunkSize));
      }
      return result;
    }

    const parts = text.split(sep);
    const chunks: string[] = [];
    let current = "";

    for (const part of parts) {
      const test = current ? current + sep + part : part;
      if (test.length <= this.chunkSize) {
        current = test;
      } else {
        if (current) {
          chunks.push(current.trim());
        }
        // 如果单个 part 太长，用下一级分隔符继续切
        if (part.length > this.chunkSize) {
          chunks.push(...this.recursiveSplit(part, remainingSeps));
          current = "";
        } else {
          current = part;
        }
      }
    }

    if (current.trim()) {
      chunks.push(current.trim());
    }

    return chunks;
  }
}
```

**优点**：尊重自然段落和句子边界。**缺点**：实现稍复杂，分块大小不均匀。

### 策略三：语义分块

用 Embedding 检测语义变化点来决定分块边界。当相邻句子的语义相似度突然下降，说明话题发生了转换，这就是一个好的分块点。

```typescript
import OpenAI from "openai";

const openaiClient = new OpenAI();

class SemanticChunker {
  /** 基于语义的分块 */
  private threshold: number;

  constructor(threshold: number = 0.5) {
    this.threshold = threshold; // 相似度阈值，低于此值就分块
  }

  async chunk(text: string): Promise<string[]> {
    // 按句子分割
    const sentences = text
      .replace(/。/g, "。\n")
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (sentences.length <= 1) return sentences;

    // 获取每个句子的 Embedding
    const response = await openaiClient.embeddings.create({
      model: "text-embedding-3-small",
      input: sentences,
    });
    const embeddings = response.data.map((item) => item.embedding);

    // 计算相邻句子的相似度
    const chunks: string[] = [];
    let currentChunk = [sentences[0]];

    for (let i = 1; i < sentences.length; i++) {
      const sim = this.cosineSim(embeddings[i - 1], embeddings[i]);
      if (sim < this.threshold) {
        // 语义断裂，开始新块
        chunks.push(currentChunk.join(""));
        currentChunk = [sentences[i]];
      } else {
        currentChunk.push(sentences[i]);
      }
    }

    if (currentChunk.length > 0) {
      chunks.push(currentChunk.join(""));
    }

    return chunks;
  }

  private cosineSim(a: number[], b: number[]): number {
    const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return dot / (normA * normB);
  }
}
```

**优点**：分块边界与语义一致。**缺点**：需要额外的 Embedding API 调用，成本较高。

### 策略四：Markdown 结构分块

对于 Markdown 格式的文档，利用标题层级来分块。

```typescript
class MarkdownChunker {
  /** 基于 Markdown 结构的分块 */

  chunk(text: string): string[] {
    const chunks: string[] = [];
    let currentHeaders: string[] = [];
    let currentContent: string[] = [];

    for (const line of text.split("\n")) {
      if (line.startsWith("#")) {
        // 遇到新标题，保存之前的内容
        if (currentContent.length > 0) {
          const headerPrefix = currentHeaders.join(" > ");
          const content = currentContent.join("\n").trim();
          if (content) {
            chunks.push(`[${headerPrefix}]\n${content}`);
          }
          currentContent = [];
        }

        // 更新标题层级
        const level = line.length - line.replace(/^#+/, "").length;
        const title = line.replace(/^#+\s*/, "").trim();
        currentHeaders = [...currentHeaders.slice(0, level - 1), title];
      } else {
        currentContent.push(line);
      }
    }

    // 最后一块
    if (currentContent.length > 0) {
      const headerPrefix = currentHeaders.join(" > ");
      const content = currentContent.join("\n").trim();
      if (content) {
        chunks.push(`[${headerPrefix}]\n${content}`);
      }
    }

    return chunks;
  }
}
```

**优点**：保留文档结构信息，检索时附带标题上下文。**缺点**：只适用于 Markdown。

### 分块策略选择指南

| 文档类型 | 推荐策略 | chunk_size |
|---------|---------|-----------|
| 短文、FAQ | 固定大小 | 200-500 |
| 长文、报告 | 递归字符 | 500-1000 |
| 技术文档 | Markdown 结构 | 按标题 |
| 话题混杂文本 | 语义分块 | 自动 |

::: warning chunk_size 和 overlap 的经验值
- **chunk_size**：300-1000 字符比较合适。太小丢失上下文，太大引入噪音。
- **overlap**：通常是 chunk_size 的 10%-20%。完全不重叠容易在边界丢信息。
- **最佳值取决于你的数据**——一定要在你的数据上做实验，不要盲信经验值。
:::

## 混合检索：向量 + 关键词

纯向量检索擅长语义匹配，但有时会错过精确的关键词匹配。比如用户搜"RFC 2616"，这是一个精确的专有名词，向量检索可能反而不如关键词搜索准确。

混合检索的思路：**两种方法各搜一遍，然后合并结果**。

```typescript
class HybridSearch {
  /** 向量 + 关键词混合检索 */
  private documents: string[] = [];
  private vectors: number[][] = [];

  add(doc: string, vector: number[]) {
    this.documents.push(doc);
    this.vectors.push(vector);
  }

  keywordSearch(query: string, topK: number = 5): [number, number][] {
    /** BM25 风格的关键词检索（简化版） */
    const queryTerms = new Set(query.toLowerCase().split(/\s+/));
    const scores: [number, number][] = this.documents.map((doc, i) => {
      const docTerms = new Set(doc.toLowerCase().split(/\s+/));
      const overlap = [...queryTerms].filter((t) => docTerms.has(t)).length;
      return [i, overlap / (queryTerms.size + 1)];
    });
    scores.sort((a, b) => b[1] - a[1]);
    return scores.slice(0, topK);
  }

  vectorSearch(queryVector: number[], topK: number = 5): [number, number][] {
    /** 向量检索 */
    const scores: [number, number][] = this.vectors.map((v, i) => {
      const dot = queryVector.reduce((sum, qi, j) => sum + qi * v[j], 0);
      const normQ = Math.sqrt(queryVector.reduce((s, qi) => s + qi * qi, 0));
      const normV = Math.sqrt(v.reduce((s, vi) => s + vi * vi, 0));
      return [i, dot / (normQ * normV)];
    });
    scores.sort((a, b) => b[1] - a[1]);
    return scores.slice(0, topK);
  }

  hybridSearch(
    query: string,
    queryVector: number[],
    topK: number = 5,
    alpha: number = 0.5
  ): { index: number; score: number; document: string }[] {
    /** 混合检索：alpha * 向量分数 + (1-alpha) * 关键词分数 */
    const kwResults = new Map(this.keywordSearch(query, topK * 2));
    const vecResults = new Map(this.vectorSearch(queryVector, topK * 2));

    const allIds = new Set([...kwResults.keys(), ...vecResults.keys()]);
    const combined: { index: number; score: number; document: string }[] = [];

    for (const idx of allIds) {
      const kwScore = kwResults.get(idx) || 0;
      const vecScore = vecResults.get(idx) || 0;
      const finalScore = alpha * vecScore + (1 - alpha) * kwScore;
      combined.push({ index: idx, score: finalScore, document: this.documents[idx] });
    }

    combined.sort((a, b) => b.score - a.score);
    return combined.slice(0, topK);
  }
}
```

`alpha` 参数控制两种检索的权重：
- `alpha=1.0`：纯向量检索
- `alpha=0.0`：纯关键词检索
- `alpha=0.5`：两者各占一半（推荐起步值）

## 查询改写

用户的查询往往不够理想——措辞模糊、视角单一。查询改写让检索更准确。

### 多查询检索（Multi-Query）

把一个查询改写成多个不同表述，分别检索后合并。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function generateMultiQueries(
  originalQuery: string,
  n: number = 3
): Promise<string[]> {
  /** 用 LLM 生成多个查询变体 */
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 512,
    messages: [{
      role: "user",
      content: `请将以下查询改写为 ${n} 个不同的版本。
每个版本应该从不同角度表达相同的信息需求。

原始查询：${originalQuery}

返回 JSON 数组：["查询1", "查询2", "查询3"]`,
    }],
  });
  const text = response.content[0].type === "text" ? response.content[0].text : "[]";
  return JSON.parse(text);
}

async function multiQueryRetrieve(
  query: string,
  collection: any,
  topK: number = 5
): Promise<string[]> {
  /** 多查询检索 */
  const queries = [query, ...(await generateMultiQueries(query, 3))];
  const allDocs: Record<string, { doc: string; distance: number }> = {};

  for (const q of queries) {
    const results = await collection.query({ queryTexts: [q], nResults: topK });
    const ids: string[] = results.ids[0];
    const docs: string[] = results.documents[0];
    const dists: number[] = results.distances[0];

    ids.forEach((docId, i) => {
      if (!(docId in allDocs) || dists[i] < allDocs[docId].distance) {
        allDocs[docId] = { doc: docs[i], distance: dists[i] };
      }
    });
  }

  return Object.values(allDocs)
    .sort((a, b) => a.distance - b.distance)
    .slice(0, topK)
    .map((item) => item.doc);
}
```

### HyDE（Hypothetical Document Embeddings）

先让 LLM 生成一个"假想的理想回答"，然后用这个假想文档的向量去检索。这样做的好处是：假想文档和知识库中的真实文档在形式上更相似（都是陈述句），比问句更容易匹配到。

```typescript
async function hydeRetrieve(
  query: string,
  collection: any,
  topK: number = 5
): Promise<string[]> {
  /** HyDE 检索：用假想文档的向量检索 */
  // 1. 生成假想文档
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 512,
    messages: [{
      role: "user",
      content: `请写一段文字，直接回答以下问题。
不需要标注来源，直接给出信息性的回答。

问题：${query}`,
    }],
  });
  const hypotheticalDoc =
    response.content[0].type === "text" ? response.content[0].text : "";

  // 2. 用假想文档检索（而不是用原始查询）
  const results = await collection.query({
    queryTexts: [hypotheticalDoc],
    nResults: topK,
  });
  return results.documents[0] as string[];
}
```

::: tip 什么时候用哪种查询改写
- **多查询检索**：适合用户措辞可能不准确的场景，多个角度提高召回率
- **HyDE**：适合问答场景，用户提的是问题但知识库存的是答案
- **Step-back**：适合具体问题需要先了解背景知识的场景
:::

## Reranking：精排提升准确度

初始检索（向量 Top-K）是"粗排"，速度快但精度有限。Reranking 是在粗排结果上做"精排"，用更重的模型精细评估每个结果的相关性。

```
查询 → 粗排（Top-50） → 精排（Top-5） → 最终结果
        向量相似度         Cross-Encoder
        毫秒级              百毫秒级
```

### 使用 Cohere Rerank API

```typescript
import cohere from "cohere-ai";

const co = new cohere.CohereClient();

async function rerankResults(
  query: string,
  documents: string[],
  topN: number = 5
): Promise<{ document: string; score: number }[]> {
  /** 使用 Cohere Rerank 重排序 */
  const response = await co.rerank({
    model: "rerank-multilingual-v3.0",
    query,
    documents,
    topN,
  });
  return response.results.map((item) => ({
    document: documents[item.index],
    score: item.relevanceScore,
  }));
}
```

### 使用开源 BGE Reranker

```typescript
/**
 * 使用本地 Reranker 模型（概念代码）
 *
 * 注：sentence_transformers 的 CrossEncoder 是 Python 特有库。
 * 在 TypeScript/Node.js 中可使用 Cohere Rerank API（见上方）或
 * 通过 ONNX Runtime 加载模型。以下为概念性演示。
 */
interface RerankerModel {
  predict(pairs: [string, string][]): number[];
}

function localRerank(
  reranker: RerankerModel,
  query: string,
  documents: string[],
  topN: number = 5
): { document: string; score: number }[] {
  /** 使用本地 Reranker 模型 */
  const pairs: [string, string][] = documents.map((doc) => [query, doc]);
  const scores = reranker.predict(pairs);

  const indexedScores = scores
    .map((score, idx) => ({ idx, score }))
    .sort((a, b) => b.score - a.score);

  return indexedScores.slice(0, topN).map(({ idx, score }) => ({
    document: documents[idx],
    score,
  }));
}
```

## 检索质量评估

优化检索不能靠感觉，需要量化指标。

```typescript
function recallAtK(retrievedIds: string[], relevantIds: Set<string>, k: number): number {
  /** Recall@K：前K个结果中命中了多少相关文档 */
  const retrievedSet = new Set(retrievedIds.slice(0, k));
  const hits = [...retrievedSet].filter((id) => relevantIds.has(id)).length;
  return relevantIds.size > 0 ? hits / relevantIds.size : 0;
}

function mrr(retrievedIds: string[], relevantIds: Set<string>): number {
  /** MRR：第一个相关文档出现在第几位 */
  for (let i = 0; i < retrievedIds.length; i++) {
    if (relevantIds.has(retrievedIds[i])) {
      return 1.0 / (i + 1);
    }
  }
  return 0.0;
}

// 评估示例
const retrieved = ["doc_3", "doc_1", "doc_7", "doc_2", "doc_5"];
const relevant = new Set(["doc_1", "doc_2", "doc_4"]);

console.log(`Recall@3: ${recallAtK(retrieved, relevant, 3).toFixed(2)}`); // 1/3 = 0.33
console.log(`Recall@5: ${recallAtK(retrieved, relevant, 5).toFixed(2)}`); // 2/3 = 0.67
console.log(`MRR: ${mrr(retrieved, relevant).toFixed(2)}`);               // 1/2 = 0.50
```

::: warning 不做评估等于盲人摸象
建议在开发初期就建立一个测试集（50-100 个查询 + 对应的相关文档），每次修改检索策略后都跑一遍评估。否则你不知道优化是在进步还是倒退。
:::

## 完整的检索流水线

把所有策略组合成一个统一的检索管道：

```typescript
interface CandidateDoc {
  document: string;
  distance: number;
  rerank_score?: number;
}

interface RerankerPredictor {
  predict(pairs: [string, string][]): number[];
}

class RetrievalPipeline {
  /** 完整的检索流水线：查询改写 -> 多路召回 -> 重排序 */
  private collection: any;
  private reranker?: RerankerPredictor;

  constructor(collection: any, reranker?: RerankerPredictor) {
    this.collection = collection;
    this.reranker = reranker;
  }

  async retrieve(
    query: string,
    strategy: string = "basic",
    topK: number = 5,
    rerank: boolean = true
  ): Promise<CandidateDoc[]> {
    /** 统一检索接口 */
    // Step 1: 根据策略检索候选（粗排，多取一些）
    const nCandidates = rerank ? topK * 3 : topK;
    let candidates: CandidateDoc[];

    if (strategy === "basic") {
      candidates = await this.basic(query, nCandidates);
    } else if (strategy === "multi_query") {
      candidates = await this.multiQuery(query, nCandidates);
    } else if (strategy === "hyde") {
      candidates = await this.hyde(query, nCandidates);
    } else {
      throw new Error(`未知策略: ${strategy}`);
    }

    // Step 2: Reranking（精排）
    if (rerank && this.reranker && candidates.length > topK) {
      const docs = candidates.map((c) => c.document);
      const scores = this.reranker.predict(
        docs.map((d) => [query, d] as [string, string])
      );
      scores.forEach((score, i) => {
        candidates[i].rerank_score = score;
      });
      candidates.sort((a, b) => (b.rerank_score || 0) - (a.rerank_score || 0));
    }

    return candidates.slice(0, topK);
  }

  private async basic(query: string, n: number): Promise<CandidateDoc[]> {
    const results = await this.collection.query({ queryTexts: [query], nResults: n });
    return (results.documents[0] as string[]).map((d: string, i: number) => ({
      document: d,
      distance: results.distances[0][i],
    }));
  }

  private async multiQuery(query: string, n: number): Promise<CandidateDoc[]> {
    const queries = [query, ...(await generateMultiQueries(query, 3))];
    const allResults: Record<string, CandidateDoc> = {};

    for (const q of queries) {
      const results = await this.collection.query({
        queryTexts: [q],
        nResults: Math.floor(n / 2),
      });
      const docs: string[] = results.documents[0];
      const dists: number[] = results.distances[0];
      const ids: string[] = results.ids[0];

      docs.forEach((doc, i) => {
        const docId = ids[i];
        if (!(docId in allResults) || dists[i] < allResults[docId].distance) {
          allResults[docId] = { document: doc, distance: dists[i] };
        }
      });
    }

    return Object.values(allResults).sort((a, b) => a.distance - b.distance);
  }

  private async hyde(query: string, n: number): Promise<CandidateDoc[]> {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 512,
      messages: [{ role: "user", content: `请直接回答：${query}` }],
    });
    const hypo = response.content[0].type === "text" ? response.content[0].text : "";
    const results = await this.collection.query({ queryTexts: [hypo], nResults: n });
    return (results.documents[0] as string[]).map((d: string, i: number) => ({
      document: d,
      distance: results.distances[0][i],
    }));
  }
}
```

## 小结

- **分块策略**：固定大小（简单）、递归字符（通用）、语义（精准）、Markdown（结构化）
- **混合检索**：向量搜索 + 关键词搜索，alpha 参数控制权重，通常比单一方法效果好
- **查询改写**：多查询提高召回率，HyDE 用假想文档缩小语义鸿沟
- **Reranking**：粗排 Top-50 → 精排 Top-5，Cross-Encoder 比向量相似度更精确
- **评估**：用 Recall@K 和 MRR 量化效果，建立测试集持续跟踪

## 练习

1. **分块对比**：用同一篇长文，分别用固定大小、递归字符、Markdown 分块，对比检索效果。
2. **混合检索实验**：调整 alpha 参数（0.3, 0.5, 0.7），找到你的数据上的最佳值。
3. **评估流水线**：建立 20 个查询-相关文档对的测试集，用 MRR 评估 basic vs multi_query vs hyde。

## 参考资源

- [HyDE: Precise Zero-Shot Dense Retrieval (arXiv:2212.10496)](https://arxiv.org/abs/2212.10496) -- HyDE 论文
- [Cohere Rerank Documentation](https://docs.cohere.com/docs/reranking) -- Cohere Rerank API 文档
- [BGE Reranker (Hugging Face)](https://huggingface.co/BAAI/bge-reranker-v2-m3) -- BGE 开源 Reranker
- [LangChain: Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/) -- LangChain 分块策略文档
- [Pinecone: What is a Vector Database](https://www.pinecone.io/learn/vector-database/) -- 向量数据库概念入门
