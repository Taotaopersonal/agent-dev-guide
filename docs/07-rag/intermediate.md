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

```python
class FixedSizeChunker:
    """固定大小分块"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end].strip())
            start = end - self.overlap
        return [c for c in chunks if c]
```

**优点**：简单、可预测。**缺点**：不尊重语义边界，可能在句子中间切断。

### 策略二：递归字符分块

LangChain 的默认策略。按多级分隔符递归分割：先尝试按段落分，段落太长就按句子分，句子太长就按字符分。

```python
class RecursiveChunker:
    """递归字符分块"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50,
                 separators: list[str] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", " ", ""]

    def chunk(self, text: str) -> list[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else [""]

        if sep:
            parts = text.split(sep)
        else:
            # 最后的兜底：按字符切
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.overlap)]

        chunks = []
        current = ""
        for part in parts:
            test = current + sep + part if current else part
            if len(test) <= self.chunk_size:
                current = test
            else:
                if current:
                    chunks.append(current.strip())
                # 如果单个 part 太长，用下一级分隔符继续切
                if len(part) > self.chunk_size:
                    chunks.extend(self._recursive_split(part, remaining_seps))
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks
```

**优点**：尊重自然段落和句子边界。**缺点**：实现稍复杂，分块大小不均匀。

### 策略三：语义分块

用 Embedding 检测语义变化点来决定分块边界。当相邻句子的语义相似度突然下降，说明话题发生了转换，这就是一个好的分块点。

```python
import numpy as np
from openai import OpenAI

openai_client = OpenAI()

class SemanticChunker:
    """基于语义的分块"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold  # 相似度阈值，低于此值就分块

    def chunk(self, text: str) -> list[str]:
        # 按句子分割
        sentences = [s.strip() for s in text.replace("。", "。\n").split("\n") if s.strip()]
        if len(sentences) <= 1:
            return sentences

        # 获取每个句子的 Embedding
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=sentences,
        )
        embeddings = [item.embedding for item in response.data]

        # 计算相邻句子的相似度
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < self.threshold:
                # 语义断裂，开始新块
                chunks.append("".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    @staticmethod
    def _cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

**优点**：分块边界与语义一致。**缺点**：需要额外的 Embedding API 调用，成本较高。

### 策略四：Markdown 结构分块

对于 Markdown 格式的文档，利用标题层级来分块。

```python
class MarkdownChunker:
    """基于 Markdown 结构的分块"""

    def chunk(self, text: str) -> list[str]:
        chunks = []
        current_headers = []
        current_content = []

        for line in text.split("\n"):
            if line.startswith("#"):
                # 遇到新标题，保存之前的内容
                if current_content:
                    header_prefix = " > ".join(current_headers)
                    content = "\n".join(current_content).strip()
                    if content:
                        chunks.append(f"[{header_prefix}]\n{content}")
                    current_content = []

                # 更新标题层级
                level = len(line) - len(line.lstrip("#"))
                title = line.lstrip("# ").strip()
                current_headers = current_headers[:level - 1] + [title]
            else:
                current_content.append(line)

        # 最后一块
        if current_content:
            header_prefix = " > ".join(current_headers)
            content = "\n".join(current_content).strip()
            if content:
                chunks.append(f"[{header_prefix}]\n{content}")

        return chunks
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

```python
import numpy as np

class HybridSearch:
    """向量 + 关键词混合检索"""

    def __init__(self):
        self.documents = []
        self.vectors = []

    def add(self, doc: str, vector: list[float]):
        self.documents.append(doc)
        self.vectors.append(vector)

    def keyword_search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """BM25 风格的关键词检索（简化版）"""
        query_terms = set(query.lower().split())
        scores = []
        for i, doc in enumerate(self.documents):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / (len(query_terms) + 1)
            scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def vector_search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        """向量检索"""
        q = np.array(query_vector)
        scores = []
        for i, v in enumerate(self.vectors):
            v = np.array(v)
            sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v)))
            scores.append((i, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def hybrid_search(self, query: str, query_vector: list[float],
                      top_k: int = 5, alpha: float = 0.5) -> list[tuple[int, float, str]]:
        """混合检索：alpha * 向量分数 + (1-alpha) * 关键词分数"""
        kw_results = dict(self.keyword_search(query, top_k * 2))
        vec_results = dict(self.vector_search(query_vector, top_k * 2))

        all_ids = set(kw_results.keys()) | set(vec_results.keys())
        combined = []
        for idx in all_ids:
            kw_score = kw_results.get(idx, 0)
            vec_score = vec_results.get(idx, 0)
            final_score = alpha * vec_score + (1 - alpha) * kw_score
            combined.append((idx, final_score, self.documents[idx]))

        combined.sort(key=lambda x: -x[1])
        return combined[:top_k]
```

`alpha` 参数控制两种检索的权重：
- `alpha=1.0`：纯向量检索
- `alpha=0.0`：纯关键词检索
- `alpha=0.5`：两者各占一半（推荐起步值）

## 查询改写

用户的查询往往不够理想——措辞模糊、视角单一。查询改写让检索更准确。

### 多查询检索（Multi-Query）

把一个查询改写成多个不同表述，分别检索后合并。

```python
import anthropic
import json

client = anthropic.Anthropic()

def generate_multi_queries(original_query: str, n: int = 3) -> list[str]:
    """用 LLM 生成多个查询变体"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""请将以下查询改写为 {n} 个不同的版本。
每个版本应该从不同角度表达相同的信息需求。

原始查询：{original_query}

返回 JSON 数组：["查询1", "查询2", "查询3"]"""
        }]
    )
    return json.loads(response.content[0].text)

def multi_query_retrieve(query: str, collection, top_k: int = 5) -> list[str]:
    """多查询检索"""
    queries = [query] + generate_multi_queries(query, 3)
    all_docs = {}  # doc_id -> (doc, min_distance)

    for q in queries:
        results = collection.query(query_texts=[q], n_results=top_k)
        for doc_id, doc, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        ):
            if doc_id not in all_docs or dist < all_docs[doc_id][1]:
                all_docs[doc_id] = (doc, dist)

    sorted_docs = sorted(all_docs.values(), key=lambda x: x[1])
    return [doc for doc, _ in sorted_docs[:top_k]]
```

### HyDE（Hypothetical Document Embeddings）

先让 LLM 生成一个"假想的理想回答"，然后用这个假想文档的向量去检索。这样做的好处是：假想文档和知识库中的真实文档在形式上更相似（都是陈述句），比问句更容易匹配到。

```python
def hyde_retrieve(query: str, collection, top_k: int = 5) -> list[str]:
    """HyDE 检索：用假想文档的向量检索"""
    # 1. 生成假想文档
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""请写一段文字，直接回答以下问题。
不需要标注来源，直接给出信息性的回答。

问题：{query}"""
        }]
    )
    hypothetical_doc = response.content[0].text

    # 2. 用假想文档检索（而不是用原始查询）
    results = collection.query(query_texts=[hypothetical_doc], n_results=top_k)
    return results["documents"][0]
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

```python
import cohere

co = cohere.Client()

def rerank_results(query: str, documents: list[str], top_n: int = 5) -> list[dict]:
    """使用 Cohere Rerank 重排序"""
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=documents,
        top_n=top_n,
    )
    return [
        {"document": documents[item.index], "score": item.relevance_score}
        for item in response.results
    ]
```

### 使用开源 BGE Reranker

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

def local_rerank(query: str, documents: list[str], top_n: int = 5) -> list[dict]:
    """使用本地 Reranker 模型"""
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)

    indexed_scores = sorted(enumerate(scores), key=lambda x: -x[1])
    return [
        {"document": documents[idx], "score": float(score)}
        for idx, score in indexed_scores[:top_n]
    ]
```

## 检索质量评估

优化检索不能靠感觉，需要量化指标。

```python
import numpy as np

def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Recall@K：前K个结果中命中了多少相关文档"""
    retrieved_set = set(retrieved_ids[:k])
    hits = len(retrieved_set & relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0

def mrr(retrieved_ids: list, relevant_ids: set) -> float:
    """MRR：第一个相关文档出现在第几位"""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

# 评估示例
retrieved = ["doc_3", "doc_1", "doc_7", "doc_2", "doc_5"]
relevant = {"doc_1", "doc_2", "doc_4"}

print(f"Recall@3: {recall_at_k(retrieved, relevant, 3):.2f}")  # 1/3 = 0.33
print(f"Recall@5: {recall_at_k(retrieved, relevant, 5):.2f}")  # 2/3 = 0.67
print(f"MRR: {mrr(retrieved, relevant):.2f}")                  # 1/2 = 0.50
```

::: warning 不做评估等于盲人摸象
建议在开发初期就建立一个测试集（50-100 个查询 + 对应的相关文档），每次修改检索策略后都跑一遍评估。否则你不知道优化是在进步还是倒退。
:::

## 完整的检索流水线

把所有策略组合成一个统一的检索管道：

```python
class RetrievalPipeline:
    """完整的检索流水线：查询改写 → 多路召回 → 重排序"""

    def __init__(self, collection, reranker=None):
        self.collection = collection
        self.reranker = reranker

    def retrieve(self, query: str, strategy: str = "basic",
                 top_k: int = 5, rerank: bool = True) -> list[dict]:
        """统一检索接口"""
        # Step 1: 根据策略检索候选（粗排，多取一些）
        n_candidates = top_k * 3 if rerank else top_k
        if strategy == "basic":
            candidates = self._basic(query, n_candidates)
        elif strategy == "multi_query":
            candidates = self._multi_query(query, n_candidates)
        elif strategy == "hyde":
            candidates = self._hyde(query, n_candidates)
        else:
            raise ValueError(f"未知策略: {strategy}")

        # Step 2: Reranking（精排）
        if rerank and self.reranker and len(candidates) > top_k:
            docs = [c["document"] for c in candidates]
            scores = self.reranker.predict([[query, d] for d in docs])
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)
            candidates.sort(key=lambda x: -x.get("rerank_score", 0))

        return candidates[:top_k]

    def _basic(self, query, n):
        results = self.collection.query(query_texts=[query], n_results=n)
        return [{"document": d, "distance": dist}
                for d, dist in zip(results["documents"][0], results["distances"][0])]

    def _multi_query(self, query, n):
        queries = [query] + generate_multi_queries(query, 3)
        all_results = {}
        for q in queries:
            results = self.collection.query(query_texts=[q], n_results=n // 2)
            for doc, dist, doc_id in zip(
                results["documents"][0], results["distances"][0], results["ids"][0]
            ):
                if doc_id not in all_results or dist < all_results[doc_id]["distance"]:
                    all_results[doc_id] = {"document": doc, "distance": dist}
        return sorted(all_results.values(), key=lambda x: x["distance"])

    def _hyde(self, query, n):
        hypo = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=512,
            messages=[{"role": "user", "content": f"请直接回答：{query}"}]
        ).content[0].text
        results = self.collection.query(query_texts=[hypo], n_results=n)
        return [{"document": d, "distance": dist}
                for d, dist in zip(results["documents"][0], results["distances"][0])]
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
