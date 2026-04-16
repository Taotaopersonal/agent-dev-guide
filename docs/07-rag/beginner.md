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

```python
"""
Embedding 基础演示
运行前：pip install openai numpy
"""
from openai import OpenAI
import numpy as np

openai_client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """获取文本的 Embedding 向量"""
    response = openai_client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding

# 试试看
vec1 = get_embedding("Python 是一门编程语言")
vec2 = get_embedding("Java 是一种编程语言")
vec3 = get_embedding("今天天气真不错")

print(f"向量维度: {len(vec1)}")  # text-embedding-3-small 输出 1536 维

# 计算余弦相似度
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"编程语言 vs 编程语言: {cosine_similarity(vec1, vec2):.4f}")  # 高相似度
print(f"编程语言 vs 天气: {cosine_similarity(vec1, vec3):.4f}")      # 低相似度
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

```python
"""
ChromaDB 入门：构建和检索知识库
运行前：pip install chromadb
"""
import chromadb
from chromadb.utils import embedding_functions

# 初始化 ChromaDB（持久化模式，数据存到磁盘）
client = chromadb.PersistentClient(path="./my_knowledge_base")

# 使用 OpenAI embedding（也可以用默认的免费 embedding）
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small",
)

# 创建一个集合（类似数据库的表）
collection = client.get_or_create_collection(
    name="company_docs",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},  # 使用余弦距离
)

# === 添加文档 ===
documents = [
    "退款政策：购买后7天内可以无条件退款，需提供订单号。",
    "会员等级：消费满1000元升级为银卡会员，享受9折优惠。",
    "配送说明：普通快递3-5天送达，顺丰次日达（需加10元）。",
    "售后服务：产品质量问题30天内免费换货，人为损坏不在保修范围内。",
    "积分规则：每消费1元获得1积分，积分可在下次购物时抵扣（100积分=1元）。",
    "营业时间：线下门店周一至周五 9:00-18:00，周末 10:00-17:00。",
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[
        {"category": "退款"},
        {"category": "会员"},
        {"category": "配送"},
        {"category": "售后"},
        {"category": "积分"},
        {"category": "门店"},
    ],
)

print(f"知识库中共 {collection.count()} 条文档")

# === 检索 ===
results = collection.query(
    query_texts=["我买的东西想退货怎么办"],
    n_results=3,
)

print("\n检索结果：")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  [{dist:.4f}] {doc}")

# === 带元数据过滤的检索 ===
results = collection.query(
    query_texts=["会员有什么优惠"],
    n_results=3,
    where={"category": "会员"},  # 只在"会员"分类中搜索
)

print("\n过滤检索（仅会员类）：")
for doc in results["documents"][0]:
    print(f"  {doc}")
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

```python
"""
完整的 RAG 文档问答系统
运行前：pip install anthropic chromadb openai
"""
import anthropic
import chromadb
from chromadb.utils import embedding_functions

anthropic_client = anthropic.Anthropic()

class SimpleRAG:
    """简单但完整的 RAG 系统"""

    def __init__(self, collection_name: str = "knowledge_base"):
        # 向量数据库
        self.chroma = chromadb.PersistentClient(path="./rag_db")
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
        )
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
        )

    # === Step 1: 加载文档 ===
    def load_documents(self, documents: list[str], metadatas: list[dict] = None):
        """将文档加入知识库"""
        ids = [f"doc_{self.collection.count() + i}" for i in range(len(documents))]
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas or [{}] * len(documents),
        )
        print(f"已加载 {len(documents)} 条文档，总计 {self.collection.count()} 条")

    # === Step 2: 分块（简单版：按段落分割） ===
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """将长文本分成小块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - overlap  # 有重叠，避免在边界丢失信息
        return [c for c in chunks if c]  # 过滤空块

    # === Step 3 & 4: 检索 ===
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """检索最相关的文档片段"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        return results["documents"][0]

    # === Step 5: 生成回答 ===
    def answer(self, question: str, top_k: int = 3) -> str:
        """检索 + 生成：完整的 RAG 流程"""
        # 检索相关文档
        relevant_docs = self.retrieve(question, top_k)

        if not relevant_docs:
            return "知识库中没有找到相关信息。"

        # 构建上下文
        context = "\n---\n".join(relevant_docs)

        # 生成回答
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""基于以下参考资料回答用户的问题。
如果参考资料中没有相关信息，如实告知用户你不确定，不要编造。
回答时引用信息来源。

参考资料：
{context}

用户问题：{question}"""
            }]
        )
        return response.content[0].text


# === 使用示例 ===
if __name__ == "__main__":
    rag = SimpleRAG()

    # 加载公司知识库
    docs = [
        "退款政策：购买后7天内可以无条件退款，需提供订单号和支付凭证。超过7天但在30天内，"
        "如果产品未拆封可以退款，需扣除10%手续费。超过30天不接受退款。",

        "会员体系：普通会员消费满1000元自动升级为银卡会员（9折），满5000元升级为金卡（8.5折），"
        "满20000元升级为钻石卡（8折）。会员等级每年1月1日重新计算。",

        "配送方式：支持普通快递（免费，3-5个工作日）、顺丰快递（加10元，次日达）、"
        "同城配送（加5元，当日达，仅限北上广深）。订单满199元免运费。",

        "积分规则：每消费1元获得1积分。积分可用于兑换优惠券或直接抵扣（100积分=1元）。"
        "积分有效期为获得之日起12个月，过期清零。每笔订单积分抵扣不超过订单金额的20%。",
    ]
    rag.load_documents(docs)

    # 问几个问题
    questions = [
        "我买了15天了还能退货吗？",
        "怎样才能成为金卡会员？有什么优惠？",
        "我在上海，最快什么时候能收到货？",
        "积分能抵多少钱？会过期吗？",
    ]

    for q in questions:
        print(f"\n问：{q}")
        print(f"答：{rag.answer(q)}")
        print("-" * 60)
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
