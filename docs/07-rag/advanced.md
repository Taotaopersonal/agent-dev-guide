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

```python
"""Agentic RAG -- Agent 自主决策的 RAG"""
import anthropic
import chromadb
import json

anthropic_client = anthropic.Anthropic()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("agentic_rag")

# 预置知识库（实际中应该有大量文档）
collection.add(
    documents=[
        "Python 3.12 引入了 type 语句，简化类型别名定义。",
        "FastAPI 使用 Pydantic 进行数据验证，支持自动生成 API 文档。",
        "Django ORM 支持惰性查询，只在真正需要数据时才执行 SQL。",
        "Docker 容器共享主机内核，比虚拟机更轻量。",
        "Kubernetes 通过 Pod 管理容器，支持自动扩缩容。",
    ],
    ids=[f"doc_{i}" for i in range(5)]
)


# 将检索定义为工具
tools = [
    {
        "name": "search_knowledge_base",
        "description": (
            "搜索内部知识库获取技术文档信息。"
            "当用户的问题涉及特定技术细节、产品文档、内部规范时使用。"
            "对于常识性问题、数学计算、简单推理，不需要使用此工具。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，尽量精确"
                },
                "n_results": {
                    "type": "integer",
                    "description": "返回结果数量，默认3"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_with_filter",
        "description": "带过滤条件搜索知识库。当需要限定搜索范围时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": ["python", "web", "devops", "database"],
                    "description": "文档分类"
                }
            },
            "required": ["query"]
        }
    }
]


def search_knowledge_base(query: str, n_results: int = 3) -> dict:
    results = collection.query(query_texts=[query], n_results=n_results)
    return {
        "query": query,
        "results": results["documents"][0] if results["documents"][0] else [],
        "count": len(results["documents"][0])
    }

def search_with_filter(query: str, category: str = None) -> dict:
    kwargs = {"query_texts": [query], "n_results": 3}
    if category:
        kwargs["where"] = {"category": category}
    results = collection.query(**kwargs)
    return {"results": results["documents"][0], "filter": category}


tool_map = {
    "search_knowledge_base": search_knowledge_base,
    "search_with_filter": search_with_filter
}


def agentic_rag(question: str) -> str:
    """Agentic RAG：Agent 自主决定是否检索"""
    messages = [{"role": "user", "content": question}]
    system = (
        "你是一个技术助手。回答技术问题时，如果你确定答案就直接回答；"
        "如果不确定或需要查询具体信息，使用搜索工具。"
        "基于搜索结果回答时，明确引用来源。"
    )

    for _ in range(5):
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024,
            system=system, tools=tools, messages=messages
        )

        if response.stop_reason == "end_turn":
            return "".join(b.text for b in response.content if hasattr(b, "text"))

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    func = tool_map[block.name]
                    result = func(**block.input)
                    print(f"  [检索] {block.name}: {block.input.get('query', '')}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
            messages.append({"role": "user", "content": results})

    return "处理超时"


# 测试：Agent 会自动判断是否需要检索
print("=== 不需要检索的问题 ===")
print(agentic_rag("1+1 等于几？"))

print("\n=== 需要检索的问题 ===")
print(agentic_rag("Python 3.12 的 type 语句怎么用？"))

print("\n=== 需要多次检索的问题 ===")
print(agentic_rag("对比 FastAPI 和 Django 在数据验证方面的差异"))
```

## Self-RAG：自我评估检索质量

Self-RAG 让模型在生成答案之前，先评估检索到的内容是否真的有用：

```python
"""Self-RAG -- 自我评估检索质量"""
import anthropic
import json

client = anthropic.Anthropic()


class SelfRAG:
    """带自我评估的 RAG 系统"""

    def __init__(self, search_func):
        self.search = search_func  # 检索函数

    def evaluate_retrieval(self, query: str, documents: list[str]) -> dict:
        """评估检索结果的质量"""
        doc_list = "\n".join([f"[{i+1}] {d}" for i, d in enumerate(documents)])

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": f"""评估以下检索结果对回答查询的有用程度。

查询：{query}

检索结果：
{doc_list}

返回 JSON：
{{
    "overall_relevance": "high/medium/low/none",
    "useful_docs": [1, 3],
    "missing_info": "缺少什么信息",
    "should_retry": true/false,
    "retry_query": "如果需要重试，用什么查询"
}}"""}]
        )

        text = response.content[0].text.strip()
        try:
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            return json.loads(text)
        except json.JSONDecodeError:
            return {"overall_relevance": "unknown", "should_retry": False}

    def evaluate_answer(self, query: str, answer: str,
                        sources: list[str]) -> dict:
        """评估生成答案的质量（是否有幻觉）"""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": f"""评估答案是否完全基于参考来源。

问题：{query}
答案：{answer}
参考来源：{json.dumps(sources, ensure_ascii=False)}

返回 JSON：
{{
    "is_supported": true/false,
    "has_hallucination": true/false,
    "confidence": 0.0-1.0,
    "unsupported_claims": ["不支持的说法"]
}}"""}]
        )
        text = response.content[0].text.strip()
        try:
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            return json.loads(text)
        except json.JSONDecodeError:
            return {"is_supported": True, "confidence": 0.5}

    def query(self, question: str, max_retries: int = 2) -> dict:
        """Self-RAG 完整流程"""
        current_query = question

        for attempt in range(max_retries + 1):
            # 1. 检索
            docs = self.search(current_query)
            print(f"[第{attempt+1}轮检索] 查询: {current_query}")

            # 2. 评估检索质量
            eval_result = self.evaluate_retrieval(question, docs)
            print(f"[评估] 相关性: {eval_result['overall_relevance']}")

            if eval_result.get("overall_relevance") == "none" and eval_result.get("should_retry"):
                current_query = eval_result.get("retry_query", question)
                print(f"[重试] 新查询: {current_query}")
                continue

            # 3. 过滤有用文档
            useful_indices = eval_result.get("useful_docs", list(range(1, len(docs)+1)))
            useful_docs = [docs[i-1] for i in useful_indices if 0 < i <= len(docs)]

            # 4. 生成答案
            context = "\n".join(useful_docs)
            response = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=500,
                messages=[{"role": "user", "content": f"""基于参考回答问题。只用参考中的信息。

参考：{context}
问题：{question}"""}]
            )
            answer = response.content[0].text

            # 5. 评估答案质量
            answer_eval = self.evaluate_answer(question, answer, useful_docs)
            print(f"[答案评估] 置信度: {answer_eval.get('confidence', 'N/A')}")

            return {
                "answer": answer,
                "sources": useful_docs,
                "retrieval_quality": eval_result["overall_relevance"],
                "answer_confidence": answer_eval.get("confidence", 0),
                "has_hallucination": answer_eval.get("has_hallucination", False),
                "attempts": attempt + 1
            }

        return {"answer": "无法找到足够的信息来回答这个问题。", "attempts": max_retries + 1}
```

## CRAG：修正性 RAG

CRAG（Corrective RAG）在 Self-RAG 基础上更进一步：如果检索结果不够好，它不只是重试，而是**切换数据源**或**改变策略**：

```python
"""CRAG -- 修正性 RAG 概念实现"""

class CorrectiveRAG:
    """修正性 RAG：根据检索质量动态调整策略"""

    def __init__(self, primary_search, web_search, llm_generate):
        self.primary_search = primary_search  # 主知识库检索
        self.web_search = web_search          # 网络搜索（备用）
        self.generate = llm_generate          # LLM 生成

    def assess_relevance(self, query: str, doc: str) -> str:
        """评估单个文档的相关性"""
        # 实际中用 LLM 或分类器
        # 返回 "correct", "incorrect", "ambiguous"
        return "correct"  # 简化

    def query(self, question: str) -> dict:
        # 1. 主检索
        docs = self.primary_search(question)

        # 2. 逐个评估文档
        correct_docs = []
        for doc in docs:
            relevance = self.assess_relevance(question, doc)
            if relevance == "correct":
                correct_docs.append(doc)

        # 3. 根据评估结果决定策略
        if len(correct_docs) >= 2:
            # 策略A：足够多的相关文档，直接生成
            strategy = "direct"
            final_docs = correct_docs
        elif len(correct_docs) == 1:
            # 策略B：部分相关，补充网络搜索
            strategy = "augmented"
            web_docs = self.web_search(question)
            final_docs = correct_docs + web_docs[:2]
        else:
            # 策略C：完全不相关，重写查询 + 全网搜索
            strategy = "web_fallback"
            final_docs = self.web_search(question)

        print(f"[CRAG] 策略: {strategy}, 文档数: {len(final_docs)}")

        # 4. 生成答案
        answer = self.generate(question, final_docs)
        return {"answer": answer, "strategy": strategy, "sources": final_docs}
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

以下是一个教学级 Graph RAG 实现，用 `networkx` 构建知识图谱，用 Claude 做实体提取和答案生成：

```python
"""Graph RAG -- 知识图谱增强的 RAG 系统"""
import anthropic
import networkx as nx
import json

client = anthropic.Anthropic()


# ============================================================
# 第一步：从文档中提取实体和关系（用 LLM）
# ============================================================

def extract_entities_and_relations(text: str) -> list[dict]:
    """用 LLM 从文本中提取实体和关系三元组"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""从以下文本中提取所有实体和它们之间的关系。

文本：
{text}

返回 JSON 数组，每个元素是一个三元组：
[
    {{"subject": "实体A", "predicate": "关系", "object": "实体B"}},
    ...
]

规则：
- subject 和 object 是具体的实体（人名、组织、技术、项目等）
- predicate 是关系描述（如"负责"、"使用"、"属于"、"依赖"等）
- 尽量提取所有能识别的关系
- 只返回 JSON 数组，不要其他内容"""}]
    )

    text_result = response.content[0].text.strip()
    try:
        # 处理可能被 markdown 代码块包裹的情况
        if "```" in text_result:
            text_result = text_result.split("```")[1].replace("json", "").strip()
        return json.loads(text_result)
    except json.JSONDecodeError:
        return []


# ============================================================
# 第二步：构建知识图谱
# ============================================================

class KnowledgeGraph:
    """基于 networkx 的知识图谱"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()  # 有向多重图（两个节点间可有多种关系）
        self.source_texts: dict[str, str] = {}  # 记录每个三元组的来源文本

    def add_from_documents(self, documents: list[str]):
        """从文档列表批量构建图谱"""
        for i, doc in enumerate(documents):
            print(f"[构建图谱] 处理文档 {i+1}/{len(documents)}...")
            triples = extract_entities_and_relations(doc)

            for triple in triples:
                subj = triple["subject"]
                pred = triple["predicate"]
                obj = triple["object"]

                # 添加节点（如果不存在）
                self.graph.add_node(subj, type="entity")
                self.graph.add_node(obj, type="entity")

                # 添加带标签的边
                self.graph.add_edge(subj, obj, relation=pred)

                # 记录来源
                key = f"{subj}-{pred}-{obj}"
                self.source_texts[key] = doc

            print(f"  提取了 {len(triples)} 个三元组")

        print(f"[图谱完成] {self.graph.number_of_nodes()} 个节点, "
              f"{self.graph.number_of_edges()} 条边")

    def get_neighbors(self, entity: str, max_depth: int = 2) -> list[dict]:
        """获取实体的邻居信息（支持多跳）

        Args:
            entity: 起始实体名称
            max_depth: 最大遍历深度（1=直接关系，2=两跳关系）

        Returns:
            关系三元组列表
        """
        if entity not in self.graph:
            return []

        relations = []
        visited = set()

        # BFS 遍历：从起始实体出发，逐层扩展
        queue = [(entity, 0)]  # (当前节点, 当前深度)
        visited.add(entity)

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            # 出边：current -> neighbor
            for _, neighbor, data in self.graph.out_edges(current, data=True):
                relations.append({
                    "from": current,
                    "relation": data["relation"],
                    "to": neighbor,
                    "depth": depth + 1
                })
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

            # 入边：neighbor -> current
            for neighbor, _, data in self.graph.in_edges(current, data=True):
                relations.append({
                    "from": neighbor,
                    "relation": data["relation"],
                    "to": current,
                    "depth": depth + 1
                })
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return relations

    def find_path(self, entity_a: str, entity_b: str) -> list[dict] | None:
        """查找两个实体之间的关系路径"""
        if entity_a not in self.graph or entity_b not in self.graph:
            return None

        try:
            # 在无向视图上找最短路径
            undirected = self.graph.to_undirected()
            path_nodes = nx.shortest_path(undirected, entity_a, entity_b)
        except nx.NetworkXNoPathError:
            return None

        # 还原路径上每条边的关系
        path_relations = []
        for i in range(len(path_nodes) - 1):
            src, dst = path_nodes[i], path_nodes[i + 1]
            # 尝试正向边
            edge_data = self.graph.get_edge_data(src, dst)
            if edge_data:
                first_edge = list(edge_data.values())[0]
                path_relations.append({
                    "from": src, "relation": first_edge["relation"], "to": dst
                })
            else:
                # 尝试反向边
                edge_data = self.graph.get_edge_data(dst, src)
                if edge_data:
                    first_edge = list(edge_data.values())[0]
                    path_relations.append({
                        "from": dst, "relation": first_edge["relation"], "to": src
                    })

        return path_relations


# ============================================================
# 第三步：Graph RAG 检索 + 生成
# ============================================================

def identify_entities_in_query(query: str, known_entities: list[str]) -> list[str]:
    """识别查询中提到的实体"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": f"""从查询中识别与已知实体匹配的实体名称。

查询：{query}
已知实体：{json.dumps(known_entities, ensure_ascii=False)}

返回 JSON 数组，只包含在查询中出现或被提及的实体名。只返回 JSON 数组。"""}]
    )
    text = response.content[0].text.strip()
    try:
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return []


def graph_rag_query(query: str, kg: KnowledgeGraph) -> str:
    """Graph RAG 完整查询流程"""
    all_entities = list(kg.graph.nodes())

    # 1. 识别查询中涉及的实体
    query_entities = identify_entities_in_query(query, all_entities)
    print(f"[Graph RAG] 识别到实体: {query_entities}")

    # 2. 图检索：获取相关子图
    all_relations = []
    for entity in query_entities:
        # 获取每个实体的 2 跳邻居关系
        neighbors = kg.get_neighbors(entity, max_depth=2)
        all_relations.extend(neighbors)

    # 如果有多个实体，还查找它们之间的路径
    if len(query_entities) >= 2:
        for i in range(len(query_entities)):
            for j in range(i + 1, len(query_entities)):
                path = kg.find_path(query_entities[i], query_entities[j])
                if path:
                    print(f"[Graph RAG] 找到路径: {query_entities[i]} -> {query_entities[j]}")
                    all_relations.extend(
                        {**r, "depth": 0, "is_path": True} for r in path
                    )

    # 3. 去重并格式化为上下文
    seen = set()
    unique_relations = []
    for r in all_relations:
        key = f"{r['from']}-{r['relation']}-{r['to']}"
        if key not in seen:
            seen.add(key)
            unique_relations.append(r)

    if not unique_relations:
        return "未找到相关的知识图谱信息。"

    # 格式化关系为自然语言上下文
    context_lines = []
    for r in unique_relations:
        context_lines.append(f"- {r['from']} --[{r['relation']}]--> {r['to']}")

    context = "\n".join(context_lines)
    print(f"[Graph RAG] 检索到 {len(unique_relations)} 条关系")

    # 4. 收集相关的原始文本片段
    source_snippets = set()
    for r in unique_relations:
        key = f"{r['from']}-{r['relation']}-{r['to']}"
        if key in kg.source_texts:
            source_snippets.add(kg.source_texts[key])

    source_context = ""
    if source_snippets:
        source_context = "\n\n原始文档片段：\n" + "\n---\n".join(source_snippets)

    # 5. LLM 生成最终答案
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": f"""基于以下知识图谱关系和原始文档回答问题。

知识图谱关系：
{context}
{source_context}

问题：{query}

请基于以上信息回答，注意利用关系链路进行推理。如果信息不足以回答，请说明。"""}]
    )

    return response.content[0].text


# ============================================================
# 运行示例
# ============================================================

# 示例文档（模拟一个公司的技术团队信息）
documents = [
    "张伟是后端团队的技术负责人，他主导了订单系统的重构项目。后端团队使用 Python 和 FastAPI 框架。",
    "李娜是前端团队的负责人，前端团队使用 React 和 TypeScript。前端团队与后端团队共同协作开发电商平台。",
    "订单系统依赖用户服务和支付服务。支付服务由王强负责开发，使用了 Go 语言。",
    "电商平台是公司的核心产品，由产品部的赵敏负责规划。电商平台包含订单系统、用户系统和推荐系统。",
    "推荐系统使用机器学习技术，由数据团队的陈磊负责。推荐系统依赖用户行为数据。",
]

# 构建知识图谱
kg = KnowledgeGraph()
kg.add_from_documents(documents)

# 测试：单跳查询（直接关系）
print("\n=== 单跳查询 ===")
print(graph_rag_query("张伟负责什么项目？", kg))

# 测试：多跳查询（需要推理链）
print("\n=== 多跳查询 ===")
print(graph_rag_query("订单系统依赖的支付服务是谁开发的？用了什么语言？", kg))

# 测试：全局关联查询
print("\n=== 全局关联查询 ===")
print(graph_rag_query("电商平台涉及哪些团队和技术栈？", kg))
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

```python
"""RAG 评估框架"""
import anthropic
import json

client = anthropic.Anthropic()


class RAGEvaluator:
    """RAG 系统评估器"""

    def evaluate_single(self, question: str, answer: str,
                        retrieved_docs: list[str],
                        ground_truth: str = None) -> dict:
        """评估单个 QA 对"""
        metrics = {}

        # 1. 相关性（Answer Relevancy）：答案是否回答了问题
        metrics["relevancy"] = self._score_relevancy(question, answer)

        # 2. 忠实度（Faithfulness）：答案是否基于检索到的文档
        metrics["faithfulness"] = self._score_faithfulness(answer, retrieved_docs)

        # 3. 上下文精度（Context Precision）：检索到的文档是否相关
        metrics["context_precision"] = self._score_context_precision(
            question, retrieved_docs
        )

        # 4. 如果有标准答案，计算正确性
        if ground_truth:
            metrics["correctness"] = self._score_correctness(
                answer, ground_truth
            )

        # 总分
        scores = [v for v in metrics.values() if isinstance(v, (int, float))]
        metrics["overall"] = sum(scores) / len(scores) if scores else 0

        return metrics

    def _score_relevancy(self, question: str, answer: str) -> float:
        """答案是否切题"""
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=50,
            messages=[{"role": "user", "content": f"""评估答案与问题的相关性。
问题：{question}
答案：{answer}
返回 0.0-1.0 的分数，只返回数字。"""}]
        )
        try:
            return float(response.content[0].text.strip())
        except ValueError:
            return 0.5

    def _score_faithfulness(self, answer: str, docs: list[str]) -> float:
        """答案是否忠实于来源文档"""
        context = "\n".join(docs)
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=50,
            messages=[{"role": "user", "content": f"""评估答案是否完全基于参考文档（没有编造信息）。
参考文档：{context}
答案：{answer}
返回 0.0-1.0 的分数，只返回数字。"""}]
        )
        try:
            return float(response.content[0].text.strip())
        except ValueError:
            return 0.5

    def _score_context_precision(self, question: str, docs: list[str]) -> float:
        """检索到的文档是否与问题相关"""
        doc_list = "\n".join([f"[{i+1}] {d}" for i, d in enumerate(docs)])
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=50,
            messages=[{"role": "user", "content": f"""评估检索到的文档与问题的相关程度。
问题：{question}
文档：{doc_list}
返回 0.0-1.0 的分数，只返回数字。"""}]
        )
        try:
            return float(response.content[0].text.strip())
        except ValueError:
            return 0.5

    def _score_correctness(self, answer: str, ground_truth: str) -> float:
        """答案是否正确"""
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=50,
            messages=[{"role": "user", "content": f"""对比答案与标准答案，评估正确性。
标准答案：{ground_truth}
实际答案：{answer}
返回 0.0-1.0 的分数，只返回数字。"""}]
        )
        try:
            return float(response.content[0].text.strip())
        except ValueError:
            return 0.5

    def evaluate_batch(self, test_cases: list[dict]) -> dict:
        """批量评估"""
        all_results = []
        for case in test_cases:
            result = self.evaluate_single(**case)
            all_results.append(result)
            print(f"  Q: {case['question'][:40]}... | 总分: {result['overall']:.2f}")

        # 聚合指标
        avg_metrics = {}
        for key in all_results[0]:
            values = [r[key] for r in all_results if isinstance(r.get(key), (int, float))]
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)

        return {"individual": all_results, "aggregate": avg_metrics}
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
2. **Graph RAG 扩展**：在 KnowledgeGraph 类中实现社区检测（提示：用 `networkx` 的连通分量或 Louvain 算法），将社区摘要作为全局查询的上下文。
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
