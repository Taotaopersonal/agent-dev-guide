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

```python
"""MemGPT 风格的记忆管理"""
import anthropic
import json
import chromadb
from datetime import datetime

client = anthropic.Anthropic()
chroma = chromadb.Client()


class MemGPTAgent:
    """MemGPT 风格的 Agent：LLM 自主管理记忆"""

    def __init__(self):
        # 核心记忆（始终在上下文中）
        self.core_memory = {
            "user_info": "",    # 用户基本信息
            "agent_info": "我是一个有长期记忆的AI助手。",
            "preferences": "",  # 用户偏好
        }
        # 归档记忆（外部存储，按需检索）
        self.archive = chroma.get_or_create_collection("memgpt_archive")
        self.archive_count = 0

        # 对话缓冲（短期记忆，有限大小）
        self.buffer: list[dict] = []
        self.buffer_limit = 10

        # 记忆管理工具
        self.tools = [
            {
                "name": "core_memory_update",
                "description": (
                    "更新核心记忆中的信息。核心记忆始终可见。"
                    "当用户透露重要的个人信息、偏好变化时使用。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": ["user_info", "preferences"],
                            "description": "要更新的记忆区域"
                        },
                        "content": {
                            "type": "string",
                            "description": "新的内容（会追加到现有内容）"
                        }
                    },
                    "required": ["section", "content"]
                }
            },
            {
                "name": "archive_memory_store",
                "description": (
                    "将信息存入归档记忆。归档记忆容量大但不在上下文中。"
                    "存储对话细节、过去的讨论结论等。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "要归档的内容"
                        },
                        "tags": {
                            "type": "string",
                            "description": "标签，用逗号分隔"
                        }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "archive_memory_search",
                "description": (
                    "搜索归档记忆。当需要回忆过去的对话或信息时使用。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    def _build_system_prompt(self) -> str:
        """构建包含核心记忆的 system prompt"""
        return f"""你是一个有记忆能力的AI助手。

=== 核心记忆（始终可见）===
用户信息：{self.core_memory['user_info'] or '暂无'}
用户偏好：{self.core_memory['preferences'] or '暂无'}
关于你自己：{self.core_memory['agent_info']}

=== 规则 ===
1. 当用户透露新的个人信息时，用 core_memory_update 保存
2. 当讨论产生重要结论时，用 archive_memory_store 归档
3. 当需要回忆过去的对话时，用 archive_memory_search 搜索
4. 自然地使用你对用户的了解来个性化回答"""

    def _execute_tool(self, name: str, params: dict) -> str:
        if name == "core_memory_update":
            section = params["section"]
            content = params["content"]
            if self.core_memory[section]:
                self.core_memory[section] += f"\n{content}"
            else:
                self.core_memory[section] = content
            return json.dumps({"status": "updated", "section": section})

        elif name == "archive_memory_store":
            self.archive_count += 1
            self.archive.add(
                documents=[params["content"]],
                ids=[f"archive_{self.archive_count}"],
                metadatas=[{
                    "tags": params.get("tags", ""),
                    "timestamp": datetime.now().isoformat()
                }]
            )
            return json.dumps({"status": "archived", "id": self.archive_count})

        elif name == "archive_memory_search":
            results = self.archive.query(
                query_texts=[params["query"]], n_results=3
            )
            memories = results["documents"][0] if results["documents"][0] else []
            return json.dumps({"results": memories})

        return json.dumps({"error": "unknown tool"})

    def chat(self, user_input: str) -> str:
        """对话（LLM 会自主管理记忆）"""
        self.buffer.append({"role": "user", "content": user_input})

        # 缓冲区满时裁剪
        if len(self.buffer) > self.buffer_limit:
            self.buffer = self.buffer[-self.buffer_limit:]
            while self.buffer and self.buffer[0]["role"] != "user":
                self.buffer.pop(0)

        for _ in range(5):
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=self._build_system_prompt(),
                tools=self.tools,
                messages=self.buffer
            )

            if response.stop_reason == "end_turn":
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                self.buffer.append({"role": "assistant", "content": response.content})
                return text

            if response.stop_reason == "tool_use":
                self.buffer.append({"role": "assistant", "content": response.content})
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        print(f"  [记忆操作] {block.name}: {json.dumps(block.input, ensure_ascii=False)[:60]}")
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                self.buffer.append({"role": "user", "content": results})

        return "处理超时"
```

## 时间衰减机制

人会遗忘 -- 越久远的记忆越模糊。给 Agent 加上类似的机制：

```python
"""时间衰减记忆"""
import math
from datetime import datetime, timedelta


class DecayingMemory:
    """带时间衰减的记忆系统"""

    def __init__(self, half_life_hours: float = 24.0):
        """
        half_life_hours: 记忆"半衰期"，单位小时
        24小时后重要性减半，48小时后减为1/4...
        """
        self.memories: list[dict] = []
        self.half_life = half_life_hours

    def store(self, content: str, importance: float = 1.0):
        """存入记忆"""
        self.memories.append({
            "content": content,
            "base_importance": importance,  # 初始重要性
            "created_at": datetime.now(),
            "access_count": 0,
            "last_accessed": datetime.now()
        })

    def _calculate_weight(self, memory: dict) -> float:
        """计算记忆的当前权重（考虑时间衰减）"""
        hours_passed = (datetime.now() - memory["created_at"]).total_seconds() / 3600

        # 指数衰减
        decay = math.pow(0.5, hours_passed / self.half_life)

        # 被访问次数的加成（每次访问延缓衰减）
        access_boost = 1 + 0.1 * memory["access_count"]

        return memory["base_importance"] * decay * access_boost

    def recall(self, query: str = None, top_k: int = 5) -> list[dict]:
        """检索记忆（按权重排序）"""
        weighted = []
        for mem in self.memories:
            weight = self._calculate_weight(mem)
            # 简单的关键词匹配加成
            relevance = 1.0
            if query:
                query_words = set(query.lower().split())
                mem_words = set(mem["content"].lower().split())
                overlap = len(query_words & mem_words)
                relevance = 1.0 + overlap * 0.5

            final_score = weight * relevance
            weighted.append({**mem, "current_weight": weight,
                             "final_score": final_score})

        # 按分数排序
        weighted.sort(key=lambda x: -x["final_score"])

        # 更新访问次数
        for mem in weighted[:top_k]:
            for original in self.memories:
                if original["content"] == mem["content"]:
                    original["access_count"] += 1
                    original["last_accessed"] = datetime.now()

        return weighted[:top_k]

    def forget(self, threshold: float = 0.01):
        """遗忘权重低于阈值的记忆"""
        before = len(self.memories)
        self.memories = [
            m for m in self.memories
            if self._calculate_weight(m) >= threshold
        ]
        forgotten = before - len(self.memories)
        if forgotten:
            print(f"[遗忘] 清除了 {forgotten} 条低权重记忆")


# 使用
memory = DecayingMemory(half_life_hours=24)
memory.store("用户名叫小明", importance=2.0)  # 重要信息，高权重
memory.store("今天讨论了天气", importance=0.5)  # 不重要的闲聊
memory.store("用户决定使用 FastAPI", importance=1.5)

for mem in memory.recall(top_k=3):
    print(f"  [{mem['current_weight']:.2f}] {mem['content']}")
```

## 记忆整合与冲突解决

当新记忆和旧记忆矛盾时（用户说"我改用 Rust 了"，但之前记的是"用户用 Java"），需要处理冲突：

```python
"""记忆冲突检测和解决"""
import anthropic
import json

client = anthropic.Anthropic()


class MemoryConflictResolver:
    """记忆冲突检测和解决"""

    def detect_conflict(self, new_memory: str,
                        existing_memories: list[str]) -> dict:
        """检测新记忆是否与已有记忆冲突"""
        existing_text = "\n".join([f"- {m}" for m in existing_memories])

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": f"""检测新信息是否与已有信息冲突。

已有信息：
{existing_text}

新信息：{new_memory}

返回 JSON：
{{
    "has_conflict": true/false,
    "conflicting_with": "冲突的已有信息",
    "resolution": "update"(用新的替换旧的) / "merge"(合并) / "keep_both"(都保留)
    "resolved_content": "解决后的内容"
}}"""}]
        )

        text = response.content[0].text.strip()
        try:
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            return json.loads(text)
        except json.JSONDecodeError:
            return {"has_conflict": False}

    def resolve(self, new_memory: str,
                existing_memories: list[str]) -> list[str]:
        """解决冲突并返回更新后的记忆列表"""
        result = self.detect_conflict(new_memory, existing_memories)

        if not result.get("has_conflict"):
            return existing_memories + [new_memory]

        print(f"[冲突检测] 发现冲突: {result.get('conflicting_with', '')}")
        print(f"[解决策略] {result.get('resolution', 'keep_both')}")

        resolution = result.get("resolution", "keep_both")
        resolved = result.get("resolved_content", new_memory)

        if resolution == "update":
            # 替换冲突的旧记忆
            updated = []
            for mem in existing_memories:
                if mem == result.get("conflicting_with"):
                    updated.append(resolved)
                else:
                    updated.append(mem)
            return updated

        elif resolution == "merge":
            # 合并为一条
            updated = [m for m in existing_memories
                       if m != result.get("conflicting_with")]
            updated.append(resolved)
            return updated

        else:  # keep_both
            return existing_memories + [new_memory]


# 使用
resolver = MemoryConflictResolver()
memories = ["用户主要使用 Java 编程", "用户在杭州工作"]
new_info = "用户最近转向了 Rust 开发"

updated = resolver.resolve(new_info, memories)
print("更新后的记忆:")
for m in updated:
    print(f"  - {m}")
```

## 记忆蒸馏

当长期记忆积累太多时，把大量具体记忆"蒸馏"成少数高级概括：

```python
"""记忆蒸馏：把具体记忆压缩成高级概括"""
import anthropic

client = anthropic.Anthropic()


def distill_memories(memories: list[str], max_output: int = 5) -> list[str]:
    """将大量具体记忆蒸馏为少量高级概括"""
    memory_text = "\n".join([f"- {m}" for m in memories])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": f"""将以下 {len(memories)} 条具体记忆蒸馏为 {max_output} 条高级概括。

要求：
1. 合并相似的记忆
2. 保留最重要的信息
3. 用概括性的语言，不是简单拼接
4. 标注信息的时间特征（如"长期偏好" vs "最近变化"）

具体记忆：
{memory_text}

返回 JSON 数组。"""}]
    )

    import json
    text = response.content[0].text.strip()
    try:
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return memories[:max_output]


# 使用
raw_memories = [
    "2024-03-15: 用户问了 Python 装饰器的用法",
    "2024-03-16: 用户在写一个 Flask 项目",
    "2024-03-20: 用户遇到了 CORS 跨域问题",
    "2024-04-01: 用户开始学习 FastAPI",
    "2024-04-05: 用户把 Flask 项目迁移到了 FastAPI",
    "2024-04-10: 用户问了 Pydantic 的 validator 用法",
    "2024-04-15: 用户在部署 FastAPI 到 Docker",
    "2024-04-20: 用户配置了 Nginx 反向代理",
]

distilled = distill_memories(raw_memories, max_output=3)
print("蒸馏后的记忆:")
for m in distilled:
    print(f"  - {m}")
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

```python
"""
Generative Agents 记忆架构简化实现
参考论文：Generative Agents: Interactive Simulacra of Human Behavior (2304.03442)

实现了三个核心机制：
1. 记忆流 (Memory Stream) — 按时间记录所有经历
2. 三维评分检索 — Recency × Importance × Relevance
3. 反思 (Reflection) — 从具体记忆中抽象出高层洞察
"""
import anthropic
import json
import math
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field

client = anthropic.Anthropic()


# ─── 数据结构 ───────────────────────────────────────────

@dataclass
class MemoryRecord:
    """记忆流中的单条记忆"""
    description: str                          # 自然语言描述
    created_at: datetime                      # 创建时间
    importance: float = 5.0                   # 重要性评分 (1-10)
    last_accessed: datetime | None = None     # 最近访问时间
    embedding: np.ndarray | None = None       # 语义向量（简化：用 LLM 模拟）
    memory_type: str = "observation"          # observation / reflection / plan

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at


# ─── 记忆流核心类 ──────────────────────────────────────

class MemoryStream:
    """
    Generative Agents 的记忆流实现。

    核心能力：
    - 存储所有经历（observations）
    - 三维评分检索（recency × importance × relevance）
    - 触发反思（reflection）生成高层洞察
    """

    def __init__(
        self,
        recency_decay: float = 0.995,       # 时效性衰减因子（每小时）
        reflection_threshold: float = 50.0,  # 触发反思的重要性累积阈值
        alpha: float = 1.0,                  # 时效性权重
        beta: float = 1.0,                   # 重要性权重
        gamma: float = 1.0,                  # 相关性权重
    ):
        self.memories: list[MemoryRecord] = []
        self.recency_decay = recency_decay
        self.reflection_threshold = reflection_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # 自上次反思以来的重要性累积值
        self._importance_accumulator: float = 0.0

    # ─── 重要性评分（由 LLM 打分）───────────────────

    def _rate_importance(self, description: str) -> float:
        """让 LLM 对一条记忆的重要性评分 (1-10)"""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": (
                    f"请对以下事件的重要性评分（1分=吃早饭等日常琐事，"
                    f"10分=求婚、重大人生转折）。只返回一个整数。\n\n"
                    f"事件：{description}"
                ),
            }],
        )
        text = response.content[0].text.strip()
        try:
            score = float(text)
            return max(1.0, min(10.0, score))
        except ValueError:
            return 5.0  # 解析失败时给默认中等评分

    # ─── 存入记忆 ──────────────────────────────────

    def add_observation(self, description: str, timestamp: datetime | None = None) -> MemoryRecord:
        """
        向记忆流中添加一条观察记忆。
        添加后检查是否需要触发反思。
        """
        ts = timestamp or datetime.now()
        importance = self._rate_importance(description)

        record = MemoryRecord(
            description=description,
            created_at=ts,
            importance=importance,
            memory_type="observation",
        )
        self.memories.append(record)
        self._importance_accumulator += importance

        print(f"  [记忆+] (重要性={importance:.0f}) {description}")

        # 累积重要性超过阈值时触发反思
        if self._importance_accumulator >= self.reflection_threshold:
            self._reflect()

        return record

    # ─── 三维评分检索 ──────────────────────────────

    def retrieve(self, query: str, top_k: int = 5, now: datetime | None = None) -> list[dict]:
        """
        三维评分检索：
        score = alpha * recency + beta * importance + gamma * relevance

        各维度先归一化到 [0, 1]，再加权求和。
        """
        if not self.memories:
            return []

        now = now or datetime.now()

        # --- 1. 计算各维度原始分 ---
        raw_recency = []
        raw_importance = []
        raw_relevance = []

        for mem in self.memories:
            # 时效性：指数衰减
            hours_passed = (now - mem.created_at).total_seconds() / 3600
            recency = math.pow(self.recency_decay, hours_passed)
            raw_recency.append(recency)

            # 重要性：直接用评分
            raw_importance.append(mem.importance)

            # 相关性：简化实现 — 用关键词重叠度模拟语义相似度
            # 生产环境应使用 embedding 余弦相似度
            relevance = self._compute_relevance(query, mem.description)
            raw_relevance.append(relevance)

        # --- 2. Min-Max 归一化到 [0, 1] ---
        def normalize(values: list[float]) -> list[float]:
            min_v, max_v = min(values), max(values)
            if max_v - min_v < 1e-9:
                return [1.0] * len(values)
            return [(v - min_v) / (max_v - min_v) for v in values]

        norm_recency = normalize(raw_recency)
        norm_importance = normalize(raw_importance)
        norm_relevance = normalize(raw_relevance)

        # --- 3. 加权求和 ---
        scored = []
        for i, mem in enumerate(self.memories):
            score = (
                self.alpha * norm_recency[i]
                + self.beta * norm_importance[i]
                + self.gamma * norm_relevance[i]
            )
            scored.append({
                "memory": mem,
                "score": score,
                "recency": norm_recency[i],
                "importance": norm_importance[i],
                "relevance": norm_relevance[i],
            })

        # 按总分降序排序
        scored.sort(key=lambda x: -x["score"])

        # 更新被检索记忆的 last_accessed
        for item in scored[:top_k]:
            item["memory"].last_accessed = now

        return scored[:top_k]

    def _compute_relevance(self, query: str, description: str) -> float:
        """
        计算查询与记忆的相关性（简化版：关键词重叠）。

        生产环境建议替换为：
          query_emb = get_embedding(query)
          mem_emb = get_embedding(description)
          return cosine_similarity(query_emb, mem_emb)
        """
        query_tokens = set(query.lower().replace("，", " ").replace("。", " ").split())
        mem_tokens = set(description.lower().replace("，", " ").replace("。", " ").split())
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & mem_tokens)
        return overlap / len(query_tokens)

    # ─── 反思机制 ──────────────────────────────────

    def _reflect(self):
        """
        反思：从近期记忆中提炼高层次洞察。

        流程：
        1. 收集最近的记忆（按重要性排序取 top-20）
        2. 让 LLM 从中抽象出 2-3 条高层洞察
        3. 将洞察作为 reflection 类型存入记忆流
        """
        print("\n  === 触发反思 ===")

        # 取最近的记忆，按重要性排序
        recent = sorted(self.memories, key=lambda m: -m.importance)[:20]
        memory_text = "\n".join(
            f"  [{m.created_at.strftime('%H:%M')}] (重要性={m.importance:.0f}) {m.description}"
            for m in recent
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""基于以下记忆，提炼出 2-3 条高层次洞察。
洞察应该是对具体事件的抽象概括，揭示模式、关系或性格特征。

记忆列表：
{memory_text}

以 JSON 数组返回，每条洞察是一个字符串。示例：
["Maria 对艺术充满热情，尤其是绘画", "John 和 Maria 正在发展友谊"]""",
            }],
        )

        text = response.content[0].text.strip()
        try:
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            insights = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            insights = [text]

        # 将反思结果存入记忆流（类型标记为 reflection）
        for insight in insights:
            record = MemoryRecord(
                description=f"[反思] {insight}",
                created_at=datetime.now(),
                importance=8.0,  # 反思结果通常比具体记忆更重要
                memory_type="reflection",
            )
            self.memories.append(record)
            print(f"  [反思结果] {insight}")

        # 重置累积器
        self._importance_accumulator = 0.0
        print("  === 反思完成 ===\n")


# ─── 使用示例 ──────────────────────────────────────────

if __name__ == "__main__":
    stream = MemoryStream(reflection_threshold=30.0)

    # 模拟一个 Agent 一天的经历（手动指定时间戳以演示时效性）
    base_time = datetime(2026, 4, 16, 8, 0, 0)

    observations = [
        (0,  "早上在公园散步，遇到了邻居 Maria"),
        (1,  "Maria 提到她正在准备下周的画展"),
        (2,  "去咖啡馆工作，写了两个小时的报告"),
        (3,  "午饭在食堂吃了意面"),
        (4,  "下午和同事 John 讨论了新项目方案，他对 AI 方向很感兴趣"),
        (5,  "John 邀请我周六去他家的烧烤派对"),
        (6,  "收到邮件，下个月公司要重组部门"),
        (8,  "晚上去了 Maria 的画室，看到她最新的油画作品非常出色"),
        (9,  "Maria 说她考虑辞职全职画画"),
        (10, "回家路上想到 John 和 Maria 都对创意工作感兴趣，也许可以介绍他们认识"),
    ]

    # 依次添加观察记忆
    for hour_offset, desc in observations:
        ts = base_time + timedelta(hours=hour_offset)
        stream.add_observation(desc, timestamp=ts)

    # --- 三维评分检索演示 ---
    print("\n" + "=" * 60)
    print("检索: 'Maria 的兴趣爱好'\n")

    results = stream.retrieve(
        "Maria 的兴趣爱好",
        top_k=5,
        now=base_time + timedelta(hours=12),
    )

    for i, item in enumerate(results, 1):
        mem = item["memory"]
        print(
            f"  #{i} [总分={item['score']:.2f}] "
            f"(时效={item['recency']:.2f} "
            f"重要={item['importance']:.2f} "
            f"相关={item['relevance']:.2f}) "
            f"{'[反思]' if mem.memory_type == 'reflection' else ''}"
            f"{mem.description}"
        )
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

```python
"""自主记忆管理 Agent"""
import anthropic
import json
from datetime import datetime

client = anthropic.Anthropic()


class AutonomousMemoryAgent:
    """自主记忆管理 Agent -- 综合 MemGPT + 时间衰减 + 冲突解决"""

    def __init__(self):
        self.core = {"user_profile": "", "preferences": "", "context": ""}
        self.working_memory: list[dict] = []
        self.long_term: list[dict] = []
        self.max_working = 15

        # Agent 有记忆管理工具
        self.tools = [
            {
                "name": "remember",
                "description": "将重要信息存入长期记忆。用户偏好、决定、关键事实应该被记住。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "要记住的内容"},
                        "importance": {"type": "number", "description": "重要性 1-5"}
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "recall",
                "description": "从长期记忆中回忆信息。需要引用过去的对话或事实时使用。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "想回忆什么"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "update_profile",
                "description": "更新对用户的认知。当了解到用户新的信息或偏好变化时使用。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": ["user_profile", "preferences"]},
                        "content": {"type": "string"}
                    },
                    "required": ["field", "content"]
                }
            }
        ]

    def _execute_tool(self, name: str, params: dict) -> str:
        if name == "remember":
            self.long_term.append({
                "content": params["content"],
                "importance": params.get("importance", 3),
                "timestamp": datetime.now().isoformat()
            })
            return json.dumps({"status": "remembered"})

        elif name == "recall":
            query = params["query"].lower()
            relevant = [
                m for m in self.long_term
                if any(w in m["content"].lower() for w in query.split())
            ]
            relevant.sort(key=lambda x: -x.get("importance", 3))
            return json.dumps({
                "memories": [m["content"] for m in relevant[:5]]
            }, ensure_ascii=False)

        elif name == "update_profile":
            self.core[params["field"]] = params["content"]
            return json.dumps({"status": "profile updated"})

        return json.dumps({"error": "unknown"})

    def _build_system(self) -> str:
        profile = self.core["user_profile"] or "暂无"
        prefs = self.core["preferences"] or "暂无"
        return f"""你是一个有记忆能力的AI助手。

=== 你对用户的了解 ===
用户画像：{profile}
用户偏好：{prefs}

=== 记忆管理规则 ===
1. 用户透露个人信息时 -> update_profile
2. 讨论产生重要结论时 -> remember
3. 需要引用过去的事时 -> recall
4. 自然地使用你的记忆，不要每次都刻意提及"""

    def chat(self, user_input: str) -> str:
        self.working_memory.append({"role": "user", "content": user_input})
        if len(self.working_memory) > self.max_working:
            self.working_memory = self.working_memory[-self.max_working:]

        for _ in range(5):
            response = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=1024,
                system=self._build_system(),
                tools=self.tools,
                messages=self.working_memory
            )

            if response.stop_reason == "end_turn":
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                self.working_memory.append({"role": "assistant", "content": response.content})
                return text

            if response.stop_reason == "tool_use":
                self.working_memory.append({"role": "assistant", "content": response.content})
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                self.working_memory.append({"role": "user", "content": results})

        return "处理超时"


if __name__ == "__main__":
    agent = AutonomousMemoryAgent()
    print("自主记忆 Agent（输入 quit 退出）\n")
    while True:
        user = input("你: ").strip()
        if user == "quit":
            break
        print(f"助手: {agent.chat(user)}\n")
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
