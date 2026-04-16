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

```python
import anthropic

client = anthropic.Anthropic()

class SummaryMemory:
    """摘要记忆：压缩早期对话"""

    def __init__(self, max_messages: int = 10, summary_threshold: int = 8):
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.messages: list[dict] = []
        self.summary: str = ""

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.summary_threshold:
            self._compress()

    def _compress(self):
        """将前半部分消息压缩为摘要"""
        n_to_compress = len(self.messages) // 2
        to_compress = self.messages[:n_to_compress]
        self.messages = self.messages[n_to_compress:]

        conversation = "\n".join([
            f"{m['role']}: {m['content']}" for m in to_compress
        ])

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""请将以下对话历史压缩为简洁的摘要。
保留关键信息：用户身份、核心需求、已达成的共识、重要决策。

{f"之前的摘要：{self.summary}" if self.summary else ""}

新对话：
{conversation}

请输出更新后的摘要："""
            }]
        )
        self.summary = response.content[0].text
        print(f"[摘要已更新] {self.summary[:100]}...")

    def get_messages(self) -> list[dict]:
        """返回包含摘要的完整上下文"""
        result = []
        if self.summary:
            result.append({
                "role": "user",
                "content": f"[对话历史摘要] {self.summary}"
            })
            result.append({
                "role": "assistant",
                "content": "好的，我已了解之前的对话背景。请继续。"
            })
        result.extend(self.messages)
        return result
```

摘要的好处很明显：10 轮对话可能需要 2000 tokens，压缩成摘要可能只需要 200 tokens，节省 90%。而且用户的名字、核心需求等关键信息都保留了。

::: warning 摘要的代价
每次压缩需要一次额外的 LLM API 调用。如果对话不长（< 10 轮），摘要压缩的收益可能不如成本。建议只在对话确实较长时才启用。
:::

## Buffer + Summary 混合策略

实际中最好用的方案：**最近的消息完整保留（Buffer），更早的消息压缩成摘要（Summary）**。

```python
class BufferSummaryMemory:
    """Buffer + Summary 混合记忆"""

    def __init__(self, buffer_size: int = 6, max_tokens: int = 2000):
        self.buffer_size = buffer_size
        self.max_tokens = max_tokens
        self.buffer: list[dict] = []    # 最近的消息（完整保留）
        self.summary: str = ""           # 早期消息的摘要
        self.total_messages: int = 0

    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})
        self.total_messages += 1

        if len(self.buffer) > self.buffer_size:
            overflow = self.buffer[:-self.buffer_size]
            self.buffer = self.buffer[-self.buffer_size:]
            self._update_summary(overflow)

    def _update_summary(self, messages: list[dict]):
        """将溢出的消息合并到摘要中"""
        conversation = "\n".join([
            f"{m['role']}: {m['content']}" for m in messages
        ])

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""将以下新对话内容整合到现有摘要中。

现有摘要：{self.summary or "（空）"}

新内容：
{conversation}

输出更新后的摘要（保持简洁，只保留重要信息）："""
            }]
        )
        self.summary = response.content[0].text

    def get_messages(self) -> list[dict]:
        result = []
        if self.summary:
            result.append({
                "role": "user",
                "content": f"[对话摘要，共 {self.total_messages - len(self.buffer)} 条消息] {self.summary}"
            })
            result.append({
                "role": "assistant",
                "content": "好的，我了解了之前的对话背景。"
            })
        result.extend(self.buffer)
        return result

    def stats(self) -> dict:
        return {
            "total_messages": self.total_messages,
            "buffer_messages": len(self.buffer),
            "has_summary": bool(self.summary),
        }
```

使用示例：

```python
memory = BufferSummaryMemory(buffer_size=6)

conversations = [
    ("user", "你好，我是产品经理小王"),
    ("assistant", "你好小王！有什么可以帮你的？"),
    ("user", "我们在做一个电商推荐系统"),
    ("assistant", "好的，请告诉我具体需求"),
    ("user", "需要基于用户行为数据做个性化推荐"),
    ("assistant", "明白，这需要协同过滤或深度学习方法"),
    ("user", "预算有限，不想用太复杂的模型"),
    ("assistant", "那可以考虑基于物品的协同过滤"),
    # 到这里早期消息开始被压缩
    ("user", "我们目前有 10 万条数据"),
    ("assistant", "10 万条足够了，可以开始搭建"),
]

for role, content in conversations:
    memory.add(role, content)

print(f"统计: {memory.stats()}")
# 早期消息已压缩为摘要，"小王"、"电商推荐"、"协同过滤"等关键信息都保留了
```

## 长期记忆：向量数据库存储

短期记忆只在单次对话中有效。用户关掉对话再回来，一切归零。长期记忆让 Agent 能够**跨会话**记住重要信息。

最常用的方案是向量数据库——把记忆编码为向量存储，检索时用语义匹配。

```python
import chromadb
from datetime import datetime
import json

class VectorLongTermMemory:
    """基于向量数据库的长期记忆"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.client = chromadb.PersistentClient(path=f"./memory_db/{user_id}")
        self.collection = self.client.get_or_create_collection(
            name="long_term_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self._counter = self.collection.count()

    def store(self, content: str, memory_type: str = "general",
              importance: float = 0.5):
        """存储一条记忆"""
        self._counter += 1
        self.collection.add(
            documents=[content],
            ids=[f"mem_{self._counter}"],
            metadatas=[{
                "type": memory_type,
                "importance": importance,
                "timestamp": datetime.now().isoformat(),
                "user_id": self.user_id,
            }]
        )

    def recall(self, query: str, top_k: int = 5,
               memory_type: str = None) -> list[dict]:
        """检索相关记忆"""
        where = {"user_id": self.user_id}
        if memory_type:
            where["type"] = memory_type

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )

        memories = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            memories.append({
                "content": doc,
                "type": meta["type"],
                "importance": meta["importance"],
                "timestamp": meta["timestamp"],
                "relevance": 1 - dist,
            })
        return memories

    def count(self) -> int:
        return self.collection.count()
```

## 结构化记忆：KV 存储

对于确定性信息（用户名、偏好设置），用键值存储比向量数据库更合适——检索精确，不需要语义匹配。

```python
import os

class KVMemory:
    """键值对长期记忆"""

    def __init__(self, user_id: str, storage_path: str = "./kv_memory"):
        self.user_id = user_id
        self.path = os.path.join(storage_path, f"{user_id}.json")
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def set(self, key: str, value):
        self.data[key] = {
            "value": value,
            "updated_at": datetime.now().isoformat(),
        }
        self._save()

    def get(self, key: str, default=None):
        entry = self.data.get(key)
        return entry["value"] if entry else default

    def get_all(self) -> dict:
        return {k: v["value"] for k, v in self.data.items()}
```

## 用户画像渐进构建

Agent 可以在每次对话中提取用户的新信息，逐步构建用户画像。

```python
class UserProfile:
    """渐进式用户画像"""

    def __init__(self, user_id: str):
        self.kv = KVMemory(user_id, "./user_profiles")
        self.profile = self.kv.get("profile", {
            "name": None,
            "occupation": None,
            "interests": [],
            "preferences": {},
            "interaction_count": 0,
        })

    def update_from_conversation(self, conversation: list[dict]):
        """从对话中更新用户画像"""
        conv_text = "\n".join([
            f"{m['role']}: {m['content']}" for m in conversation
        ])

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""基于以下对话，提取用户的个人信息来更新画像。

当前画像：{json.dumps(self.profile, ensure_ascii=False)}

新对话：
{conv_text}

只返回需要更新的字段（JSON），没有新信息则返回 {{}}。
可更新的字段：name, occupation, interests(数组), preferences(对象)"""
            }]
        )

        updates = json.loads(response.content[0].text)
        for key, value in updates.items():
            if key == "interests" and isinstance(value, list):
                existing = self.profile.get("interests", [])
                self.profile["interests"] = list(set(existing + value))
            elif key == "preferences" and isinstance(value, dict):
                self.profile.setdefault("preferences", {}).update(value)
            else:
                self.profile[key] = value

        self.profile["interaction_count"] = self.profile.get("interaction_count", 0) + 1
        self.kv.set("profile", self.profile)

    def get_summary(self) -> str:
        """生成用户画像摘要"""
        p = self.profile
        parts = []
        if p.get("name"): parts.append(f"用户名: {p['name']}")
        if p.get("occupation"): parts.append(f"职业: {p['occupation']}")
        if p.get("interests"): parts.append(f"兴趣: {', '.join(p['interests'])}")
        if p.get("preferences"):
            prefs = "; ".join(f"{k}={v}" for k, v in p["preferences"].items())
            parts.append(f"偏好: {prefs}")
        parts.append(f"交互次数: {p.get('interaction_count', 0)}")
        return " | ".join(parts) if parts else "新用户，暂无画像"
```

## 重要信息提取器

不是所有对话内容都值得存入长期记忆。需要一个"重要性筛选器"。

```python
class MemoryExtractor:
    """从对话中提取值得记忆的信息"""

    def extract(self, conversation: list[dict]) -> list[dict]:
        conv_text = "\n".join([
            f"{m['role']}: {m['content']}" for m in conversation
        ])

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""从以下对话中提取值得长期记忆的信息。

对话：
{conv_text}

识别以下类型：
1. 用户个人信息（名字、职业、偏好）
2. 重要决策和结论
3. 用户的具体需求和目标
4. 有价值的反馈

返回 JSON 数组：
[{{"content": "记忆内容", "type": "personal/decision/need/feedback", "importance": 0.1-1.0}}]

没有值得记忆的信息就返回 []。"""
            }]
        )
        return json.loads(response.content[0].text)
```

## 多层记忆系统

把短期记忆和长期记忆组合成一个完整的系统：

```python
class MultiLayerMemory:
    """多层记忆系统：短期 + 长期"""

    def __init__(self, user_id: str):
        self.short_term = BufferSummaryMemory(buffer_size=8)
        self.long_term = VectorLongTermMemory(user_id)
        self.profile = UserProfile(user_id)
        self.extractor = MemoryExtractor()

    def before_response(self, user_input: str) -> dict:
        """在生成回复前，收集所有相关记忆"""
        # 从长期记忆中检索相关信息
        relevant_memories = self.long_term.recall(user_input, top_k=3)
        memory_context = "\n".join([m["content"] for m in relevant_memories])

        return {
            "messages": self.short_term.get_messages(),
            "user_profile": self.profile.get_summary(),
            "relevant_memories": memory_context,
        }

    def after_response(self, user_input: str, assistant_reply: str):
        """在生成回复后，更新所有记忆层"""
        # 更新短期记忆
        self.short_term.add("user", user_input)
        self.short_term.add("assistant", assistant_reply)

        # 提取重要信息存入长期记忆
        recent = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_reply},
        ]
        memories = self.extractor.extract(recent)
        for mem in memories:
            self.long_term.store(
                content=mem["content"],
                memory_type=mem["type"],
                importance=mem["importance"],
            )

        # 更新用户画像
        self.profile.update_from_conversation(recent)
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
