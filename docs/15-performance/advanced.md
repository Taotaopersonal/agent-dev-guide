# 性能优化 · 高级篇

::: tip 学习目标
- 理解语义缓存的原理，用向量相似度匹配复用 LLM 响应
- 设计多级缓存架构（内存 -> Redis -> 语义缓存）
- 构建智能模型路由器，综合成本、质量、速度做最优选择
- 实现 Fallback 降级链和预测性执行
:::

::: info 学完你能做到
- 让语义相近的问题直接命中缓存，避免重复调用 LLM
- 设计三级缓存架构，逐级降速但容量递增
- 构建一个多因子评分的模型路由器，实时适应模型状态变化
- 在 LLM 推理期间预取工具结果，减少等待时间
:::

## 语义缓存：相似问题不重复调用

进阶篇讲了 Prompt Caching（API 层面）和工具结果缓存（精确匹配）。但有一个更大的优化空间：**用户问的问题语义相近但措辞不同时，能不能复用之前的回答？**

"北京今天天气怎么样" 和 "今天北京天气如何" 含义一样，但精确匹配会当作两个不同的查询。语义缓存通过向量相似度匹配来解决这个问题。

### 语义缓存实现

```python
"""semantic_cache.py — 基于嵌入向量的语义缓存"""

import time
import numpy as np
from dataclasses import dataclass

@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: np.ndarray
    created_at: float
    hit_count: int = 0
    ttl: float = 3600.0  # 默认 1 小时过期

class SemanticCache:
    """语义缓存——用向量相似度匹配"""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_entries: int = 1000,
    ):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.entries: list[CacheEntry] = []
        self._stats = {"hits": 0, "misses": 0}

    async def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量

        生产环境使用 Voyage AI 或 OpenAI Embeddings API：
            response = await voyage_client.embed([text], model="voyage-3")
            return np.array(response.embeddings[0])

        这里用简化方式演示逻辑：
        """
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        return np.frombuffer(hash_bytes, dtype=np.float32)[:8]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def get(self, query: str) -> str | None:
        """查找语义相似的缓存条目"""
        query_embedding = await self.get_embedding(query)
        now = time.time()

        best_match: CacheEntry | None = None
        best_similarity = 0.0

        for entry in self.entries:
            # 跳过过期条目
            if now - entry.created_at > entry.ttl:
                continue
            similarity = self._cosine_similarity(
                query_embedding, entry.embedding
            )
            if similarity > self.threshold and similarity > best_similarity:
                best_match = entry
                best_similarity = similarity

        if best_match:
            best_match.hit_count += 1
            self._stats["hits"] += 1
            return best_match.response

        self._stats["misses"] += 1
        return None

    async def set(self, query: str, response: str, ttl: float = 3600.0):
        """添加缓存条目"""
        embedding = await self.get_embedding(query)

        # 容量控制：淘汰最旧的 25% 条目
        if len(self.entries) >= self.max_entries:
            self.entries.sort(key=lambda e: e.created_at)
            self.entries = self.entries[len(self.entries) // 4:]

        self.entries.append(CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            created_at=time.time(),
            ttl=ttl,
        ))

    @property
    def hit_rate(self) -> float:
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0
```

### 多级缓存架构

单一缓存层往往不够。更好的设计是三级缓存，逐级降速但容量递增：

```python
"""multi_level_cache.py — L1(内存) -> L2(Redis) -> L3(语义缓存)"""

import time
import redis.asyncio as redis

class MultiLevelCache:
    """三级缓存架构"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.l1: dict[str, tuple[str, float]] = {}  # 内存（最快）
        self.l2 = redis.from_url(redis_url)          # Redis（分布式）
        self.l3 = SemanticCache(similarity_threshold=0.95)  # 语义（最智能）

    async def get(self, query: str, exact_key: str | None = None) -> str | None:
        """逐级查找，命中后回填上层"""

        # L1: 精确匹配（内存，微秒级）
        if exact_key and exact_key in self.l1:
            result, ts = self.l1[exact_key]
            if time.time() - ts < 300:  # 5 分钟 TTL
                return result

        # L2: 精确匹配（Redis，毫秒级）
        if exact_key:
            cached = await self.l2.get(f"agent:cache:{exact_key}")
            if cached:
                result = cached.decode()
                # 回填 L1
                self.l1[exact_key] = (result, time.time())
                return result

        # L3: 语义匹配（向量计算，十毫秒级）
        semantic_result = await self.l3.get(query)
        if semantic_result:
            return semantic_result

        return None

    async def set(self, query: str, response: str, exact_key: str | None = None):
        """写入所有层级"""
        if exact_key:
            self.l1[exact_key] = (response, time.time())
            await self.l2.setex(f"agent:cache:{exact_key}", 3600, response)
        await self.l3.set(query, response)
```

### 缓存失效策略

缓存最难的部分不是写入，而是失效——数据变了，旧缓存就不对了：

```python
"""cache_invalidation.py — 缓存失效策略"""

class CacheInvalidator:
    """当底层数据变更时，主动清除相关缓存"""

    def __init__(self, cache: MultiLevelCache):
        self.cache = cache

    async def invalidate_by_tool(self, tool_name: str):
        """工具执行写操作时，清除相关的读缓存"""
        # 写操作 -> 需要清除的读缓存
        write_to_read_map = {
            "insert_record": ["query", "list_tables"],
            "update_record": ["query"],
            "delete_record": ["query", "list_tables"],
        }
        related_tools = write_to_read_map.get(tool_name, [])
        for related in related_tools:
            keys_to_remove = [
                k for k in self.cache.l1 if k.startswith(related)
            ]
            for k in keys_to_remove:
                del self.cache.l1[k]

    async def invalidate_by_pattern(self, pattern: str):
        """按模式批量清除 Redis 缓存"""
        keys = []
        async for key in self.cache.l2.scan_iter(f"agent:cache:{pattern}*"):
            keys.append(key)
        if keys:
            await self.cache.l2.delete(*keys)
```

## 智能模型路由器

进阶篇用信号量做并发控制，但模型选择还是固定的。高级路由器综合**成本、质量、速度、可用性**四个维度做实时决策。

### 模型画像 + 多因子评分

```python
"""smart_router.py — 智能模型路由器"""

import time
from dataclasses import dataclass
from enum import Enum

class Complexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

@dataclass
class ModelProfile:
    """模型画像：静态属性 + 动态状态"""
    name: str
    provider: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: float         # 实时更新
    capability_score: float       # 0-1 能力评分
    is_available: bool = True
    error_rate: float = 0.0       # 近期错误率（实时更新）

@dataclass
class RoutingDecision:
    model: str
    provider: str
    reason: str
    estimated_cost: float
    estimated_latency_ms: float

class SmartModelRouter:
    """智能模型路由器——综合多因子做最优选择"""

    def __init__(self):
        self.models = {
            "claude-opus-4-20250514": ModelProfile(
                name="claude-opus-4-20250514", provider="anthropic",
                cost_per_1k_input=0.015, cost_per_1k_output=0.075,
                avg_latency_ms=3000, capability_score=1.0,
            ),
            "claude-sonnet-4-20250514": ModelProfile(
                name="claude-sonnet-4-20250514", provider="anthropic",
                cost_per_1k_input=0.003, cost_per_1k_output=0.015,
                avg_latency_ms=1500, capability_score=0.85,
            ),
            "claude-haiku-3-20250414": ModelProfile(
                name="claude-haiku-3-20250414", provider="anthropic",
                cost_per_1k_input=0.00025, cost_per_1k_output=0.00125,
                avg_latency_ms=500, capability_score=0.6,
            ),
            "gpt-4o": ModelProfile(
                name="gpt-4o", provider="openai",
                cost_per_1k_input=0.0025, cost_per_1k_output=0.01,
                avg_latency_ms=1200, capability_score=0.85,
            ),
            "gpt-4o-mini": ModelProfile(
                name="gpt-4o-mini", provider="openai",
                cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
                avg_latency_ms=400, capability_score=0.55,
            ),
        }

    def route(
        self,
        message: str,
        priority: str = "balanced",   # "cost", "quality", "speed", "balanced"
        budget_remaining: float | None = None,
    ) -> RoutingDecision:
        """智能路由决策"""
        # 1. 判断任务复杂度
        complexity = self._classify(message)
        estimated_tokens = len(message) // 2

        # 2. 过滤可用模型（排除故障和高错误率的）
        available = {
            name: m for name, m in self.models.items()
            if m.is_available and m.error_rate < 0.3
        }

        # 3. 多因子评分
        scored = []
        for name, model in available.items():
            score = self._score(
                model, complexity, priority,
                estimated_tokens, budget_remaining,
            )
            if score >= 0:
                scored.append((name, model, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        best_name, best_model, _ = scored[0]

        est_cost = (
            estimated_tokens * best_model.cost_per_1k_input / 1000
            + 500 * best_model.cost_per_1k_output / 1000
        )
        return RoutingDecision(
            model=best_name,
            provider=best_model.provider,
            reason=f"complexity={complexity.value}, priority={priority}",
            estimated_cost=round(est_cost, 6),
            estimated_latency_ms=best_model.avg_latency_ms,
        )

    def _classify(self, message: str) -> Complexity:
        """规则引擎判断复杂度（零额外成本）"""
        msg_len = len(message)
        if msg_len < 50 or any(k in message for k in ["翻译", "格式化", "分类"]):
            return Complexity.SIMPLE
        if msg_len > 500 or any(k in message for k in ["分析", "设计", "推理"]):
            return Complexity.COMPLEX
        return Complexity.MEDIUM

    def _score(
        self, model: ModelProfile, complexity: Complexity,
        priority: str, est_tokens: int, budget: float | None,
    ) -> float:
        """多因子评分"""
        # 能力门槛
        min_cap = {
            Complexity.SIMPLE: 0.4,
            Complexity.MEDIUM: 0.7,
            Complexity.COMPLEX: 0.9,
        }[complexity]
        if model.capability_score < min_cap:
            return -1  # 能力不足，排除

        # 预算检查
        est_cost = est_tokens * model.cost_per_1k_input / 1000
        if budget is not None and est_cost > budget:
            return -1  # 超预算

        # 权重矩阵
        weights = {
            "cost":     {"cost": 0.6, "quality": 0.2, "speed": 0.2},
            "quality":  {"cost": 0.1, "quality": 0.7, "speed": 0.2},
            "speed":    {"cost": 0.1, "quality": 0.2, "speed": 0.7},
            "balanced": {"cost": 0.33, "quality": 0.34, "speed": 0.33},
        }[priority]

        # 归一化评分（0-1）
        cost_score = 1.0 - min(model.cost_per_1k_input / 0.015, 1.0)
        quality_score = model.capability_score
        speed_score = 1.0 - min(model.avg_latency_ms / 5000, 1.0)

        return (
            weights["cost"] * cost_score
            + weights["quality"] * quality_score
            + weights["speed"] * speed_score
        )

    def update_model_stats(self, model: str, latency_ms: int, success: bool):
        """每次调用完成后更新模型实时状态"""
        if model in self.models:
            m = self.models[model]
            # 滑动平均更新延迟
            m.avg_latency_ms = m.avg_latency_ms * 0.9 + latency_ms * 0.1
            # 更新错误率
            if not success:
                m.error_rate = min(m.error_rate + 0.1, 1.0)
            else:
                m.error_rate = max(m.error_rate - 0.01, 0.0)


# 使用示例
router = SmartModelRouter()

decision = router.route("帮我翻译这段话", priority="cost")
print(f"选择: {decision.model}, 原因: {decision.reason}")

decision = router.route(
    "请设计一个分布式消息队列系统，要求支持百万级并发...",
    priority="quality",
)
print(f"选择: {decision.model}, 原因: {decision.reason}")
```

## Fallback 降级链

当主模型不可用时（超时、限流、宕机），自动降级到备选模型，而不是直接报错：

```python
"""fallback_chain.py — 跨提供商 Fallback 降级链"""

import asyncio
import anthropic
import openai

class ModelWithFallback:
    """带降级的模型调用——主模型失败自动切换"""

    FALLBACK_CHAIN = [
        {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
        {"provider": "openai",    "model": "gpt-4o"},
        {"provider": "anthropic", "model": "claude-haiku-3-20250414"},
        {"provider": "openai",    "model": "gpt-4o-mini"},
    ]

    def __init__(self):
        self.anthropic_client = anthropic.AsyncAnthropic()
        self.openai_client = openai.AsyncOpenAI()

    async def call(self, messages: list[dict], max_fallbacks: int = 3) -> dict:
        """逐级尝试，直到成功"""
        errors = []
        for i, option in enumerate(self.FALLBACK_CHAIN[:max_fallbacks + 1]):
            try:
                result = await asyncio.wait_for(
                    self._call_provider(
                        option["provider"], option["model"], messages
                    ),
                    timeout=30.0,
                )
                if i > 0:
                    result["degraded"] = True
                    result["original_model"] = self.FALLBACK_CHAIN[0]["model"]
                return result
            except Exception as e:
                errors.append(f"{option['model']}: {e}")
                continue

        raise Exception(f"所有模型均不可用: {errors}")

    async def _call_provider(
        self, provider: str, model: str, messages: list[dict]
    ) -> dict:
        if provider == "anthropic":
            response = await self.anthropic_client.messages.create(
                model=model, max_tokens=4096, messages=messages,
            )
            return {
                "text": response.content[0].text,
                "model": model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }
        elif provider == "openai":
            response = await self.openai_client.chat.completions.create(
                model=model, messages=messages, max_tokens=4096,
            )
            return {
                "text": response.choices[0].message.content,
                "model": model,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
            }
```

## 预测性执行

在 LLM 推理期间，如果能预判它可能需要的工具结果并提前获取，就可以节省一轮等待时间：

```python
"""predictive_execution.py — 预测性工具预取"""

import asyncio

async def execute_tool(tool_name: str, tool_input: dict) -> str:
    """执行工具调用（需根据实际工具注册表实现）"""
    raise NotImplementedError(f"请实现工具 {tool_name} 的执行逻辑")

class PredictiveExecutor:
    """在 LLM 推理的同时，预取可能需要的工具结果"""

    def __init__(self):
        self.prefetch_cache: dict[str, asyncio.Task] = {}

    async def prefetch(self, likely_tools: list[dict]):
        """提前发起可能需要的工具调用"""
        for tool in likely_tools:
            cache_key = f"{tool['name']}:{hash(str(tool['input']))}"
            if cache_key not in self.prefetch_cache:
                self.prefetch_cache[cache_key] = asyncio.create_task(
                    execute_tool(tool["name"], tool["input"])
                )

    async def get_or_execute(self, tool_name: str, tool_input: dict) -> str:
        """优先从预取缓存获取，否则正常执行"""
        cache_key = f"{tool_name}:{hash(str(tool_input))}"
        if cache_key in self.prefetch_cache:
            task = self.prefetch_cache.pop(cache_key)
            return await task
        return await execute_tool(tool_name, tool_input)

    def clear(self):
        """清理未使用的预取任务"""
        for task in self.prefetch_cache.values():
            task.cancel()
        self.prefetch_cache.clear()


# 使用场景：
# 用户问"今天北京天气怎么样"
# -> 在 LLM 推理时，预测它大概率会调用 get_weather(city="北京")
# -> 提前发起天气查询
# -> LLM 返回 tool_use 时，结果已经准备好了，省去一轮等待
```

::: warning 预测性执行的代价
预取是有成本的——如果预测错了，工具调用白做了。适合以下场景：
- 工具调用模式高度可预测（如 "天气" -> get_weather）
- 工具调用耗时较长但成本低（如数据库查询）
- 错误的预取不会产生副作用（只读操作）

**不适合**：写操作、高成本 API 调用、不确定的场景。
:::

## 小结

高级性能优化的四大策略：

1. **语义缓存**：用向量相似度匹配复用 LLM 响应，语义相近的问题不重复调用
2. **多级缓存**：L1 内存（微秒）-> L2 Redis（毫秒）-> L3 语义（十毫秒），逐级降速但容量递增
3. **智能路由**：综合成本、质量、速度、可用性四个维度实时评分，每次选最优模型
4. **预测性执行**：在 LLM 推理期间预取工具结果，减少一轮等待时间

## 练习

1. 实现 `SemanticCache`，用真实的嵌入 API（如 Voyage 或 OpenAI Embeddings），测试相似问题的命中率
2. 构建 `SmartModelRouter`，准备不同复杂度的 10 个问题，分别以 "cost"、"quality"、"speed" 优先级路由，验证模型选择是否合理
3. 实现 `ModelWithFallback`，模拟主模型超时的场景，验证降级链是否正常工作

## 参考资源

- [GPTCache](https://github.com/zilliztech/GPTCache) -- 开源语义缓存库
- [arXiv:2311.04934 - Semantic Caching for LLM Applications](https://arxiv.org/abs/2311.04934) -- 语义缓存研究
- [arXiv:2404.14219 - RouterBench](https://arxiv.org/abs/2404.14219) -- 模型路由基准测试
- [arXiv:2402.07625 - RouteLLM](https://arxiv.org/abs/2402.07625) -- 基于偏好的路由
- [OpenRouter](https://openrouter.ai/) -- 多模型统一 API 网关
- [LiteLLM](https://github.com/BerriAI/litellm) -- 开源多模型代理工具
- [Redis 官方文档](https://redis.io/docs/) -- 缓存数据库
