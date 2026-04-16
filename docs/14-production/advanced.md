# 生产工程化 · 高级篇

::: tip 学习目标
- 理解 LLM 调用的成本构成，掌握成本计算方法
- 实现 Prompt Caching 节省 80%+ 的输入成本
- 设计成本感知的模型路由器和预算控制系统
- 构建完整的 CI/CD 流水线和灰度发布机制
:::

::: info 学完你能做到
- 精确计算每次 Agent 任务的成本并设置预算上限
- 用 Prompt Caching 大幅降低高频 Agent 的运行成本
- 让简单任务自动走便宜模型，复杂任务走强模型
- 搭建从测试到部署的自动化流水线，支持灰度发布
:::

## 成本控制：Agent 的隐形杀手

普通 API 调用是一次请求一次回复。而 Agent 的一次任务可能涉及多轮 LLM 调用，成本呈倍数放大：

```
用户请求 -> LLM 决策（5K in + 500 out）
         -> 工具调用 -> 工具结果
         -> LLM 分析（8K in + 800 out）  <- 上下文累积！
         -> 工具调用 -> 工具结果
         -> LLM 总结（12K in + 1K out）  <- 持续增长

总计: 25K input + 2.3K output
实际费用可能是单次调用的 5-10 倍
```

### 成本计算器

```python
"""cost_calculator.py — LLM 调用成本计算"""

# 2025 年主流模型定价参考（每百万 Token，美元）
PRICING = {
    "claude-opus-4-20250514": {
        "input": 15.0,
        "output": 75.0,
        "cache_write": 18.75,
        "cache_read": 1.5,
    },
    "claude-sonnet-4-20250514": {
        "input": 3.0,
        "output": 15.0,
        "cache_write": 3.75,
        "cache_read": 0.3,
    },
    "claude-haiku-3-20250414": {
        "input": 0.25,
        "output": 1.25,
        "cache_write": 0.3,
        "cache_read": 0.03,
    },
    "gpt-4o": {
        "input": 2.5,
        "output": 10.0,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.6,
    },
}

def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """计算单次调用成本（美元）"""
    price = PRICING[model]
    cost = (
        (input_tokens - cache_read_tokens) * price["input"] / 1_000_000
        + output_tokens * price["output"] / 1_000_000
        + cache_read_tokens * price.get("cache_read", price["input"]) / 1_000_000
        + cache_write_tokens * price.get("cache_write", price["input"]) / 1_000_000
    )
    return round(cost, 6)

# 示例
cost = calculate_cost("claude-sonnet-4-20250514", input_tokens=5000, output_tokens=1000)
print(f"单次调用成本: ${cost:.4f}")  # $0.0300
```

::: warning 成本警示
一个设计不当的 Agent 可能在单次任务中消耗数美元。面向公众用户时，不做成本控制可能导致账单失控。
:::

### 用户级预算控制

```python
"""cost_tracker.py — 用户级成本追踪与预算控制"""

import time
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class CostTracker:
    """用户级别的成本追踪器"""

    daily_limit_cents: int = 100       # $1.00/天
    monthly_limit_cents: int = 2000    # $20.00/月

    _daily_costs: dict = field(default_factory=lambda: defaultdict(float))
    _monthly_costs: dict = field(default_factory=lambda: defaultdict(float))

    def record(self, user_id: str, cost_dollars: float):
        """记录一次成本"""
        cost_cents = cost_dollars * 100
        today = time.strftime("%Y-%m-%d")
        month = time.strftime("%Y-%m")

        self._daily_costs[f"{user_id}:{today}"] += cost_cents
        self._monthly_costs[f"{user_id}:{month}"] += cost_cents

    def check_budget(self, user_id: str) -> dict:
        """检查用户是否超出预算"""
        today = time.strftime("%Y-%m-%d")
        month = time.strftime("%Y-%m")

        daily = self._daily_costs.get(f"{user_id}:{today}", 0)
        monthly = self._monthly_costs.get(f"{user_id}:{month}", 0)

        return {
            "daily_used_cents": round(daily, 2),
            "daily_limit_cents": self.daily_limit_cents,
            "daily_remaining_pct": max(0, (1 - daily / self.daily_limit_cents) * 100),
            "monthly_used_cents": round(monthly, 2),
            "can_proceed": (
                daily < self.daily_limit_cents
                and monthly < self.monthly_limit_cents
            ),
            "reason": self._get_block_reason(daily, monthly),
        }

    def _get_block_reason(self, daily: float, monthly: float) -> str | None:
        if daily >= self.daily_limit_cents:
            return "已达今日用量上限，请明天再试"
        if monthly >= self.monthly_limit_cents:
            return "已达本月用量上限"
        return None
```

## Prompt Caching：节省 80%+ 输入成本

Agent 的每轮 LLM 调用都要发送 System Prompt 和工具定义，这些内容每次都一样。Anthropic 的 Prompt Caching 让你标记这些内容为可缓存，后续请求只需支付缓存读取的费用（仅原价的 10%）。

```python
"""prompt_caching.py — Prompt Caching 实现"""

import anthropic

client = anthropic.Anthropic()

# 假设有 20+ 个工具定义和一段长 System Prompt
SYSTEM_PROMPT = "你是一个专业的数据分析助手..."  # 很长的指令
TOOLS = [
    {"name": "search", "description": "搜索文档库...", "input_schema": {}},
    # ... 20+ 个工具定义
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    system=[{
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},  # 标记可缓存
    }],
    tools=[
        {**tool, "cache_control": {"type": "ephemeral"}}
        if i == len(TOOLS) - 1 else tool
        for i, tool in enumerate(TOOLS)
    ],
    messages=[{"role": "user", "content": "分析上周的用户增长数据"}],
)

# 查看缓存命中
usage = response.usage
print(f"输入 Token: {usage.input_tokens}")
print(f"缓存写入: {usage.cache_creation_input_tokens}")
print(f"缓存读取: {usage.cache_read_input_tokens}")
```

### 缓存节省了多少钱

```python
"""cache_savings.py — 计算缓存带来的成本节省"""

def calculate_cache_savings(
    cacheable_tokens: int,       # System Prompt + 工具定义的 token 数
    requests_per_hour: int,      # 每小时请求数
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    price = PRICING[model]

    # 无缓存：每次都按 input 定价
    no_cache = cacheable_tokens * requests_per_hour * price["input"] / 1_000_000

    # 有缓存：第一次写入 + 后续读取
    with_cache = (
        cacheable_tokens * price["cache_write"] / 1_000_000
        + cacheable_tokens * (requests_per_hour - 1) * price["cache_read"] / 1_000_000
    )

    return {
        "hourly_without_cache": f"${no_cache:.4f}",
        "hourly_with_cache": f"${with_cache:.4f}",
        "savings_pct": f"{(1 - with_cache / no_cache) * 100:.1f}%",
    }

# 示例：10000 token 的固定内容，每小时 100 次请求
print(calculate_cache_savings(10000, 100))
# {'hourly_without_cache': '$3.0000',
#  'hourly_with_cache': '$0.3345',
#  'savings_pct': '88.9%'}
```

## 模型路由：用对的模型做对的事

不是所有任务都需要最强的模型。翻译一句话用 Opus 就是浪费钱。

### 基于规则的路由（零额外成本）

```python
"""model_router.py — 成本感知的模型路由"""

class RuleBasedRouter:
    """基于规则的模型路由器——不需要额外的 LLM 调用"""

    SIMPLE_PATTERNS = [
        "翻译", "总结", "格式化", "提取", "分类",
        "是否", "对不对", "帮我改一下",
    ]
    COMPLEX_PATTERNS = [
        "写一个完整的", "设计一个系统", "分析", "调试",
        "多步骤", "比较.*优缺点", "为什么",
    ]

    def route(self, message: str, tool_count: int = 0) -> str:
        msg_lower = message.lower()
        msg_len = len(message)

        # 需要工具的任务用 Sonnet
        if tool_count > 0:
            return "claude-sonnet-4-20250514"

        # 短消息 + 简单模式 -> 便宜模型
        if msg_len < 100 and any(p in msg_lower for p in self.SIMPLE_PATTERNS):
            return "claude-haiku-3-20250414"

        # 长消息或复杂模式 -> 中等模型
        if msg_len > 500 or any(p in msg_lower for p in self.COMPLEX_PATTERNS):
            return "claude-sonnet-4-20250514"

        return "claude-haiku-3-20250414"  # 默认用便宜的
```

### 对话历史压缩

Agent 的上下文会随轮次越来越长。压缩历史消息可以直接减少 Token 消耗：

```python
"""history_compression.py — 对话历史压缩"""

import anthropic

async def compress_history(
    messages: list[dict],
    keep_recent: int = 4,
) -> list[dict]:
    """压缩对话历史：保留最近 N 条，之前的压缩为摘要"""
    if len(messages) <= keep_recent:
        return messages

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    # 用小模型生成摘要（便宜）
    client = anthropic.AsyncAnthropic()
    summary_response = await client.messages.create(
        model="claude-haiku-3-20250414",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                "用2-3句话总结以下对话的关键信息：\n\n"
                + "\n".join(
                    f"{m['role']}: {str(m['content'])[:200]}"
                    for m in old_messages
                )
            ),
        }],
    )
    summary = summary_response.content[0].text

    return [
        {"role": "user", "content": f"[对话历史摘要] {summary}"},
        {"role": "assistant", "content": "好的，我了解之前的讨论内容。"},
        *recent_messages,
    ]
```

### 成本感知的完整 Agent

把预算控制、模型路由、成本追踪整合在一起：

```python
"""cost_aware_agent.py — 成本感知的 Agent"""

import anthropic

class CostAwareAgent:
    """整合预算控制 + 模型路由 + 成本追踪的 Agent"""

    def __init__(self):
        self.tracker = CostTracker()
        self.router = RuleBasedRouter()
        self.client = anthropic.AsyncAnthropic()

    async def chat(self, user_id: str, message: str) -> str:
        # 1. 预算检查
        budget = self.tracker.check_budget(user_id)
        if not budget["can_proceed"]:
            return budget["reason"]

        # 2. 模型路由
        model = self.router.route(message)

        # 3. 执行
        response = await self.client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": message}],
        )

        # 4. 记录成本
        cost = calculate_cost(
            model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        self.tracker.record(user_id, cost)

        # 5. 用量提醒
        new_budget = self.tracker.check_budget(user_id)
        warning = ""
        if new_budget["daily_remaining_pct"] < 20:
            warning = (f"\n\n(提示：今日用量已使用 "
                       f"{100 - new_budget['daily_remaining_pct']:.0f}%)")

        return response.content[0].text + warning
```

## 灰度发布

新版本的模型或 System Prompt 不应该一下子推给所有用户。灰度发布让你先对小比例用户验证，确认无问题再全量推出：

```python
"""gray_release.py — 灰度发布控制"""

import hashlib

class GrayRelease:
    """基于用户 ID 的灰度发布"""

    def __init__(self, rollout_percentage: int = 10):
        self.rollout_percentage = rollout_percentage

    def is_in_experiment(self, user_id: str, experiment: str) -> bool:
        """判断用户是否在灰度范围内"""
        hash_input = f"{user_id}:{experiment}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        return bucket < self.rollout_percentage

    def get_model(self, user_id: str) -> str:
        """灰度切换模型版本"""
        if self.is_in_experiment(user_id, "new_model_v2"):
            return "claude-sonnet-4-20250514"       # 新版本（灰度测试中）
        return "claude-haiku-3-20250414"            # 稳定版本

    def get_system_prompt(self, user_id: str) -> str:
        """灰度切换 System Prompt 版本"""
        if self.is_in_experiment(user_id, "new_prompt_v3"):
            return "你是专业的数据分析助手。回答要简洁。（v3）"  # 新 Prompt
        return "你是一个数据分析助手。（v2）"                    # 旧 Prompt

# 使用：先 10% 用户用新版，观察指标后逐步扩大
gray = GrayRelease(rollout_percentage=10)
```

## CI/CD 流水线

自动化的测试、构建、部署流程是生产系统的标配：

```yaml
# .github/workflows/deploy.yml
name: Agent Service CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/ -v --cov=src --cov-report=xml
      - run: ruff check src/
      - run: mypy src/

  build:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Rolling update
        run: |
          kubectl set image deployment/agent-api \
            agent-api=ghcr.io/${{ github.repository }}:${{ github.sha }}
          kubectl rollout status deployment/agent-api --timeout=300s
```

## 健康检查与监控

```python
"""health_check.py — 多层健康检查"""

from fastapi import FastAPI
from datetime import datetime
import asyncio

app = FastAPI()

class HealthChecker:
    def __init__(self):
        self.start_time = datetime.utcnow()

    async def check_database(self) -> dict:
        try:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_redis(self) -> dict:
        try:
            await redis_client.ping()
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_llm_api(self) -> dict:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic()
            await asyncio.wait_for(
                client.messages.create(
                    model="claude-haiku-3-20250414",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "hi"}],
                ),
                timeout=10.0,
            )
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "degraded", "error": str(e)}

    async def full_check(self) -> dict:
        db, cache, llm = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_llm_api(),
        )
        all_healthy = all(c["status"] == "healthy" for c in [db, cache, llm])
        return {
            "status": "healthy" if all_healthy else "degraded",
            "uptime_seconds": (
                datetime.utcnow() - self.start_time
            ).total_seconds(),
            "checks": {"database": db, "redis": cache, "llm_api": llm},
        }

checker = HealthChecker()

@app.get("/health")
async def health():
    """轻量健康检查（K8s liveness probe）"""
    return {"status": "ok"}

@app.get("/health/ready")
async def readiness():
    """就绪检查（K8s readiness probe）"""
    result = await checker.full_check()
    return result
```

## 小结

高级生产工程化的四大策略：

1. **成本控制**：精确追踪每次调用的 Token 和费用，设置用户级日/月预算上限
2. **Prompt Caching**：标记固定上下文为可缓存，节省 80-90% 的输入成本
3. **模型路由**：简单任务走 Haiku（便宜 12 倍），复杂任务走 Sonnet，按需升级 Opus
4. **发布运维**：CI/CD 自动化（测试-构建-部署）+ 灰度发布（按比例放量）+ 多层健康检查

## 练习

1. 用 `calculate_cost` 计算你的 Agent 单次任务的平均成本，对比有无 Prompt Caching 的差异
2. 实现 `RuleBasedRouter`，准备 10 个不同复杂度的问题，验证路由是否合理
3. 为你的 Agent 设置 CI/CD 流水线，跑通 pytest + docker build + 部署的全流程

## 参考资源

- [Anthropic Pricing](https://www.anthropic.com/pricing) -- 模型定价
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) -- 缓存文档
- [Token Counting Guide](https://docs.anthropic.com/en/docs/build-with-claude/token-counting) -- Token 计算方法
- [12 Factor App](https://12factor.net/) -- 现代应用 12 要素
- [GitHub Actions 文档](https://docs.github.com/en/actions) -- CI/CD 平台
- [Kubernetes 官方文档](https://kubernetes.io/docs/) -- 容器编排
