# Agent 框架高级：源码分析与自建框架

::: tip 学习目标
- 理解主流框架的核心设计模式和源码结构
- 掌握自建 Agent 框架的设计哲学：组合优于继承、接口稳定实现可变、中间件管道
- 完整实现一个约 500 行代码的 Mini Agent Framework，包含 Model Provider、Tool Registry、Memory Manager、Middleware Pipeline 和 Agent Runner

**学完你能做到：** 阅读和理解 LangChain/OpenAI Agents SDK 的核心源码，从零构建一个可投入生产的轻量 Agent 框架。
:::

## 为什么要自建框架

使用第三方框架可以快速起步，但在生产环境中往往遇到瓶颈：

1. **过度抽象**：框架为了通用性引入大量抽象层，调试困难
2. **版本不稳定**：快速迭代的框架频繁 breaking changes
3. **性能开销**：通用化带来的序列化、适配、中间件开销
4. **定制受限**：特殊需求难以在框架约束内实现

::: info 何时该自建
- 你的 Agent 逻辑相对固定，不需要框架的通用调度能力
- 你需要极致的性能和可控性
- 你的团队有能力维护核心模块
- 你希望深入理解 Agent 的底层运作机制（学习目的）
:::

### 现有框架的优缺点

| 框架 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **LangChain** | 生态丰富、社区大 | 抽象过重、API 变化频繁 | 快速原型 |
| **LangGraph** | 图状态机清晰 | 学习曲线陡 | 复杂工作流 |
| **CrewAI** | 多 Agent 协作直观 | 灵活性不足 | 多 Agent 场景 |
| **OpenAI Agents SDK** | 简洁、类型安全 | 绑定 OpenAI | OpenAI 生态 |
| **自建框架** | 完全可控 | 需自行维护 | 生产定制 |

## 设计哲学

```python
"""
Mini Agent Framework 设计原则：
1. 简单 -- 核心代码不超过 500 行
2. 可扩展 -- 插件化的 Model/Tool/Memory
3. 可测试 -- 每个模块可独立测试
4. 透明 -- 没有隐藏的魔法，代码即文档
"""
```

### 原则 1：组合优于继承

```python
# 坏的设计：深度继承
class BaseAgent: ...
class ToolAgent(BaseAgent): ...
class RAGToolAgent(ToolAgent): ...
class RAGToolMemoryAgent(RAGToolAgent): ...  # 无穷无尽

# 好的设计：组合
class Agent:
    def __init__(self, model, tools, memory):
        self.model = model      # 可插拔的模型
        self.tools = tools      # 可插拔的工具
        self.memory = memory    # 可插拔的记忆
```

### 原则 2：接口稳定，实现可变

```python
from abc import ABC, abstractmethod

# 稳定的接口
class ModelProvider(ABC):
    @abstractmethod
    async def generate(self, messages, tools=None) -> ModelResponse:
        ...

# 可变的实现 -- 换模型只需换实现类
class AnthropicProvider(ModelProvider): ...
class OpenAIProvider(ModelProvider): ...
```

### 原则 3：中间件管道

```python
# 类似 Express/Koa 的中间件模式
pipeline = [
    InputValidator(),      # 输入验证
    CostTracker(),          # 成本追踪
    RateLimiter(),          # 速率限制
    AgentRunner(),          # 核心执行
    OutputValidator(),      # 输出验证
]
```

## 核心抽象层

### 类型定义

```python
"""mini_agent/types.py -- 核心类型定义"""

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class ToolResult:
    tool_call_id: str
    output: str
    is_error: bool = False

@dataclass
class ModelResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Any = None

@dataclass
class AgentResult:
    output: str
    iterations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_calls_made: list[str] = field(default_factory=list)
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
```

### Model Provider

```python
"""mini_agent/models.py -- 模型提供者"""

from abc import ABC, abstractmethod
import anthropic
from .types import ModelResponse, ToolCall, ToolDefinition

class ModelProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict], tools: list[ToolDefinition] | None = None,
                        system: str = "", max_tokens: int = 4096) -> ModelResponse:
        ...

class AnthropicProvider(ModelProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools]

    async def generate(self, messages, tools=None, system="", max_tokens=4096):
        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        resp = await self.client.messages.create(**kwargs)

        text_parts, tool_calls = [], []
        for block in resp.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        return ModelResponse(
            content="".join(text_parts), tool_calls=tool_calls,
            stop_reason=resp.stop_reason,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
        )

class OpenAIProvider(ModelProvider):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        import openai
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)

    def _format_tools(self, tools):
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                 "parameters": t.parameters}} for t in tools]

    async def generate(self, messages, tools=None, system="", max_tokens=4096):
        import json
        msgs = messages.copy()
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": msgs}
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id, name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return ModelResponse(
            content=choice.message.content or "", tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            model=self.model,
        )
```

### Tool Registry

`@tool` 装饰器从函数签名自动生成 JSON Schema，注册表统一管理和执行：

```python
"""mini_agent/tools.py -- 工具注册与执行"""

import asyncio
import inspect
from typing import Callable, get_type_hints, Any
from .types import ToolDefinition, ToolCall, ToolResult

TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array"}

def tool(name: str | None = None, description: str | None = None):
    """工具装饰器 -- 从函数签名自动生成 schema"""
    def decorator(func):
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        props, required = {}, []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            ptype = hints.get(pname, str)
            props[pname] = {"type": TYPE_MAP.get(ptype, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        func._tool_def = ToolDefinition(
            name=name or func.__name__,
            description=(description or func.__doc__ or "").strip(),
            parameters={"type": "object", "properties": props, "required": required},
            handler=func,
        )
        return func
    return decorator

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, func_or_def):
        if hasattr(func_or_def, "_tool_def"):
            d = func_or_def._tool_def
        elif isinstance(func_or_def, ToolDefinition):
            d = func_or_def
        else:
            raise TypeError("使用 @tool 装饰器或传入 ToolDefinition")
        self._tools[d.name] = d
        return func_or_def

    @property
    def definitions(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    async def execute(self, call: ToolCall) -> ToolResult:
        defn = self._tools.get(call.name)
        if not defn:
            return ToolResult(call.id, f"未知工具: {call.name}", is_error=True)
        try:
            fn = defn.handler
            result = await fn(**call.arguments) if asyncio.iscoroutinefunction(fn) else fn(**call.arguments)
            return ToolResult(call.id, str(result))
        except Exception as e:
            return ToolResult(call.id, f"执行错误: {e}", is_error=True)

    async def execute_parallel(self, calls: list[ToolCall]) -> list[ToolResult]:
        return list(await asyncio.gather(*[self.execute(c) for c in calls]))
```

### Memory Manager

```python
"""mini_agent/memory.py -- 记忆管理"""

class BufferMemory:
    """简单缓冲记忆 -- 保留最近 N 条消息"""
    def __init__(self, max_messages: int = 100):
        self.max = max_messages
        self._messages: list[dict] = []

    def add(self, message: dict):
        self._messages.append(message)
        if len(self._messages) > self.max:
            self._messages = self._messages[-self.max:]

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self):
        self._messages.clear()

    @property
    def size(self) -> int:
        return len(self._messages)

class WindowMemory:
    """滑动窗口记忆 -- 保留最近 N 轮对话"""
    def __init__(self, window_size: int = 10):
        self.window = window_size
        self._messages: list[dict] = []

    def add(self, message: dict):
        self._messages.append(message)

    def get_messages(self) -> list[dict]:
        keep = self.window * 2  # 每轮 = user + assistant
        return self._messages[-keep:] if len(self._messages) > keep else list(self._messages)

    def clear(self):
        self._messages.clear()
```

## 插件系统

### 中间件管道

借鉴 Express/Koa 的洋葱模型 -- 每个中间件可以在 Agent 执行前后插入逻辑：

```python
"""mini_agent/middleware.py -- 中间件系统"""

import time
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, field

@dataclass
class Context:
    metadata: dict = field(default_factory=dict)

class Middleware(ABC):
    @abstractmethod
    async def __call__(self, ctx: Context, next_fn) -> Any:
        ...

class Pipeline:
    def __init__(self):
        self._mw: list[Middleware] = []

    def use(self, mw: Middleware):
        self._mw.append(mw)
        return self

    async def run(self, ctx: Context, core_fn):
        index = 0
        async def next_fn():
            nonlocal index
            if index < len(self._mw):
                mw = self._mw[index]
                index += 1
                return await mw(ctx, next_fn)
            else:
                return await core_fn(ctx)
        return await next_fn()
```

### 内置中间件

```python
class TimingMiddleware(Middleware):
    """耗时统计"""
    async def __call__(self, ctx, next_fn):
        start = time.monotonic()
        result = await next_fn()
        ctx.metadata["duration_ms"] = int((time.monotonic() - start) * 1000)
        return result

class CostMiddleware(Middleware):
    """成本追踪"""
    PRICES = {
        "claude-sonnet-4-20250514": (3.0, 15.0),
        "claude-haiku-3-20250414": (0.25, 1.25),
    }
    async def __call__(self, ctx, next_fn):
        result = await next_fn()
        if result and hasattr(result, "total_input_tokens"):
            pi, po = self.PRICES.get(result.model, (3.0, 15.0))
            ctx.metadata["cost_usd"] = round(
                result.total_input_tokens * pi / 1e6 + result.total_output_tokens * po / 1e6, 6)
        return result

class MaxIterationsMiddleware(Middleware):
    """迭代次数限制"""
    def __init__(self, max_iter: int = 25):
        self.max = max_iter
    async def __call__(self, ctx, next_fn):
        ctx.metadata["max_iterations"] = self.max
        return await next_fn()
```

### Hook 系统

比中间件更细粒度的生命周期钩子：

```python
"""Hook 系统 -- 细粒度的生命周期钩子"""

import asyncio
from typing import Callable, Any
from collections import defaultdict

class HookManager:
    VALID_HOOKS = [
        "before_llm_call",       # LLM 调用前
        "after_llm_call",        # LLM 调用后
        "before_tool_call",      # 工具调用前
        "after_tool_call",       # 工具调用后
        "on_error",              # 发生错误时
        "on_iteration",          # 每次迭代时
        "on_complete",           # 任务完成时
    ]

    def __init__(self):
        self._hooks: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, handler: Callable):
        if event not in self.VALID_HOOKS:
            raise ValueError(f"未知钩子: {event}. 可用: {self.VALID_HOOKS}")
        self._hooks[event].append(handler)

    async def emit(self, event: str, **kwargs) -> list[Any]:
        results = []
        for handler in self._hooks.get(event, []):
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**kwargs)
            else:
                result = handler(**kwargs)
            results.append(result)
        return results

# 使用示例
hooks = HookManager()
hooks.on("before_llm_call", lambda messages, **kw:
    print(f"即将调用 LLM，消息数: {len(messages)}"))
hooks.on("after_tool_call", lambda tool_name, result, **kw:
    print(f"工具 {tool_name} 执行完成"))
```

### 动态工具注册

支持运行时增减工具，按上下文过滤相关工具：

```python
class DynamicToolRegistry:
    def __init__(self):
        self._tools = {}
        self._on_change_callbacks = []

    def register(self, tool_def):
        if hasattr(tool_def, "_tool_def"):
            defn = tool_def._tool_def
        else:
            defn = tool_def
        self._tools[defn.name] = defn
        self._notify_change("added", defn.name)

    def unregister(self, name: str):
        if name in self._tools:
            del self._tools[name]
            self._notify_change("removed", name)

    def on_change(self, callback):
        self._on_change_callbacks.append(callback)

    def _notify_change(self, action: str, tool_name: str):
        for cb in self._on_change_callbacks:
            cb(action, tool_name)

    def get_tools_for_context(self, context: str | None = None) -> list:
        """根据上下文返回相关工具（而非全部）"""
        if not context:
            return list(self._tools.values())
        relevant = []
        for defn in self._tools.values():
            desc = f"{defn.name} {defn.description}".lower()
            if any(kw in desc for kw in context.lower().split()):
                relevant.append(defn)
        return relevant or list(self._tools.values())
```

## 完整实现：Mini Agent Framework

将所有模块整合为完整框架。项目结构：

```
mini_agent/
├── __init__.py          # 对外导出
├── types.py             # 类型定义（58 行）
├── models.py            # Model Provider（95 行）
├── tools.py             # Tool Registry（90 行）
├── memory.py            # Memory Manager（60 行）
├── middleware.py         # 中间件系统（80 行）
├── agent.py             # Agent Runner（120 行）
总计约 503 行
```

### Agent Runner -- 核心循环

```python
"""mini_agent/agent.py -- Agent Runner 核心循环"""

from .types import AgentResult, ModelResponse
from .models import ModelProvider
from .tools import ToolRegistry
from .memory import BufferMemory
from .middleware import Pipeline, Context

class Agent:
    """Mini Agent Framework 核心类"""

    def __init__(
        self,
        model: ModelProvider,
        tools: ToolRegistry | None = None,
        memory: BufferMemory | None = None,
        system_prompt: str = "",
        max_iterations: int = 25,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.tools = tools or ToolRegistry()
        self.memory = memory or BufferMemory()
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.pipeline = Pipeline()

    def use(self, middleware):
        """添加中间件"""
        self.pipeline.use(middleware)
        return self

    async def run(self, message: str) -> AgentResult:
        """运行 Agent"""
        ctx = Context()
        ctx.metadata["max_iterations"] = self.max_iterations

        async def core(ctx):
            return await self._agent_loop(message, ctx)

        return await self.pipeline.run(ctx, core)

    async def _agent_loop(self, message: str, ctx: Context) -> AgentResult:
        """Agent 核心循环"""
        self.memory.add({"role": "user", "content": message})
        messages = self.memory.get_messages()
        tool_defs = self.tools.definitions if self.tools.definitions else None

        total_in, total_out = 0, 0
        tool_calls_made = []
        max_iter = ctx.metadata.get("max_iterations", self.max_iterations)

        for iteration in range(max_iter):
            response: ModelResponse = await self.model.generate(
                messages=messages,
                tools=tool_defs,
                system=self.system_prompt,
                max_tokens=self.max_tokens,
            )

            total_in += response.input_tokens
            total_out += response.output_tokens

            # 没有工具调用 -> 返回最终结果
            if not response.has_tool_calls:
                self.memory.add({"role": "assistant", "content": response.content})
                return AgentResult(
                    output=response.content,
                    iterations=iteration + 1,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                    tool_calls_made=tool_calls_made,
                    model=response.model,
                )

            # 处理工具调用
            messages.append({"role": "assistant", "content": response.content,
                           "_raw_content": response})

            results = await self.tools.execute_parallel(response.tool_calls)
            tool_calls_made.extend([tc.name for tc in response.tool_calls])

            tool_result_content = [
                {"type": "tool_result", "tool_use_id": r.tool_call_id,
                 "content": r.output, "is_error": r.is_error}
                for r in results
            ]
            messages.append({"role": "user", "content": tool_result_content})

        return AgentResult(
            output="达到最大迭代次数",
            iterations=max_iter,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            tool_calls_made=tool_calls_made,
            model=response.model,
        )
```

### 对外导出

```python
"""mini_agent -- 一个约 500 行代码的 Agent 框架"""

from .agent import Agent
from .models import AnthropicProvider, OpenAIProvider, ModelProvider
from .tools import tool, ToolRegistry, ToolDefinition
from .memory import BufferMemory, WindowMemory
from .middleware import Pipeline, TimingMiddleware, CostMiddleware, Middleware, Context
from .types import AgentResult, ModelResponse

__all__ = [
    "Agent", "AnthropicProvider", "OpenAIProvider", "ModelProvider",
    "tool", "ToolRegistry", "ToolDefinition",
    "BufferMemory", "WindowMemory",
    "Pipeline", "TimingMiddleware", "CostMiddleware", "Middleware", "Context",
    "AgentResult", "ModelResponse",
]
```

### 使用示例

```python
"""example.py -- Mini Agent Framework 使用示例"""

import asyncio
from mini_agent import (
    Agent, AnthropicProvider, ToolRegistry, tool,
    BufferMemory, TimingMiddleware, CostMiddleware,
)

# 1. 定义工具
@tool(name="calculate", description="计算数学表达式")
def calculate(expression: str) -> str:
    """计算数学表达式。expression: 合法的 Python 数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

@tool(name="get_time", description="获取当前时间")
def get_time() -> str:
    """获取当前日期和时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 2. 注册工具
tools = ToolRegistry()
tools.register(calculate)
tools.register(get_time)

# 3. 创建 Agent
agent = Agent(
    model=AnthropicProvider(model="claude-sonnet-4-20250514"),
    tools=tools,
    memory=BufferMemory(max_messages=50),
    system_prompt="你是一个有用的助手。使用工具来回答问题。",
    max_iterations=10,
)

# 4. 添加中间件
agent.use(TimingMiddleware())
agent.use(CostMiddleware())

# 5. 运行
async def main():
    result = await agent.run("现在几点了？然后帮我算一下 123 * 456 + 789")
    print(f"回复: {result.output}")
    print(f"迭代次数: {result.iterations}")
    print(f"Token 用量: {result.total_tokens}")
    print(f"工具调用: {result.tool_calls_made}")

asyncio.run(main())
```

## 小结

- 自建框架不是重复造轮子，而是为了：深入理解底层机制、获得完全可控的生产系统、按需定制不受框架限制
- 核心设计哲学：组合优于继承、接口稳定实现可变、中间件管道模式
- **Model Provider** 统一不同 LLM 的调用接口，换模型只需换实现类
- **Tool Registry** 用装饰器自动生成 schema，注册表统一管理和并行执行
- **Memory Manager** 可插拔的记忆后端，从简单缓冲到滑动窗口
- **Middleware Pipeline** 洋葱模型，before/after 逻辑自由组合
- **Hook 系统** 细粒度的生命周期事件，支持动态工具注册
- 500 行代码涵盖了生产级框架的所有核心组件

## 练习题

1. 给 Mini Agent Framework 添加一个 `SummaryMemory`：当消息超过阈值时，自动调用 LLM 压缩历史消息为摘要。
2. 实现一个 `RetryMiddleware`：当 LLM 调用失败时自动重试，支持指数退避和最大重试次数配置。
3. 给框架添加 Streaming 支持：实现 `ModelProvider.stream()` 方法和 `Agent.run_stream()` 方法。
4. 阅读 [OpenAI Agents SDK 源码](https://github.com/openai/openai-agents-python)，对比它的抽象设计与本章实现的异同。

## 参考资源

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) -- 约 1000 行的官方框架参考
- [Pydantic AI](https://github.com/pydantic/pydantic-ai) -- 类型安全的 Agent 框架
- [smolagents (Hugging Face)](https://github.com/huggingface/smolagents) -- 轻量级 Agent 库
- [LangChain 源码](https://github.com/langchain-ai/langchain) -- 参考其抽象设计
- [Clean Architecture (Robert C. Martin)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) -- 架构设计原则
- [Koa.js 中间件机制](https://koajs.com/) -- 洋葱模型参考
- [arXiv:2308.08155 - A Survey on LLM-based Autonomous Agents](https://arxiv.org/abs/2308.08155) -- Agent 架构综述
- [Anthropic Claude Agent 文档](https://docs.anthropic.com/en/docs/build-with-claude/agentic-systems) -- 官方 Agent 指南
