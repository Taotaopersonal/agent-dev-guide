# Python 基础 · 高级

::: info 学习目标
- 用 FastAPI 构建完整的 Web API 服务
- 掌握 Python 高级类型系统（Protocol、Generic、TypeVar）
- 学会性能分析和常见优化技巧
- 理解生产级 Python 服务的最佳实践
- 学完能独立开发和部署生产级 Python 服务

预计学习时间：3-4 小时
:::

## FastAPI 完整实战

FastAPI 是目前 Python Web 框架的首选，也是 Agent 后端服务最常用的框架。它基于 Pydantic 做数据验证，基于 ASGI 支持异步，自动生成 API 文档。

### 第一个 FastAPI 应用

```bash
# 安装依赖
uv add fastapi uvicorn
```

```python
"""main.py -- 第一个 FastAPI 应用"""
from fastapi import FastAPI

app = FastAPI(title="My Agent API", version="0.1.0")

@app.get("/")
async def root():
    return {"message": "Hello, Agent!"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

```bash
# 启动服务
uvicorn main:app --reload --port 8000

# 访问 http://localhost:8000/docs 查看自动生成的 API 文档
```

### 请求和响应模型

FastAPI 与 Pydantic 深度集成，请求体和响应都用 Pydantic 模型定义：

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import anthropic

app = FastAPI()
client = anthropic.Anthropic()

# 请求模型
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    system: str = Field(default="你是一位友好的 AI 助手。")
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=1)

# 响应模型
class ChatResponse(BaseModel):
    content: str
    model: str
    input_tokens: int
    output_tokens: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话接口"""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=request.system,
            messages=[{"role": "user", "content": request.message}],
        )
        return ChatResponse(
            content=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    except anthropic.BadRequestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")
    except Exception as e:
        raise HTTPException(status_code=500, detail="服务内部错误")
```

### 流式输出端点

Agent 应用的核心能力 -- 让前端实时看到 LLM 生成的文字：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import anthropic
import json

app = FastAPI()

# 允许跨域（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic()

class ChatRequest(BaseModel):
    message: str
    system: str = "你是一位友好的 AI 助手。"

async def generate_stream(request: ChatRequest):
    """生成 SSE 流"""
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=request.system,
        messages=[{"role": "user", "content": request.message}]
    ) as stream:
        for text in stream.text_stream:
            data = json.dumps({"type": "text", "content": text}, ensure_ascii=False)
            yield f"data: {data}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式对话端点"""
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
```

### 依赖注入

FastAPI 的依赖注入系统让你优雅地管理共享资源：

```python
from fastapi import FastAPI, Depends
from functools import lru_cache
import anthropic

app = FastAPI()

# 配置类
class Settings(BaseModel):
    anthropic_api_key: str = ""
    model_name: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024

@lru_cache
def get_settings() -> Settings:
    """加载配置（只执行一次）"""
    return Settings()

def get_llm_client(settings: Settings = Depends(get_settings)):
    """获取 LLM 客户端"""
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)

@app.post("/chat")
async def chat(
    message: str,
    client: anthropic.Anthropic = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
):
    response = client.messages.create(
        model=settings.model_name,
        max_tokens=settings.max_tokens,
        messages=[{"role": "user", "content": message}],
    )
    return {"content": response.content[0].text}
```

### 中间件和生命周期

```python
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import time
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("服务启动，初始化资源...")
    yield
    # 关闭时执行
    logger.info("服务关闭，清理资源...")

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    return response
```

## 类型系统高级用法

Python 的类型系统在 3.10+ 变得相当强大。在 Agent 开发中，良好的类型定义能让代码更可靠、IDE 提示更准确。

### Protocol -- 结构化子类型

Protocol 实现"鸭子类型"的类型检查。不需要继承，只要有相同的方法签名就算符合：

```python
from typing import Protocol

class LLMProvider(Protocol):
    """LLM 提供商协议 -- 任何实现了 chat 方法的类都符合"""
    def chat(self, messages: list[dict], **kwargs) -> str: ...

class ClaudeProvider:
    """不需要继承 LLMProvider，只要有 chat 方法就行"""
    def chat(self, messages: list[dict], **kwargs) -> str:
        # 调用 Claude API
        return "Claude 的回复"

class OpenAIProvider:
    def chat(self, messages: list[dict], **kwargs) -> str:
        # 调用 OpenAI API
        return "GPT 的回复"

def run_agent(provider: LLMProvider, task: str) -> str:
    """接受任何符合 LLMProvider 协议的对象"""
    return provider.chat([{"role": "user", "content": task}])

# 两个都可以传入，不需要继承关系
run_agent(ClaudeProvider(), "你好")
run_agent(OpenAIProvider(), "你好")
```

### Generic -- 泛型

```python
from typing import TypeVar, Generic
from pydantic import BaseModel

T = TypeVar("T")

class APIResponse(BaseModel, Generic[T]):
    """通用 API 响应包装"""
    success: bool
    data: T | None = None
    error: str | None = None

class User(BaseModel):
    name: str
    age: int

class Tool(BaseModel):
    name: str
    description: str

# 使用泛型 -- IDE 能正确推断 data 的类型
def get_user() -> APIResponse[User]:
    return APIResponse(success=True, data=User(name="Alice", age=25))

def get_tool() -> APIResponse[Tool]:
    return APIResponse(success=True, data=Tool(name="search", description="搜索"))

user_resp = get_user()
print(user_resp.data.name)  # IDE 知道这是 User 类型
```

### TypeVar 和 Callable

```python
from typing import TypeVar, Callable
from functools import wraps

T = TypeVar("T")

def cache_result(func: Callable[..., T]) -> Callable[..., T]:
    """带类型保持的缓存装饰器"""
    _cache: dict[str, T] = {}

    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        key = f"{args}-{kwargs}"
        if key not in _cache:
            _cache[key] = func(*args, **kwargs)
        return _cache[key]

    return wrapper

@cache_result
def expensive_computation(x: int) -> float:
    """返回类型信息被保留"""
    return x ** 0.5

result = expensive_computation(16)  # IDE 知道 result 是 float
```

### Literal 和 TypeAlias

```python
from typing import Literal, TypeAlias

# Literal -- 限定值范围
ModelName = Literal["claude-sonnet-4-20250514", "claude-haiku-3-5-20241022", "gpt-4o"]
StopReason = Literal["complete", "max_tokens", "tool_use"]

def create_client(model: ModelName) -> None:
    pass

create_client("claude-sonnet-4-20250514")  # OK
create_client("invalid-model")  # 类型检查报错

# TypeAlias -- 类型别名
MessageList: TypeAlias = list[dict[str, str]]
ToolResult: TypeAlias = dict[str, str | int | bool]
```

## 性能分析和优化

### 使用 cProfile 分析

```python
import cProfile
import pstats
from io import StringIO

def profile_function(func, *args, **kwargs):
    """分析函数性能"""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(10)  # 打印前 10 个最耗时的调用
    print(stream.getvalue())

    return result

# 使用
def process_data(data: list[dict]) -> list[str]:
    results = []
    for item in data:
        # 模拟数据处理
        results.append(item.get("name", "").upper())
    return results

data = [{"name": f"user_{i}"} for i in range(100000)]
profile_function(process_data, data)
```

### 常见优化技巧

```python
import time

# 1. 列表推导式比 for+append 快
def benchmark():
    data = list(range(100000))

    # 慢：for + append
    start = time.time()
    result1 = []
    for x in data:
        result1.append(x * 2)
    print(f"for+append: {time.time() - start:.4f}s")

    # 快：列表推导式
    start = time.time()
    result2 = [x * 2 for x in data]
    print(f"推导式: {time.time() - start:.4f}s")

benchmark()

# 2. 用 set 做成员检查（O(1) vs O(n)）
large_list = list(range(100000))
large_set = set(large_list)

# 慢
99999 in large_list  # O(n)
# 快
99999 in large_set   # O(1)

# 3. 字符串拼接用 join
parts = ["hello"] * 10000

# 慢
result = ""
for p in parts:
    result += p

# 快
result = "".join(parts)

# 4. 使用 lru_cache 缓存重复计算
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

fibonacci(100)  # 有缓存，瞬间完成
```

### 异步并发控制

在 Agent 应用中，你可能需要同时调用很多 API，但不能无限制并发（会触发限流）：

```python
import asyncio
import anthropic

async def controlled_batch(
    prompts: list[str],
    max_concurrent: int = 5
) -> list[str]:
    """控制并发数的批量 LLM 调用"""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[str] = []

    async def call_with_limit(prompt: str) -> str:
        async with semaphore:  # 限制同时运行的任务数
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

    tasks = [call_with_limit(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

# 使用：同时最多 5 个请求
# asyncio.run(controlled_batch(["问题1", "问题2", ...], max_concurrent=5))
```

## 生产级项目结构

一个完整的 Agent API 服务项目结构：

```
my-agent-api/
├── pyproject.toml          # 项目配置和依赖
├── src/
│   └── agent_api/
│       ├── __init__.py
│       ├── main.py         # FastAPI 入口
│       ├── config.py       # 配置管理
│       ├── models.py       # Pydantic 数据模型
│       ├── routes/
│       │   ├── __init__.py
│       │   └── chat.py     # 对话路由
│       ├── services/
│       │   ├── __init__.py
│       │   └── llm.py      # LLM 服务层
│       └── middleware/
│           ├── __init__.py
│           └── logging.py  # 日志中间件
├── tests/
│   ├── __init__.py
│   └── test_chat.py
└── Dockerfile
```

```python
"""src/agent_api/config.py -- 配置管理"""
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """从环境变量加载配置"""
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

```python
"""src/agent_api/services/llm.py -- LLM 服务层"""
import anthropic
from typing import Protocol

class LLMService(Protocol):
    async def chat(self, messages: list[dict], **kwargs) -> str: ...

class ClaudeLLMService:
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat(self, messages: list[dict], **kwargs) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=messages,
        )
        return response.content[0].text
```

## 小结

1. **FastAPI** 是 Agent 后端的首选框架，与 Pydantic 深度集成，自动生成 API 文档，原生支持异步
2. **Protocol** 实现鸭子类型的类型检查，让代码更灵活又保持类型安全
3. **Generic** 让你写出类型安全的通用组件，IDE 能正确推断泛型参数
4. **性能优化**：列表推导式、set 查找、join 拼接、lru_cache 缓存、Semaphore 并发控制
5. **生产级项目**要有清晰的分层结构：路由 -> 服务 -> 模型，配置从环境变量加载

## 练习

1. **FastAPI 练习**：用 FastAPI 构建一个完整的对话 API，包含 `/chat`（同步）和 `/chat/stream`（流式）两个端点，请求和响应都用 Pydantic 模型定义。

2. **类型系统练习**：定义一个 `LLMProvider` Protocol，包含 `chat` 和 `stream` 两个方法。分别为 Claude 和 OpenAI 实现具体类，然后写一个 `run_agent(provider: LLMProvider)` 函数，验证两个实现都能传入。

3. **性能练习**：写一个脚本，用 `asyncio.Semaphore` 控制并发数，批量处理 20 个 LLM 请求（最多同时 3 个），记录总耗时，对比无并发控制的版本。

4. **综合练习**：搭建一个完整的 Agent API 项目，包含项目结构、配置管理、LLM 服务层、对话路由、请求日志中间件，能用 `uvicorn` 启动并通过 `/docs` 查看 API 文档。

## 参考资源

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Python typing 文档](https://docs.python.org/3/library/typing.html)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Real Python - FastAPI 教程](https://realpython.com/fastapi-python-web-apis/)
