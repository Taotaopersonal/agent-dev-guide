# 生产工程化 · 入门篇

::: tip 学习目标
- 理解从 Python 脚本到可部署服务的转变
- 掌握标准的项目结构和配置管理方式
- 学会用环境变量安全地管理 API Key 等敏感信息
- 实现简单但有效的错误处理和日志记录
:::

::: info 学完你能做到
- 把一个 Notebook 里的 Agent 原型改造成标准项目结构
- 用 `.env` 文件和 Pydantic Settings 管理配置
- 写出规范的错误处理代码，而不是到处 `try-except pass`
- 让你的程序有清晰可查的日志输出
:::

## 从脚本到服务：为什么需要工程化

你的 Agent 原型可能长这样：

```python
# agent.py —— 一个典型的原型脚本
import anthropic

client = anthropic.Anthropic()  # API Key 硬编码在环境变量里

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=[{"role": "user", "content": input("你想问什么？")}],
)
print(response.content[0].text)
```

这个脚本能跑，但距离"别人也能用"差了十万八千里：

- **API Key 怎么管理？** 换台机器就跑不了
- **出错了怎么办？** 网络断了直接崩溃
- **别人怎么调用？** 只能命令行手动输入
- **出了 Bug 怎么查？** 没有日志，全靠 print

接下来我们一步步解决这些问题。

## 标准项目结构

一个可部署的 Agent 服务至少需要这样的结构：

```
my-agent-service/
├── src/
│   ├── __init__.py
│   ├── main.py           # 应用入口（FastAPI）
│   ├── agent.py           # Agent 核心逻辑
│   ├── config.py          # 配置管理
│   └── tools/             # 工具定义
│       ├── __init__.py
│       └── search.py
├── tests/
│   └── test_agent.py
├── .env                   # 环境变量（不提交到 Git！）
├── .env.example           # 环境变量模板（提交到 Git）
├── .gitignore
├── requirements.txt
└── README.md
```

::: warning 千万不要提交 .env 文件
`.env` 里存的是 API Key、数据库密码等敏感信息。在 `.gitignore` 中加上 `.env`，只提交 `.env.example` 作为模板。
:::

## 环境变量与配置管理

### 为什么不能硬编码配置

```python
# 错误示范 —— 硬编码 API Key
client = anthropic.Anthropic(api_key="sk-ant-api03-xxxx")  # 泄露！

# 错误示范 —— 硬编码模型名称
model = "claude-sonnet-4-20250514"  # 想换模型就得改代码
```

### 用 Pydantic Settings 管理配置

```python
"""config.py — 配置管理模块"""

from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """应用配置 —— 从环境变量自动读取，支持 .env 文件"""

    # API Keys（必填，不提供会启动报错）
    anthropic_api_key: str

    # 可选配置（有默认值）
    default_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    agent_timeout_seconds: int = 120

    # 日志配置
    log_level: str = "info"

    # 运行环境
    environment: str = "development"
    debug: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

@lru_cache()
def get_settings() -> Settings:
    """获取配置单例（整个应用生命周期只加载一次）"""
    return Settings()
```

对应的 `.env` 文件：

```bash
# .env —— 本地开发环境配置
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
DEFAULT_MODEL=claude-sonnet-4-20250514
LOG_LEVEL=debug
ENVIRONMENT=development
```

`.env.example` 文件（提交到 Git，告诉其他开发者需要哪些配置）：

```bash
# .env.example —— 配置模板
ANTHROPIC_API_KEY=your-api-key-here
DEFAULT_MODEL=claude-sonnet-4-20250514
LOG_LEVEL=info
ENVIRONMENT=development
```

### 在代码中使用配置

```python
"""agent.py — 使用配置的 Agent"""

import anthropic
from config import get_settings

settings = get_settings()

# API Key 从配置中读取，不再硬编码
client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

async def chat(message: str) -> str:
    response = await client.messages.create(
        model=settings.default_model,    # 模型名称可配置
        max_tokens=settings.max_tokens,  # Token 限制可配置
        messages=[{"role": "user", "content": message}],
    )
    return response.content[0].text
```

## 简单但有效的错误处理

### Agent 系统中常见的错误类型

| 错误类型 | 举例 | 正确处理方式 |
|---------|------|------------|
| 认证失败 | API Key 无效（401） | 不重试，提醒用户检查配置 |
| 请求无效 | 参数错误（400） | 不重试，修复代码 |
| 速率限制 | 请求太频繁（429） | 等一会儿再试 |
| 服务器错误 | API 临时故障（500） | 等一会儿再试 |
| 网络超时 | 网络不通 | 等一会儿再试，多次失败则报错 |

### 基础的重试机制

```python
"""retry.py — 简单的重试装饰器"""

import asyncio
import random
from functools import wraps

def retry_on_error(max_retries: int = 3, base_delay: float = 1.0):
    """带指数退避的重试装饰器

    指数退避的意思是：第一次等 1 秒，第二次等 2 秒，第三次等 4 秒...
    加上随机抖动（jitter），避免所有请求同时重试导致"惊群效应"。
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # 最后一次重试也失败了，不再等待
                    if attempt >= max_retries:
                        break

                    # 计算等待时间：指数递增 + 随机抖动
                    delay = base_delay * (2 ** attempt)
                    delay = random.uniform(0, delay)  # 随机取 [0, delay]
                    print(f"[重试] 第 {attempt + 1} 次，"
                          f"等待 {delay:.1f}s，原因: {e}")
                    await asyncio.sleep(delay)

            raise last_error
        return wrapper
    return decorator

# 使用示例
@retry_on_error(max_retries=3, base_delay=2.0)
async def call_llm(messages: list[dict]) -> str:
    """调用 LLM API（自动重试）"""
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=messages,
    )
    return response.content[0].text
```

### 区分"该重试"和"不该重试"的错误

```python
"""smart_error_handling.py — 更聪明的错误处理"""

import anthropic

async def safe_chat(message: str) -> str:
    """带错误分类的聊天函数"""
    client = anthropic.AsyncAnthropic()

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": message}],
        )
        return response.content[0].text

    except anthropic.AuthenticationError:
        # 401 —— API Key 有问题，重试没意义
        return "系统配置错误，请联系管理员。"

    except anthropic.BadRequestError as e:
        # 400 —— 请求参数有问题
        return f"请求参数错误: {e}"

    except anthropic.RateLimitError:
        # 429 —— 请求太频繁，应该重试
        # 实际项目中这里会触发重试逻辑
        return "系统繁忙，请稍后再试。"

    except anthropic.InternalServerError:
        # 500 —— 服务端临时故障，应该重试
        return "服务暂时不可用，请稍后再试。"

    except Exception as e:
        # 兜底处理
        print(f"[错误] 未知异常: {e}")
        return "处理过程中出现问题，请稍后重试。"
```

## 日志：从 print 到结构化记录

### 为什么不能用 print

`print` 的问题：没有时间戳、没有级别、没有上下文、生产环境看不到。

### 基础日志设置

```python
"""logging_setup.py — 结构化日志"""

import logging
import json
import sys

class JSONFormatter(logging.Formatter):
    """JSON 格式化器 —— 方便日志平台解析"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)

def setup_logging(level: str = "INFO"):
    """初始化日志系统"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    root.handlers = [handler]

# 在应用启动时调用一次
setup_logging("INFO")

# 在业务代码中使用
logger = logging.getLogger("agent")

async def chat_with_logging(message: str) -> str:
    logger.info("收到用户消息", extra={"message_length": len(message)})

    try:
        result = await call_llm([{"role": "user", "content": message}])
        logger.info("LLM 调用成功")
        return result
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}", exc_info=True)
        return "处理出错，请稍后重试。"
```

## 用 FastAPI 暴露为 HTTP 服务

最后一步，把你的 Agent 包装成一个 HTTP 服务，让前端或其他系统可以调用：

```python
"""main.py — 应用入口"""

from fastapi import FastAPI
from pydantic import BaseModel
from config import get_settings

settings = get_settings()
app = FastAPI(title="My Agent Service")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """聊天接口"""
    reply = await safe_chat(request.message)
    return ChatResponse(reply=reply)

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}
```

启动服务：

```bash
# 安装依赖
pip install fastapi uvicorn pydantic-settings anthropic

# 启动服务
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## 小结

从脚本到服务的关键步骤：

1. **项目结构**：清晰的目录组织，代码、配置、测试分离
2. **配置管理**：Pydantic Settings + `.env` 文件，敏感信息不入代码
3. **错误处理**：区分可重试和不可重试的错误，用指数退避处理瞬时故障
4. **日志记录**：JSON 格式的结构化日志，方便查问题
5. **HTTP 接口**：FastAPI 暴露服务，提供健康检查端点

## 练习

1. 用上面的项目结构创建一个新项目，实现一个简单的翻译 Agent
2. 给你的 Agent 添加一个 `max_concurrent_requests` 配置项，限制同时处理的请求数
3. 故意把 API Key 改错，观察错误处理是否按预期工作

## 参考资源

- [FastAPI 官方文档](https://fastapi.tiangolo.com/) -- Python Web 框架，专为 API 设计
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) -- 配置管理库
- [The Twelve-Factor App](https://12factor.net/) -- 现代应用设计 12 原则（尤其是第三条：配置）
- [Anthropic API Error Handling](https://docs.anthropic.com/en/api/errors) -- API 错误码参考
