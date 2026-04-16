# API 速查表

本附录汇总了 Agent 开发中最常用的 API 格式和参数说明，方便你在编码时快速查阅。

## Anthropic Messages API

### 基础调用

```python
import anthropic

client = anthropic.Anthropic()  # 自动读取 ANTHROPIC_API_KEY

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="你是一个有用的助手。",        # 可选：System Prompt
    messages=[
        {"role": "user", "content": "你好"},
    ],
)

print(response.content[0].text)
print(response.usage.input_tokens)
print(response.usage.output_tokens)
print(response.stop_reason)  # "end_turn" | "tool_use" | "max_tokens"
```

### 模型列表

| 模型 ID | 名称 | 上下文窗口 | 适用场景 |
|---------|------|-----------|---------|
| `claude-opus-4-20250514` | Claude Opus 4 | 200K | 复杂推理、代码生成 |
| `claude-sonnet-4-20250514` | Claude Sonnet 4 | 200K | 日常任务的最佳平衡 |
| `claude-haiku-3-5-20241022` | Claude Haiku 3.5 | 200K | 高速低成本任务 |

### 流式输出

```python
# 方式 1：stream 上下文管理器（推荐）
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "写一首诗"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

    # 获取最终消息（含 usage）
    final = stream.get_final_message()
    print(f"\nTokens: {final.usage.input_tokens} in, {final.usage.output_tokens} out")
```

```python
# 方式 2：底层事件流
with client.messages.stream(...) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                print(event.delta.text, end="")
        elif event.type == "message_delta":
            print(f"\nStop reason: {event.delta.stop_reason}")
```

### Tool Use

```python
# 1. 定义工具
tools = [
    {
        "name": "get_weather",
        "description": "查询城市天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
            },
            "required": ["city"],
        },
    },
]

# 2. 调用（带工具）
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "北京天气"}],
)

# 3. 处理 tool_use 响应
if response.stop_reason == "tool_use":
    for block in response.content:
        if block.type == "tool_use":
            tool_name = block.name       # "get_weather"
            tool_input = block.input     # {"city": "北京"}
            tool_use_id = block.id       # "toolu_xxx"

# 4. 返回工具结果
messages.append({"role": "assistant", "content": response.content})
messages.append({
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": "北京：晴，25度",
        }
    ],
})
```

### 多模态（图片输入）

```python
import base64

# 方式 1：base64 编码
with open("image.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data,
                },
            },
            {"type": "text", "text": "描述这张图片"},
        ],
    }],
)

# 方式 2：URL 引用
content = [
    {
        "type": "image",
        "source": {
            "type": "url",
            "url": "https://example.com/image.png",
        },
    },
    {"type": "text", "text": "描述这张图片"},
]
```

### 请求参数速查

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | 模型 ID |
| `max_tokens` | int | 是 | 最大输出 Token 数 |
| `messages` | array | 是 | 消息列表 |
| `system` | string | 否 | System Prompt |
| `tools` | array | 否 | 工具定义列表 |
| `temperature` | float | 否 | 采样温度 (0-1)，默认 1.0 |
| `top_p` | float | 否 | 核采样概率，默认无限制 |
| `top_k` | int | 否 | Top-K 采样 |
| `stop_sequences` | array | 否 | 自定义停止序列 |
| `metadata` | object | 否 | 请求元数据（如 user_id） |

### 响应结构

```python
# response 对象
response.id            # "msg_xxx" 消息 ID
response.type          # "message"
response.role          # "assistant"
response.content       # [ContentBlock, ...] 内容块列表
response.model         # "claude-sonnet-4-20250514"
response.stop_reason   # "end_turn" | "tool_use" | "max_tokens" | "stop_sequence"
response.usage.input_tokens   # 输入 Token 数
response.usage.output_tokens  # 输出 Token 数
```

---

## OpenAI Chat Completions API

### 基础调用

```python
from openai import OpenAI

client = OpenAI()  # 自动读取 OPENAI_API_KEY

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好"},
    ],
)

print(response.choices[0].message.content)
print(response.usage.prompt_tokens)
print(response.usage.completion_tokens)
```

### 流式输出

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

### Function Calling (Tool Use)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                },
                "required": ["city"],
            },
        },
    },
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京天气"}],
    tools=tools,
    tool_choice="auto",
)

# 处理工具调用
message = response.choices[0].message
if message.tool_calls:
    for call in message.tool_calls:
        name = call.function.name
        args = json.loads(call.function.arguments)
        call_id = call.id
```

---

## Anthropic vs OpenAI 格式对比

| 特性 | Anthropic | OpenAI |
|------|-----------|--------|
| System Prompt | `system` 参数（顶级） | `messages` 中 role="system" |
| 工具定义 | `tools` (顶级) | `tools` (顶级，多一层 type+function) |
| 工具调用信号 | `stop_reason == "tool_use"` | `message.tool_calls` 非空 |
| 工具结果格式 | `tool_result` content 块 | role="tool" 消息 |
| 流式事件 | Server-Sent Events | Server-Sent Events |
| Token 统计 | `response.usage` | `response.usage` |
| 多模态 | content 块数组 | content 块数组 |

---

## 常见错误码

### Anthropic

| 状态码 | 错误类型 | 含义 | 处理建议 |
|--------|---------|------|---------|
| 400 | `invalid_request_error` | 请求格式错误 | 检查参数格式 |
| 401 | `authentication_error` | API Key 无效 | 检查 ANTHROPIC_API_KEY |
| 403 | `permission_error` | 权限不足 | 检查 API Key 权限 |
| 429 | `rate_limit_error` | 请求过于频繁 | 添加重试逻辑和退避 |
| 500 | `api_error` | 服务端错误 | 重试或联系支持 |
| 529 | `overloaded_error` | API 过载 | 稍后重试 |

### OpenAI

| 状态码 | 含义 | 处理建议 |
|--------|------|---------|
| 401 | API Key 无效 | 检查 OPENAI_API_KEY |
| 429 | 速率限制或配额不足 | 退避重试或检查账户余额 |
| 500 | 服务端错误 | 重试 |
| 503 | 服务过载 | 稍后重试 |

### 通用重试模式

```python
import time
import anthropic

client = anthropic.Anthropic()

def call_with_retry(func, max_retries=3, base_delay=1):
    """带指数退避的重试"""
    for attempt in range(max_retries):
        try:
            return func()
        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"速率限制，{delay}s 后重试...")
            time.sleep(delay)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay)
            else:
                raise
```

## Tool Use 格式速查

### Anthropic Tool Schema

```json
{
  "name": "tool_name",
  "description": "工具的详细描述，帮助模型理解何时使用。",
  "input_schema": {
    "type": "object",
    "properties": {
      "param1": {
        "type": "string",
        "description": "参数说明"
      },
      "param2": {
        "type": "integer",
        "description": "参数说明",
        "default": 10
      }
    },
    "required": ["param1"]
  }
}
```

### OpenAI Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "tool_name",
    "description": "工具的详细描述。",
    "parameters": {
      "type": "object",
      "properties": {
        "param1": {
          "type": "string",
          "description": "参数说明"
        }
      },
      "required": ["param1"]
    }
  }
}
```

::: tip 工具描述最佳实践
1. 描述要具体说明**什么时候应该使用**这个工具
2. 参数的 description 要说明格式和约束（如"日期格式: YYYY-MM-DD"）
3. 提供默认值减少模型的决策负担
4. 使用 enum 约束有限选项（如 `"enum": ["asc", "desc"]`）
:::

## 参考资源

- [Anthropic API Reference](https://docs.anthropic.com/en/api/messages) -- 完整 API 文档
- [Anthropic SDK (Python)](https://github.com/anthropics/anthropic-sdk-python) -- Python SDK 源码
- [Anthropic SDK (TypeScript)](https://github.com/anthropics/anthropic-sdk-typescript) -- TypeScript SDK 源码
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat) -- 完整 API 文档
- [OpenAI SDK (Python)](https://github.com/openai/openai-python) -- Python SDK 源码
- [Anthropic Pricing](https://www.anthropic.com/pricing) -- 模型定价
- [OpenAI Pricing](https://openai.com/pricing) -- 模型定价
