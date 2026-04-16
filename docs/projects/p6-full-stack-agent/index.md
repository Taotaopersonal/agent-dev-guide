# P6: 全栈 Agent 应用

::: info 项目信息
**难度**: 高级 | **代码量**: ~2000 行 | **预计时间**: 15-20 小时
**对应章节**: 高级篇全部（第 13-16 章）
**定位**: 全书毕业项目
:::

## 项目目标

构建一个**生产级的全栈 Agent 应用** -- **AgentHub**。这是一个带 Web UI 的 AI Agent 平台，集成了多工具调用、RAG 知识库、对话历史持久化、用户认证、执行轨迹可视化、成本统计等完整功能。

这是全书的毕业项目，综合运用前面所有章节的知识。完成它，你就具备了独立开发生产级 Agent 应用的能力。

### 功能全景

- [x] **Web Chat UI** -- 现代化的对话界面（Vue 3）
- [x] **多工具 Agent** -- 搜索、代码执行、文件管理
- [x] **RAG 知识库** -- 文档导入、语义检索、智能问答
- [x] **对话历史** -- PostgreSQL 持久化存储
- [x] **用户认证** -- JWT Token 认证
- [x] **执行轨迹可视化** -- 实时展示 Agent 的思考和工具调用过程
- [x] **成本统计面板** -- Token 用量和费用追踪
- [x] **Docker 部署** -- 一键部署

## 架构设计

```mermaid
graph TB
    subgraph "前端 (Vue 3)"
        A[Chat UI] --> B[对话管理]
        A --> C[知识库管理]
        A --> D[执行轨迹面板]
        A --> E[成本统计面板]
    end

    subgraph "后端 (FastAPI)"
        F[API Gateway] --> G[Agent Engine]
        F --> H[Auth Service]
        F --> I[Chat History Service]
        G --> J[Tool Registry]
        G --> K[RAG Module]
    end

    subgraph "存储层"
        L[(PostgreSQL<br/>对话历史+用户)]
        M[(ChromaDB<br/>向量索引)]
    end

    subgraph "外部服务"
        N[Anthropic API]
        O[Web Search API]
    end

    A <-->|WebSocket| F
    G --> L
    G --> M
    G --> N
    J --> O
    I --> L
    K --> M

    style A fill:#e8f5e9
    style G fill:#fff3e0
    style L fill:#e3f2fd
    style M fill:#e3f2fd
```

## 技术栈

| 层 | 技术 | 理由 |
|----|------|------|
| 前端 | Vue 3 + Vite + TailwindCSS | 发挥前端优势，组件化开发 |
| 后端 | FastAPI + Uvicorn | 异步原生，自动生成 API 文档 |
| Agent 引擎 | 自建（基于 P2 的 Agent 循环） | 深度理解，完全可控 |
| 数据库 | PostgreSQL | 成熟稳定，支持 JSON 字段 |
| 向量库 | ChromaDB | 轻量嵌入式，无需独立部署 |
| 实时通信 | WebSocket | Agent 流式输出必需 |
| 认证 | JWT | 无状态，适合 API 服务 |
| 部署 | Docker Compose | 多服务编排 |

## 项目结构

```
agent-hub/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI 入口
│   │   ├── config.py            # 配置管理
│   │   ├── auth/
│   │   │   ├── __init__.py
│   │   │   ├── router.py        # 认证路由
│   │   │   ├── service.py       # JWT 逻辑
│   │   │   └── models.py        # 用户模型
│   │   ├── agent/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py        # Agent 核心引擎
│   │   │   ├── tools.py         # 工具注册表
│   │   │   └── tracer.py        # 执行轨迹记录
│   │   ├── chat/
│   │   │   ├── __init__.py
│   │   │   ├── router.py        # 对话路由
│   │   │   ├── service.py       # 对话管理
│   │   │   └── models.py        # 对话模型
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── router.py        # 知识库路由
│   │   │   ├── indexer.py       # 文档索引
│   │   │   └── retriever.py     # 检索器
│   │   └── database.py          # 数据库连接
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.vue
│   │   ├── main.js
│   │   ├── views/
│   │   │   ├── ChatView.vue     # 对话主界面
│   │   │   ├── KnowledgeView.vue# 知识库管理
│   │   │   ├── StatsView.vue    # 统计面板
│   │   │   └── LoginView.vue    # 登录页
│   │   ├── components/
│   │   │   ├── MessageBubble.vue# 消息气泡
│   │   │   ├── ToolTrace.vue    # 工具调用轨迹
│   │   │   ├── CostBadge.vue    # 费用显示
│   │   │   └── FileUpload.vue   # 文档上传
│   │   ├── composables/
│   │   │   ├── useWebSocket.js  # WebSocket 封装
│   │   │   └── useAuth.js       # 认证 Hook
│   │   └── stores/
│   │       └── chat.js          # Pinia 状态管理
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── .env.example
```

## 核心代码实现

### 后端：Agent 引擎

```python
# backend/app/agent/engine.py
"""Agent 核心引擎 - 基于 P2 的 Agent 循环升级版"""

import time
import anthropic
from typing import AsyncGenerator

from .tools import ToolRegistry
from .tracer import ExecutionTracer


class AgentEngine:
    """生产级 Agent 引擎"""

    MAX_ITERATIONS = 15
    MODEL = "claude-sonnet-4-20250514"

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.tool_registry = ToolRegistry()

    async def run_stream(
        self,
        user_message: str,
        conversation_history: list[dict],
        system_prompt: str = "",
        rag_context: str = "",
    ) -> AsyncGenerator[dict, None]:
        """
        流式运行 Agent，通过 yield 输出每个事件。

        事件类型:
        - {"type": "text", "content": "..."}          文本输出
        - {"type": "tool_call", "name": "...", ...}   工具调用
        - {"type": "tool_result", "name": "...", ...}  工具结果
        - {"type": "usage", "input_tokens": N, ...}    Token 统计
        - {"type": "done", "full_response": "..."}     完成
        """
        messages = list(conversation_history)

        # 如果有 RAG 上下文，注入到用户消息中
        if rag_context:
            enhanced_message = (
                f"参考资料：\n{rag_context}\n\n用户问题：{user_message}"
            )
        else:
            enhanced_message = user_message

        messages.append({"role": "user", "content": enhanced_message})

        tracer = ExecutionTracer()
        total_input_tokens = 0
        total_output_tokens = 0
        full_response = ""

        for iteration in range(self.MAX_ITERATIONS):
            tracer.start_step(f"LLM Call #{iteration + 1}")

            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=4096,
                system=system_prompt or "你是一个功能强大的 AI 助手，可以使用工具来完成任务。",
                tools=self.tool_registry.get_schemas(),
                messages=messages,
            )

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # 处理文本内容
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        full_response += block.text
                        yield {"type": "text", "content": block.text}

                messages.append({"role": "assistant", "content": response.content})
                break

            # 处理工具调用
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                # 输出思考文本
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        yield {"type": "text", "content": block.text}
                        full_response += block.text

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        yield {
                            "type": "tool_call",
                            "name": block.name,
                            "input": block.input,
                            "id": block.id,
                        }

                        # 执行工具
                        start_time = time.time()
                        result = await self.tool_registry.execute(
                            block.name, block.input
                        )
                        elapsed = time.time() - start_time

                        yield {
                            "type": "tool_result",
                            "name": block.name,
                            "result": result[:1000],  # 截断
                            "elapsed_ms": int(elapsed * 1000),
                        }

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})

        # 完成事件
        yield {
            "type": "usage",
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        }
        yield {"type": "done", "full_response": full_response}
```

### 后端：WebSocket 路由

```python
# backend/app/chat/router.py
"""对话路由 - WebSocket 实时通信"""

import json
from fastapi import APIRouter, WebSocket, Depends
from ..agent.engine import AgentEngine
from ..auth.service import get_current_user_ws
from .service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])
agent_engine = AgentEngine()
chat_service = ChatService()


@router.websocket("/ws/{conversation_id}")
async def chat_websocket(
    websocket: WebSocket,
    conversation_id: str,
    user=Depends(get_current_user_ws),
):
    """WebSocket 对话端点"""
    await websocket.accept()

    try:
        while True:
            # 接收用户消息
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            use_rag = data.get("use_rag", False)

            if not user_message:
                continue

            # 获取对话历史
            history = await chat_service.get_history(conversation_id)

            # RAG 检索（如果启用）
            rag_context = ""
            if use_rag:
                from ..rag.retriever import retriever
                results = retriever.search(user_message)
                rag_context = "\n\n".join(r["content"] for r in results[:5])

            # 流式运行 Agent
            async for event in agent_engine.run_stream(
                user_message=user_message,
                conversation_history=history,
                rag_context=rag_context,
            ):
                await websocket.send_json(event)

            # 保存到历史
            full_response = ""
            async for event in []:  # 从上面的流中已经收集
                pass
            await chat_service.save_message(
                conversation_id, "user", user_message
            )

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()


@router.get("/conversations")
async def list_conversations(user=Depends(get_current_user_ws)):
    """获取用户的对话列表"""
    return await chat_service.list_conversations(user["id"])


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, user=Depends(get_current_user_ws)):
    """获取指定对话的历史消息"""
    return await chat_service.get_full_history(conversation_id)
```

### 后端：FastAPI 主入口

```python
# backend/app/main.py
"""FastAPI 应用入口"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .auth.router import router as auth_router
from .chat.router import router as chat_router
from .rag.router import router as rag_router

app = FastAPI(
    title="AgentHub API",
    description="生产级 AI Agent 应用后端",
    version="1.0.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(rag_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
```

### 前端：对话主界面

```vue
<!-- frontend/src/views/ChatView.vue -->
<template>
  <div class="flex h-screen">
    <!-- 侧边栏：对话列表 -->
    <aside class="w-64 bg-gray-50 border-r overflow-y-auto">
      <div class="p-4">
        <button
          class="w-full py-2 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          @click="createConversation"
        >
          新对话
        </button>
      </div>
      <div
        v-for="conv in conversations"
        :key="conv.id"
        class="px-4 py-3 cursor-pointer hover:bg-gray-100"
        :class="{ 'bg-blue-50': conv.id === activeConversation }"
        @click="switchConversation(conv.id)"
      >
        <div class="text-sm font-medium truncate">{{ conv.title }}</div>
        <div class="text-xs text-gray-500">{{ conv.updated_at }}</div>
      </div>
    </aside>

    <!-- 主内容区 -->
    <main class="flex-1 flex flex-col">
      <!-- 消息列表 -->
      <div ref="messageList" class="flex-1 overflow-y-auto p-4 space-y-4">
        <MessageBubble
          v-for="msg in messages"
          :key="msg.id"
          :message="msg"
        />

        <!-- 工具调用轨迹 -->
        <ToolTrace
          v-if="currentTrace.length > 0"
          :traces="currentTrace"
        />

        <!-- 流式输出 -->
        <div v-if="streamingText" class="p-4 bg-white rounded-lg shadow-sm">
          <div class="text-sm text-gray-500 mb-1">Assistant</div>
          <div class="whitespace-pre-wrap">{{ streamingText }}</div>
          <span class="animate-pulse">|</span>
        </div>
      </div>

      <!-- 底部工具栏 -->
      <div class="border-t p-2 flex items-center gap-2 text-sm text-gray-500">
        <label class="flex items-center gap-1">
          <input v-model="useRag" type="checkbox" />
          知识库增强
        </label>
        <CostBadge :tokens="totalTokens" />
      </div>

      <!-- 输入框 -->
      <div class="border-t p-4">
        <div class="flex gap-2">
          <textarea
            v-model="inputText"
            class="flex-1 border rounded-lg p-3 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows="2"
            placeholder="输入消息... (Shift+Enter 换行)"
            @keydown.enter.exact.prevent="sendMessage"
          />
          <button
            class="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
            :disabled="!inputText.trim() || isLoading"
            @click="sendMessage"
          >
            发送
          </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { useWebSocket } from '../composables/useWebSocket'
import MessageBubble from '../components/MessageBubble.vue'
import ToolTrace from '../components/ToolTrace.vue'
import CostBadge from '../components/CostBadge.vue'

const messages = ref([])
const conversations = ref([])
const activeConversation = ref(null)
const inputText = ref('')
const streamingText = ref('')
const currentTrace = ref([])
const isLoading = ref(false)
const useRag = ref(false)
const totalTokens = ref({ input: 0, output: 0 })
const messageList = ref(null)

const { connect, send, onMessage } = useWebSocket()

onMounted(async () => {
  // 加载对话列表
  const resp = await fetch('/api/chat/conversations', {
    headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
  })
  conversations.value = await resp.json()
})

function sendMessage() {
  if (!inputText.value.trim() || isLoading.value) return

  const message = inputText.value.trim()
  inputText.value = ''
  isLoading.value = true
  streamingText.value = ''
  currentTrace.value = []

  // 添加用户消息
  messages.value.push({
    id: Date.now(),
    role: 'user',
    content: message,
  })

  // 通过 WebSocket 发送
  send({
    message,
    use_rag: useRag.value,
  })

  scrollToBottom()
}

// 处理 WebSocket 事件
onMessage((event) => {
  switch (event.type) {
    case 'text':
      streamingText.value += event.content
      break

    case 'tool_call':
      currentTrace.value.push({
        type: 'call',
        name: event.name,
        input: event.input,
      })
      break

    case 'tool_result':
      currentTrace.value.push({
        type: 'result',
        name: event.name,
        result: event.result,
        elapsed_ms: event.elapsed_ms,
      })
      break

    case 'usage':
      totalTokens.value.input += event.input_tokens
      totalTokens.value.output += event.output_tokens
      break

    case 'done':
      messages.value.push({
        id: Date.now(),
        role: 'assistant',
        content: event.full_response,
        traces: [...currentTrace.value],
      })
      streamingText.value = ''
      currentTrace.value = []
      isLoading.value = false
      scrollToBottom()
      break
  }
})

async function scrollToBottom() {
  await nextTick()
  if (messageList.value) {
    messageList.value.scrollTop = messageList.value.scrollHeight
  }
}
</script>
```

### 前端：执行轨迹组件

```vue
<!-- frontend/src/components/ToolTrace.vue -->
<template>
  <div class="border border-yellow-200 rounded-lg bg-yellow-50 p-3">
    <div class="text-xs font-medium text-yellow-700 mb-2">
      Agent 执行轨迹
    </div>
    <div class="space-y-2">
      <div
        v-for="(trace, index) in traces"
        :key="index"
        class="text-xs"
      >
        <div v-if="trace.type === 'call'" class="flex items-center gap-1">
          <span class="text-yellow-600">--></span>
          <span class="font-mono font-medium">{{ trace.name }}</span>
          <span class="text-gray-500">({{ formatInput(trace.input) }})</span>
        </div>
        <div v-if="trace.type === 'result'" class="flex items-center gap-1 text-green-700">
          <span>&lt;--</span>
          <span class="font-mono">{{ trace.name }}</span>
          <span class="text-gray-500">{{ trace.elapsed_ms }}ms</span>
          <span class="truncate max-w-md">{{ trace.result }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  traces: { type: Array, default: () => [] },
})

function formatInput(input) {
  return JSON.stringify(input).slice(0, 80)
}
</script>
```

### Docker 部署配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://agent:agent123@postgres:5432/agenthub
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - JWT_SECRET=${JWT_SECRET}
      - CHROMA_PERSIST_DIR=/data/chroma
    volumes:
      - chroma_data:/data/chroma
    depends_on:
      - postgres

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: agent123
      POSTGRES_DB: agenthub
    volumes:
      - pg_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  pg_data:
  chroma_data:
```

后端 Dockerfile：

```dockerfile
# backend/Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

前端 Dockerfile：

```dockerfile
# frontend/Dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

## 运行方式

### 开发模式

```bash
# 后端
cd backend
uv init && uv add fastapi uvicorn anthropic chromadb pyjwt passlib python-dotenv asyncpg sqlalchemy
uv run uvicorn app.main:app --reload --port 8000

# 前端
cd frontend
npm install
npm run dev
```

### Docker 部署

```bash
# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 ANTHROPIC_API_KEY 和 JWT_SECRET

# 启动
docker compose up -d

# 访问
# 前端: http://localhost:3000
# API 文档: http://localhost:8000/docs
```

## 扩展建议

1. **SSO 集成** -- 接入企业级认证（OAuth2.0、OIDC）
2. **多模型支持** -- 除了 Claude，支持 GPT-4、Gemini 等模型
3. **插件系统** -- 动态加载工具插件，无需重启服务
4. **团队协作** -- 对话分享、知识库共享、权限管理
5. **可观测性** -- 接入 LangSmith 或自建 Tracing 系统
6. **A/B 测试** -- 不同 System Prompt 或模型的效果对比

::: tip 毕业寄语
完成这个项目，你就走完了从零到生产的 Agent 开发全过程。你现在具备了独立设计、开发、部署 AI Agent 应用的能力。接下来，去构建你自己的 Agent 产品吧。
:::

## 参考资源

- [FastAPI 官方文档](https://fastapi.tiangolo.com/) -- 后端框架完整参考
- [Vue 3 官方文档](https://vuejs.org/) -- 前端框架完整参考
- [Anthropic Streaming Guide](https://docs.anthropic.com/en/api/messages-streaming) -- 流式输出实现参考
- [Docker Compose 文档](https://docs.docker.com/compose/) -- 容器编排参考
- [WebSocket RFC 6455](https://www.rfc-editor.org/rfc/rfc6455) -- WebSocket 协议规范
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) -- LLM 应用安全指南
- [LangSmith 文档](https://docs.smith.langchain.com/) -- Agent 可观测性平台
