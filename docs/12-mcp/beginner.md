# MCP 入门：理解协议架构

::: tip 学习目标
- 理解 MCP 的诞生背景和它要解决的 N x M 问题
- 掌握 Host/Client/Server 三层架构和各自的职责
- 了解 MCP 的三大核心能力：Tools、Resources、Prompts
- 理解 MCP 与 Function Calling 的关系和区别

**学完你能做到：** 清晰解释 MCP 是什么、为什么需要它，以及它的协议架构。能运行一个最简单的 MCP Server + Client 示例。
:::

## MCP 要解决什么问题

想象这样的场景：你开发了一个连接 GitHub 的工具，它在 LangChain 中能用，但换到 CrewAI 就需要重写适配层；你写了一个数据库查询工具，在 Claude Desktop 中无法直接复用到你自己的 Agent 系统。

这就是经典的 **N x M 问题**：

```
N 个 Agent 框架 x M 个工具/数据源 = N x M 个适配器

LangChain ──┬── GitHub
             ├── Slack
             ├── Database
             └── FileSystem

CrewAI    ──┬── GitHub（重写）
             ├── Slack（重写）
             └── Database（重写）

自建 Agent ──┬── GitHub（又重写）
              └── ...
```

MCP 的解决方案：定义一个标准协议，让工具只实现一次，任何遵循 MCP 的 Agent 都能直接使用。

```
N 个 Agent/Host          M 个 MCP Server
┌─────────┐              ┌──────────────┐
│ Claude   │              │ GitHub Server│
│ Desktop  │◄──── MCP ───►│ Slack Server │
│ 自建Agent│  (标准协议)   │ DB Server    │
│ Cursor   │              │ FS Server    │
└─────────┘              └──────────────┘

只需 N + M 个实现，而非 N x M
```

## 协议架构：Host -> Client -> Server

MCP 采用三层架构，各层职责明确：

```
┌──────────────────────────────────────────┐
│                  Host                     │
│  (Claude Desktop / IDE / 自建 Agent)      │
│                                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Client A │  │ Client B │  │ Client C │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
└───────┼──────────────┼──────────────┼────────┘
        │              │              │
   ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
   │ Server A │  │ Server B │  │ Server C │
   │ (GitHub) │  │ (Slack)  │  │ (DB)     │
   └──────────┘  └──────────┘  └──────────┘
```

**Host（宿主）**：面向用户的应用程序。比如 Claude Desktop、IDE 插件、你自建的 Agent 应用。Host 管理多个 Client 的生命周期，控制安全策略和权限。

**Client（客户端）**：Host 内部的协议连接器，每个 Client 维护与一个 Server 的 1:1 连接。Client 负责协议协商（capability negotiation）、消息路由和连接生命周期管理。

**Server（服务器）**：独立的服务进程，暴露特定领域的能力。比如一个 GitHub Server 提供 Issue 查询和 PR 操作，一个 DB Server 提供 SQL 查询。Server 声明自己提供的 Tools、Resources、Prompts，处理来自 Client 的请求。

## 三大核心能力

### 1. Tools（工具）

Tools 是模型可以调用的函数，由 Server 定义，模型决定何时调用。这是 MCP 最核心的能力。

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

const server = new McpServer({ name: "demo-server", version: "1.0.0" });

server.tool("get_weather", "获取指定城市的天气信息", { city: z.string() }, async ({ city }) => ({
  content: [{ type: "text", text: `${city}的天气：晴，25 C` }],
}));

server.tool(
  "search_documents",
  "搜索文档库",
  { query: z.string(), limit: z.number().default(10) },
  async ({ query, limit }) => ({
    content: [{ type: "text", text: JSON.stringify([{ title: "文档1", snippet: "..." }]) }],
  })
);
```

::: info 工具调用流程
1. Client 调用 `tools/list` 获取可用工具列表
2. LLM 根据用户需求决定调用哪个工具
3. Client 调用 `tools/call` 执行工具
4. Server 返回工具执行结果
5. LLM 基于结果生成最终回复
:::

### 2. Resources（资源）

Resources 是 Server 暴露的数据源，类似 REST API 的 GET 端点。与 Tools 的区别在于：Tools 由模型决定何时调用，Resources 由应用程序决定何时获取。

```typescript
server.resource("file", "file://{path}", async (uri) => {
  const path = uri.pathname;
  const content = await import("fs").then((fs) => fs.readFileSync(path, "utf-8"));
  return { contents: [{ uri: uri.href, text: content }] };
});

server.resource("user", "db://users/{user_id}", async (uri) => {
  const userId = uri.pathname.split("/").pop();
  // const user = await db.query(`SELECT * FROM users WHERE id = ${userId}`);
  return { contents: [{ uri: uri.href, text: JSON.stringify({ id: userId }) }] };
});
```

### 3. Prompts（提示模板）

Prompts 是预定义的提示模板，方便用户快速复用常见的交互模式。

```typescript
server.prompt("code_review", "代码审查提示模板", { code: z.string(), language: z.string().default("typescript") }, ({ code, language }) => ({
  messages: [
    {
      role: "user",
      content: {
        type: "text",
        text: `请审查以下 ${language} 代码，关注：
1. 代码质量和可读性
2. 潜在的 bug
3. 性能问题
4. 安全风险

代码：
\`\`\`${language}
${code}
\`\`\``,
      },
    },
  ],
}));
```

## MCP 与 Function Calling 的关系

很多人会问：MCP 和 OpenAI/Anthropic 的 Function Calling 有什么区别？

简单说：**Function Calling 是模型层的特性，MCP 是应用层的协议**。

- **Function Calling** 解决的是"模型如何表达工具调用意图"——模型决定调用哪个函数，返回结构化的参数。
- **MCP** 解决的是"工具如何标准化接入"——工具如何被发现、描述、调用和返回结果。

```
┌─────────────────────────────────────────┐
│            应用层 (MCP)                  │
│  工具发现、连接管理、权限控制、传输      │
├─────────────────────────────────────────┤
│         模型层 (Function Calling)        │
│  模型决策、参数生成、结果理解            │
├─────────────────────────────────────────┤
│            传输层 (HTTP/stdio)           │
│  消息序列化、连接维护                    │
└─────────────────────────────────────────┘
```

MCP 使用 Function Calling 作为底层机制之一，但提供了完整的生态层协议。

## 传输层

MCP 支持三种传输方式：

| 传输方式 | 适用场景 | 特点 |
|---------|---------|------|
| stdio | 本地开发、CLI 工具、桌面应用 | 最简单，Server 作为子进程 |
| SSE (Server-Sent Events) | 远程服务（旧版） | HTTP + 服务端推送 |
| Streamable HTTP | 远程服务（推荐） | 单一 HTTP 端点，协议最新推荐方式 |

::: warning 传输方式选择建议
- **本地集成**（Claude Desktop、IDE）：用 stdio，简单可靠
- **远程部署**：用 Streamable HTTP，是协议推荐的现代方案
- **已有 SSE 实现**：继续使用，但新项目建议用 Streamable HTTP
:::

## 协议消息格式

MCP 基于 JSON-RPC 2.0：

```json
// 请求
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": { "city": "北京" }
  }
}

// 响应
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "北京的天气：晴，25 C"
      }
    ]
  }
}
```

## 最小示例：MCP Server + Client

```typescript
// ===== server.ts =====
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "hello-server", version: "1.0.0" });

server.tool("greet", "向用户打招呼", { name: z.string() }, async ({ name }) => ({
  content: [{ type: "text", text: `你好, ${name}! 欢迎使用 MCP。` }],
}));

server.resource("version", "info://version", async () => ({
  contents: [{ uri: "info://version", text: "1.0.0" }],
}));

const transport = new StdioServerTransport();
await server.connect(transport); // 默认 stdio 传输
```

```typescript
// ===== client.ts =====
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

async function main() {
  const transport = new StdioClientTransport({
    command: "npx",
    args: ["tsx", "server.ts"],
  });

  const client = new Client({ name: "hello-client", version: "1.0.0" });
  await client.connect(transport);

  // 发现工具
  const { tools } = await client.listTools();
  console.log(`可用工具: ${tools.map((t) => t.name)}`);

  // 调用工具
  const result = await client.callTool({ name: "greet", arguments: { name: "开发者" } });
  console.log(`结果: ${(result.content as Array<{ type: string; text: string }>)[0].text}`);

  await client.close();
}

main();
```

运行这个例子你就能看到 MCP 的核心流程：Server 声明能力 -> Client 发现能力 -> Client 调用工具 -> Server 返回结果。

## 小结

- MCP 是 AI Agent 生态的"USB 接口"，将 N x M 问题简化为 N + M
- 三层架构：Host（应用）-> Client（连接器）-> Server（能力提供者）
- 三大能力：Tools（模型调用的函数）、Resources（数据源）、Prompts（模板）
- MCP 是应用层协议，Function Calling 是模型层特性，两者互补
- 本地用 stdio，远程用 Streamable HTTP

## 练习

1. 安装 MCP TypeScript SDK（`npm install @modelcontextprotocol/sdk zod`），运行上面的 Server + Client 示例。
2. 给 hello-server 添加一个新工具：`calculate(expression: string) => string`，能计算简单的数学表达式。
3. 思考：如果你要为你的公司内部系统开发一个 MCP Server，你会暴露哪些 Tools 和 Resources？

## 参考资源

- [Model Context Protocol 官方文档](https://modelcontextprotocol.io/) -- 协议完整规范
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) -- TypeScript SDK
- [Introducing the Model Context Protocol (Anthropic Blog)](https://www.anthropic.com/news/model-context-protocol) -- 发布博客
- [MCP Specification](https://spec.modelcontextprotocol.io/) -- 协议技术规范
- [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) -- 社区 Server 合集
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) -- Python SDK
