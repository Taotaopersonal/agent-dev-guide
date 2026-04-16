# MCP 高级：生态、安全与部署

::: tip 学习目标
- 了解现有 MCP Server 生态和热门项目
- 掌握在 Claude Desktop 和 Claude Code 中使用 MCP 的配置方法
- 理解 MCP Server 的安全考虑：权限最小化、输入验证、速率限制、敏感信息保护
- 学会 MCP Server 的生产部署方案：stdio、Docker、Streamable HTTP、多 Server 编排

**学完你能做到：** 评估和选用社区已有的 MCP Server，为你自己的 Server 添加完善的安全措施，并选择合适的部署方案推向生产。
:::

## MCP Server 生态

MCP 发布以来，社区迅速构建了丰富的 Server 生态。在开发自己的 Server 之前，先看看有没有现成的能用。

### 官方 Server

| Server | 功能 | 传输方式 |
|--------|------|---------|
| `@modelcontextprotocol/server-filesystem` | 文件系统读写 | stdio |
| `@modelcontextprotocol/server-github` | GitHub API | stdio |
| `@modelcontextprotocol/server-postgres` | PostgreSQL 查询 | stdio |
| `@modelcontextprotocol/server-sqlite` | SQLite 查询 | stdio |
| `@modelcontextprotocol/server-puppeteer` | 浏览器自动化 | stdio |
| `@modelcontextprotocol/server-brave-search` | Brave 搜索 | stdio |
| `@modelcontextprotocol/server-memory` | 知识图谱记忆 | stdio |

### 社区热门 Server

```typescript
// 按领域分类的社区 MCP Server
const communityServers: Record<string, string[]> = {
  "开发工具": [
    "mcp-server-git",         // Git 操作
    "mcp-server-docker",      // Docker 管理
    "mcp-server-kubernetes",  // K8s 集群管理
  ],
  "数据与 API": [
    "mcp-server-notion",      // Notion 文档
    "mcp-server-slack",       // Slack 消息
    "mcp-server-linear",      // Linear 项目管理
    "mcp-server-jira",        // Jira 工单
  ],
  "搜索与知识": [
    "mcp-server-rag",         // RAG 检索
    "mcp-server-arxiv",       // arXiv 论文
    "mcp-server-web-search",  // 网页搜索
  ],
  "云服务": [
    "mcp-server-aws",         // AWS 管理
    "mcp-server-cloudflare",  // Cloudflare
    "mcp-server-vercel",      // Vercel 部署
  ],
};
```

::: tip 善用已有生态
开发之前先去 [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) 找找有没有现成的。不要重复造轮子。
:::

## 在 Claude Desktop 中使用 MCP

Claude Desktop 是 MCP 最主要的 Host 应用之一。配置文件在 `~/Library/Application Support/Claude/claude_desktop_config.json`（macOS）：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/me/projects"
      ]
    },
    "database": {
      "command": "npx",
      "args": ["tsx", "/path/to/db_server.ts"],
      "env": {
        "DB_PATH": "/data/app.db"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxx"
      }
    }
  }
}
```

配置完成后，Claude Desktop 会在启动时自动连接所有 Server，将 Tools 显示在聊天界面。Claude 在对话中可以自主决定何时调用工具，调用前会请求用户确认。

## 在 Claude Code 中使用 MCP

```json
// .claude/settings.json
{
  "mcpServers": {
    "jira": {
      "command": "npx",
      "args": ["-y", "mcp-server-jira"],
      "env": {
        "JIRA_URL": "https://your-org.atlassian.net",
        "JIRA_API_TOKEN": "xxx"
      }
    }
  }
}
```

::: info Claude Code 的 MCP 场景
在 Claude Code 中，MCP Server 可以让 Agent 在编码过程中直接访问 Jira 工单、查询数据库 schema、读取 Confluence 文档，实现"代码 + 上下文"的无缝协作。
:::

## 安全考虑

MCP Server 运行在用户的机器上，拥有本地文件系统和网络访问权限。安全不可妥协。

### 1. 权限最小化

只暴露必要的能力，写操作要严格限制。

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import Database from "better-sqlite3";
import { z } from "zod";

const DB_PATH = process.env.DB_PATH!;

const server = new McpServer({
  name: "safe-db",
  version: "1.0.0",
});

server.tool(
  "query",
  "执行只读 SQL 查询",
  { sql: z.string() },
  async ({ sql }) => {
    // 严格过滤写操作
    const forbidden = [
      "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
      "CREATE", "TRUNCATE", "GRANT", "REVOKE",
    ];
    const sqlUpper = sql.trim().toUpperCase();
    for (const keyword of forbidden) {
      if (sqlUpper.includes(keyword)) {
        return { content: [{ type: "text", text: `安全拒绝：不允许执行 ${keyword} 操作` }] };
      }
    }

    // 使用只读连接
    const db = new Database(DB_PATH, { readonly: true });
    // ...
  }
);
```

### 2. 输入验证

对所有外部输入做严格验证，特别是文件路径和 SQL。

```typescript
import path from "path";
import fs from "fs";

const PROJECT_ROOT = "/home/user/project";

server.tool(
  "read_file",
  "读取项目文件",
  { path: z.string() },
  async ({ path: filePath }) => {
    // 路径遍历防护
    const safePath = path.join(PROJECT_ROOT, filePath);
    const resolved = path.resolve(safePath);
    if (!resolved.startsWith(path.resolve(PROJECT_ROOT))) {
      throw new Error("路径越界：不允许访问项目目录之外的文件");
    }

    // 文件类型限制
    const allowedExtensions = new Set([".py", ".js", ".ts", ".md", ".json", ".yaml"]);
    if (!allowedExtensions.has(path.extname(resolved))) {
      throw new Error(`不允许读取 ${path.extname(resolved)} 类型的文件`);
    }

    const content = fs.readFileSync(resolved, "utf-8");
    return { content: [{ type: "text", text: content }] };
  }
);
```

### 3. 速率限制

防止滥用，特别是对外部 API 调用的工具。

```typescript
class RateLimiter {
  /** 简单的速率限制器 */
  private maxCalls: number;
  private window: number;
  private calls: Map<string, number[]> = new Map();

  constructor(maxCalls: number = 60, windowSeconds: number = 60) {
    this.maxCalls = maxCalls;
    this.window = windowSeconds;
  }

  check(toolName: string): boolean {
    const now = Date.now() / 1000;
    const timestamps = this.calls.get(toolName) ?? [];
    const filtered = timestamps.filter((t) => now - t < this.window);
    if (filtered.length >= this.maxCalls) {
      this.calls.set(toolName, filtered);
      return false;
    }
    filtered.push(now);
    this.calls.set(toolName, filtered);
    return true;
  }
}

const limiter = new RateLimiter(30, 60);

server.tool(
  "expensive_operation",
  "执行代价较高的操作",
  { query: z.string() },
  async ({ query }) => {
    if (!limiter.check("expensive_operation")) {
      return { content: [{ type: "text", text: "请求过于频繁，请稍后再试" }] };
    }
    // 正常逻辑...
  }
);
```

### 4. 敏感信息保护

工具返回的内容可能包含敏感信息，需要脱敏。

```typescript
function redactSensitive(text: string): string {
  /** 脱敏处理 */
  text = text.replace(/\b[\w.-]+@[\w.-]+\.\w+\b/g, "[EMAIL]");
  text = text.replace(/\b1[3-9]\d{9}\b/g, "[PHONE]");
  text = text.replace(/(sk-|ghp_|gho_|glpat-)[a-zA-Z0-9]+/g, "[API_KEY]");
  return text;
}

server.tool(
  "search_logs",
  "搜索日志",
  { keyword: z.string() },
  async ({ keyword }) => {
    const rawResult = doSearch(keyword);
    return { content: [{ type: "text", text: redactSensitive(rawResult) }] }; // 脱敏后返回
  }
);
```

::: warning 安全第一原则
- 所有写操作都需要明确的用户确认
- 不要在 Server 中硬编码 API Key，使用环境变量
- 对外部输入做严格验证
- 返回的内容要检查并脱敏敏感信息
:::

## 部署方案

### 方案 1：本地 stdio（入门推荐）

```bash
# 直接运行
npx tsx server.ts

# 通过 Claude Desktop 配置自动启动
```

最简单，适合个人开发使用。

### 方案 2：Docker 容器化

```dockerfile
FROM node:20-slim

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install --omit=dev

COPY server.ts .

CMD ["npx", "tsx", "server.ts"]
```

```json
{
  "mcpServers": {
    "my-server": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "my-mcp-server:latest"]
    }
  }
}
```

### 方案 3：远程部署（Streamable HTTP）

```typescript
// server.ts
import express from "express";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";

const server = new McpServer({ name: "remote-server", version: "1.0.0" });
// ... 工具定义 ...

const app = express();
app.use(express.json());

app.post("/mcp", async (req, res) => {
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
  res.on("close", () => { transport.close(); });
  await server.connect(transport);
  await transport.handleRequest(req, res);
});

app.listen(8000, "0.0.0.0", () => {
  console.log("MCP Server listening on http://0.0.0.0:8000/mcp");
});
```

```yaml
# docker-compose.yml
version: "3.8"
services:
  mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_PATH=/data/app.db
    volumes:
      - ./data:/data:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 方案 4：多 Server 编排

```typescript
/** MCP Server 编排管理器 */

import { spawn, ChildProcess } from "child_process";

interface ServerConfig {
  name: string;
  command: string;
  args: string[];
  env?: Record<string, string>;
}

class ServerOrchestrator {
  /** 管理多个 MCP Server 的生命周期 */
  private processes: Map<string, ChildProcess> = new Map();

  startServer(config: ServerConfig): void {
    /** 启动一个 Server */
    const env = { ...process.env, ...(config.env ?? {}) };
    const proc = spawn(config.command, config.args, {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.processes.set(config.name, proc);
    console.log(`[Orchestrator] 已启动 ${config.name} (PID: ${proc.pid})`);
  }

  stopAll(): void {
    /** 停止所有 Server */
    for (const [name, proc] of this.processes) {
      proc.kill("SIGTERM");
      console.log(`[Orchestrator] 已停止 ${name}`);
    }
  }

  healthCheck(): Record<string, boolean> {
    /** 检查所有 Server 状态 */
    const result: Record<string, boolean> = {};
    for (const [name, proc] of this.processes) {
      result[name] = proc.exitCode === null; // null = 仍在运行
    }
    return result;
  }
}
```

## 最佳实践总结

### Server 设计原则

```typescript
// 好的设计：职责单一、描述清晰
server.tool(
  "search_users",
  "在企业通讯录中搜索用户。支持按姓名、邮箱、工号搜索。可选按部门筛选。",
  {
    query: z.string().describe("搜索关键词（姓名、邮箱或工号）"),
    department: z.string().optional().describe("部门名称筛选（可选）"),
    limit: z.number().default(10).describe("返回结果数量上限，默认 10"),
  },
  async ({ query, department, limit }) => {
    // ...
  }
);

// 坏的设计：职责混乱、描述模糊
server.tool(
  "do_stuff",
  "做一些事情",  // LLM 无法理解应该何时调用
  {
    action: z.string(),
    data: z.string(),
  },
  async ({ action, data }) => {
    // ...
  }
);
```

### 错误处理

```typescript
import Database from "better-sqlite3";

server.tool(
  "robust_query",
  "执行 SQL 查询",
  { sql: z.string() },
  async ({ sql }) => {
    try {
      const result = executeSql(sql);
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    } catch (e) {
      if (e instanceof Database.SqliteError) {
        // 返回有帮助的错误信息，帮助 Agent 自我纠正
        return {
          content: [{
            type: "text",
            text: `SQL 语法错误：${e.message}\n\n`
              + `请检查表名和列名是否正确。`
              + `可以先用 list_tables 和 describe_table 查看结构。`,
          }],
        };
      }
      return {
        content: [{ type: "text", text: `未预期的错误：${(e as Error).constructor.name}: ${e}` }],
      };
    }
  }
);
```

## 小结

- 善用已有生态：先找社区 Server，不要重复造轮子
- 安全不可妥协：权限最小化、输入验证、速率限制、敏感信息脱敏
- 部署按需选择：本地用 stdio，远程用 Streamable HTTP，容器化保证一致性
- Server 设计要职责单一、描述清晰、错误信息有帮助
- MCP 仍在快速迭代，持续关注协议演进和新特性

## 练习

1. 在 Claude Desktop 中配置一个文件系统 MCP Server，验证 Claude 能通过对话读写文件。
2. 给你的 MCP Server 添加完整的安全措施：输入验证、速率限制、敏感信息脱敏。
3. 用 Docker 部署一个 MCP Server，配置 Streamable HTTP 传输，从另一台机器连接测试。

## 参考资源

- [MCP 官方 Server 仓库](https://github.com/modelcontextprotocol/servers) -- 官方 Server 合集
- [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) -- 社区索引
- [MCP Security Best Practices](https://modelcontextprotocol.io/docs/concepts/security) -- 官方安全指南
- [Claude Desktop MCP 配置指南](https://modelcontextprotocol.io/quickstart/user) -- 用户快速入门
- [MCP 协议安全模型](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/security/) -- 安全规范
- [Docker + MCP 部署示例](https://github.com/modelcontextprotocol/servers/tree/main/src) -- 容器化参考
