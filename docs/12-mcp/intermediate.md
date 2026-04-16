# MCP 进阶：开发 Server 与 Client

::: tip 学习目标
- 使用 McpServer 开发完整的 MCP Server，掌握 Tools/Resources/Prompts 的定义方法
- 完成两个实战项目：飞书文档 Server 和数据库查询 Server
- 掌握 MCP Client 的连接管理和工具调用流程
- 将 MCP Client 集成到 Agent 系统中，实现完整的工具调用循环

**学完你能做到：** 用 TypeScript 开发一个功能完整的 MCP Server，编写 Client 连接并调用它，以及将 MCP 工具桥接到 Claude 的 Function Calling 中构建一个 MCP Agent。
:::

## 开发 MCP Server

MCP Server 是一个独立运行的服务进程，负责将某个领域的能力通过 MCP 协议暴露给 Agent。一个好的 Server 应该：职责单一、声明清晰、错误友好、安全可控。

### 环境准备

```bash
# 初始化项目并安装 MCP TypeScript SDK
npm init -y
npm install @modelcontextprotocol/sdk zod

# 验证安装
npx tsx -e "import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'; console.log('MCP SDK ready')"
```

### 基础工具定义

`McpServer` 是 TypeScript SDK 提供的核心 API，配合 Zod 定义参数 schema：

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "my-server",
  version: "1.0.0",
});

server.tool(
  "add",
  "将两个数字相加",
  {
    a: z.number().describe("第一个数字"),
    b: z.number().describe("第二个数字"),
  },
  async ({ a, b }) => ({
    content: [{ type: "text", text: String(a + b) }],
  })
);
```

::: info 关于工具描述
MCP SDK 使用 `server.tool()` 的第二个参数作为工具描述，Zod schema 的 `.describe()` 作为参数描述。写好这些描述直接影响 LLM 能否正确使用你的工具。
:::

### 异步工具

```typescript
server.tool(
  "fetch_url",
  "获取指定 URL 的内容",
  {
    url: z.string().describe("要获取内容的 URL 地址"),
  },
  async ({ url }) => {
    const response = await fetch(url, { redirect: "follow" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const text = await response.text();
    return {
      content: [{ type: "text", text: text.slice(0, 5000) }],
    };
  }
);
```

### 定义 Resources 和 Prompts

```typescript
import fs from "node:fs";
import path from "node:path";

server.resource(
  "app-config",
  "config://app",
  async (uri) => {
    const config = {
      version: "2.1.0",
      environment: "production",
      features: ["search", "export", "notifications"],
    };
    return {
      contents: [
        {
          uri: uri.href,
          text: JSON.stringify(config, null, 2),
          mimeType: "application/json",
        },
      ],
    };
  }
);

server.resource(
  "project-file",
  "file://{path}",
  async (uri, { path: filePath }) => {
    const projectRoot = process.env.PROJECT_ROOT || ".";
    const fullPath = path.join(projectRoot, filePath);

    // 安全检查：防止路径遍历
    const realPath = fs.realpathSync(fullPath);
    if (!realPath.startsWith(fs.realpathSync(projectRoot))) {
      throw new Error("不允许访问项目目录之外的文件");
    }

    const content = fs.readFileSync(realPath, "utf-8");
    return {
      contents: [{ uri: uri.href, text: content, mimeType: "text/plain" }],
    };
  }
);

server.prompt(
  "debug_error",
  "帮助调试错误信息",
  {
    error_message: z.string().describe("错误信息"),
    stack_trace: z.string().optional().describe("堆栈信息"),
  },
  async ({ error_message, stack_trace }) => {
    let prompt = `请帮我分析以下错误并给出修复建议：\n\n错误信息：${error_message}`;
    if (stack_trace) {
      prompt += `\n\n堆栈信息：\n${stack_trace}`;
    }
    prompt += "\n\n请分析：1. 错误原因 2. 触发条件 3. 修复方案 4. 预防措施";
    return {
      messages: [{ role: "user", content: { type: "text", text: prompt } }],
    };
  }
);
```

## 实战：数据库查询 MCP Server

一个完整的、生产可用的 MCP Server 示例——提供安全的只读数据库查询能力。

```typescript
/** 数据库查询 MCP Server —— 安全的只读 SQL 查询 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import Database from "better-sqlite3";

const server = new McpServer({
  name: "db-query",
  version: "1.0.0",
});

const DB_PATH = process.env.DB_PATH || "app.db";

function getReadonlyDb(): Database.Database {
  return new Database(DB_PATH, { readonly: true });
}

server.tool(
  "list_tables",
  "列出数据库中所有表的名称和行数",
  {},
  async () => {
    const db = getReadonlyDb();
    try {
      const tables = db
        .prepare(
          "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        .all() as { name: string }[];

      const result = tables.map((row) => {
        const count = db
          .prepare(`SELECT COUNT(*) as cnt FROM [${row.name}]`)
          .get() as { cnt: number };
        return { table: row.name, row_count: count.cnt };
      });

      return {
        content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
      };
    } finally {
      db.close();
    }
  }
);

server.tool(
  "describe_table",
  "查看表结构，包括列名、类型和示例数据",
  {
    table_name: z.string().describe("要查看的表名"),
  },
  async ({ table_name }) => {
    const db = getReadonlyDb();
    try {
      const columnsRaw = db
        .prepare(`PRAGMA table_info([${table_name}])`)
        .all() as {
        name: string;
        type: string;
        notnull: number;
        pk: number;
      }[];

      const columns = columnsRaw.map((row) => ({
        name: row.name,
        type: row.type,
        nullable: !row.notnull,
        primary_key: Boolean(row.pk),
      }));

      const sampleData = db
        .prepare(`SELECT * FROM [${table_name}] LIMIT 3`)
        .all();

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              { table: table_name, columns, sample_data: sampleData },
              null,
              2
            ),
          },
        ],
      };
    } finally {
      db.close();
    }
  }
);

server.tool(
  "query",
  "执行只读 SQL 查询。仅支持 SELECT 语句，结果限制最多 100 行。禁止 INSERT、UPDATE、DELETE、DROP 等修改操作。",
  {
    sql: z.string().describe("要执行的 SQL 查询语句"),
  },
  async ({ sql }) => {
    // 安全检查
    const sqlUpper = sql.trim().toUpperCase();
    const forbidden = [
      "INSERT",
      "UPDATE",
      "DELETE",
      "DROP",
      "ALTER",
      "CREATE",
      "TRUNCATE",
    ];
    for (const keyword of forbidden) {
      if (sqlUpper.startsWith(keyword)) {
        return {
          content: [
            {
              type: "text",
              text: `错误：不允许执行 ${keyword} 操作。本 Server 仅支持只读查询。`,
            },
          ],
        };
      }
    }

    if (!sqlUpper.startsWith("SELECT") && !sqlUpper.startsWith("WITH")) {
      return {
        content: [
          { type: "text", text: "错误：仅支持 SELECT 查询和 WITH（CTE）查询。" },
        ],
      };
    }

    // 自动添加 LIMIT
    let finalSql = sql;
    if (!sqlUpper.includes("LIMIT")) {
      finalSql = sql.replace(/;$/, "") + " LIMIT 100";
    }

    const db = getReadonlyDb();
    try {
      const rows = db.prepare(finalSql).all();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              { row_count: rows.length, data: rows },
              null,
              2
            ),
          },
        ],
      };
    } catch (e) {
      return {
        content: [
          { type: "text", text: `SQL 执行错误：${(e as Error).message}` },
        ],
      };
    } finally {
      db.close();
    }
  }
);

server.resource(
  "database-overview",
  "schema://overview",
  async (uri) => {
    const db = getReadonlyDb();
    try {
      const tables = db
        .prepare(
          "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        .all() as { name: string }[];

      const result = tables.map((row) => {
        const count = db
          .prepare(`SELECT COUNT(*) as cnt FROM [${row.name}]`)
          .get() as { cnt: number };
        return { table: row.name, row_count: count.cnt };
      });

      return {
        contents: [
          {
            uri: uri.href,
            text: JSON.stringify(result, null, 2),
            mimeType: "application/json",
          },
        ],
      };
    } finally {
      db.close();
    }
  }
);

server.prompt(
  "analyze_data",
  "数据分析提示模板",
  {
    question: z.string().describe("要分析的数据问题"),
  },
  async ({ question }) => ({
    messages: [
      {
        role: "user",
        content: {
          type: "text",
          text:
            `我想分析数据库中的数据来回答这个问题：${question}\n\n` +
            `请按以下步骤执行：\n` +
            `1. 先用 list_tables 查看有哪些表\n` +
            `2. 用 describe_table 了解相关表的结构\n` +
            `3. 编写并执行 SQL 查询\n` +
            `4. 分析查询结果并给出答案`,
        },
      },
    ],
  })
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
```

### 测试和调试

```bash
# 用 MCP Inspector 交互式调试
npx @modelcontextprotocol/inspector npx tsx server.ts
```

Inspector 提供 Web UI，可以查看工具列表、手动调用工具、浏览资源。

也可以用编程方式自动化测试：

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

async function testServer() {
  /** 自动化测试 MCP Server */
  const transport = new StdioClientTransport({
    command: "npx",
    args: ["tsx", "server.ts"],
    env: { ...process.env, DB_PATH: "test.db" },
  });

  const client = new Client({ name: "test-client", version: "1.0.0" });
  await client.connect(transport);

  // 测试工具列表
  const tools = await client.listTools();
  const toolNames = tools.tools.map((t) => t.name);
  console.assert(toolNames.includes("list_tables"));
  console.assert(toolNames.includes("query"));
  console.log(`[PASS] 工具列表: ${toolNames}`);

  // 测试安全拦截
  const result = await client.callTool({
    name: "query",
    arguments: { sql: "DROP TABLE users" },
  });
  const text = (result.content as { type: string; text: string }[])[0].text;
  console.assert(text.includes("不允许"));
  console.log(`[PASS] 危险操作被正确拦截`);

  console.log("\n所有测试通过!");
  await client.close();
}

testServer().catch(console.error);
```

## 开发 MCP Client

MCP Client 是 Host 和 Server 之间的桥梁。核心任务：管理连接、发现工具、转发调用。

### 连接方式

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

// stdio 连接（本地）
async function connectStdio() {
  const transport = new StdioClientTransport({
    command: "npx",
    args: ["tsx", "server.ts"],
    env: { ...process.env, DB_PATH: "/data/app.db" },
  });
  const client = new Client({ name: "my-client", version: "1.0.0" });
  await client.connect(transport);
  const tools = await client.listTools();
  console.log(`发现 ${tools.tools.length} 个工具`);
  await client.close();
}

// SSE 连接（远程）
async function connectSSE() {
  const transport = new SSEClientTransport(
    new URL("http://localhost:8000/sse")
  );
  const client = new Client({ name: "my-client", version: "1.0.0" });
  await client.connect(transport);
  console.log("SSE 连接成功！");
  await client.close();
}

// Streamable HTTP 连接（远程，推荐）
async function connectStreamableHTTP() {
  const transport = new StreamableHTTPClientTransport(
    new URL("http://localhost:8000/mcp")
  );
  const client = new Client({ name: "my-client", version: "1.0.0" });
  await client.connect(transport);
  console.log("Streamable HTTP 连接成功！");
  await client.close();
}
```

### 工具发现和调用

```typescript
async function useServer(client: Client) {
  // 发现能力
  const tools = await client.listTools();
  for (const tool of tools.tools) {
    console.log(`工具: ${tool.name} - ${tool.description}`);
  }

  const resources = await client.listResources();
  for (const resource of resources.resources) {
    console.log(`资源: ${resource.uri} - ${resource.name}`);
  }

  // 调用工具
  const result = await client.callTool({
    name: "query",
    arguments: { sql: "SELECT * FROM users LIMIT 5" },
  });
  for (const contentItem of result.content as {
    type: string;
    text?: string;
  }[]) {
    if (contentItem.type === "text") {
      console.log(`结果: ${contentItem.text}`);
    }
  }

  // 读取资源
  const resource = await client.readResource({
    uri: "schema://overview",
  });
  for (const content of resource.contents) {
    if ("text" in content) {
      console.log(`概览: ${content.text}`);
    }
  }
}
```

## 将 MCP 集成到 Agent

这是最关键的部分——将 MCP Server 的工具桥接到 LLM 的 Function Calling 中。

```typescript
/** 完整的 MCP Agent：将 MCP 工具集成到 Claude Agent 中 */

import Anthropic from "@anthropic-ai/sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

interface ToolDef {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

class MCPAgent {
  /** 集成 MCP 的 AI Agent */

  private model: string;
  private anthropic: Anthropic;
  private sessions: Map<string, Client> = new Map();
  private tools: ToolDef[] = [];
  private toolToSession: Map<string, Client> = new Map();

  constructor(model = "claude-sonnet-4-20250514") {
    this.model = model;
    this.anthropic = new Anthropic();
  }

  async connectServer(
    name: string,
    command: string,
    args: string[],
    env?: Record<string, string>
  ): Promise<void> {
    /** 连接一个 MCP Server 并注册其工具 */
    const transport = new StdioClientTransport({
      command,
      args,
      env: { ...process.env, ...env } as Record<string, string>,
    });

    const client = new Client({ name: `agent-${name}`, version: "1.0.0" });
    await client.connect(transport);

    this.sessions.set(name, client);

    // 获取工具并转换为 Anthropic API 格式
    const toolsResult = await client.listTools();
    for (const tool of toolsResult.tools) {
      this.tools.push({
        name: tool.name,
        description: tool.description || "",
        input_schema: tool.inputSchema as Record<string, unknown>,
      });
      this.toolToSession.set(tool.name, client);
    }

    console.log(
      `[MCP] 已连接 '${name}'，注册 ${toolsResult.tools.length} 个工具`
    );
  }

  private async executeTool(
    toolName: string,
    toolInput: Record<string, unknown>
  ): Promise<string> {
    /** 通过 MCP 执行工具调用 */
    const client = this.toolToSession.get(toolName);
    if (!client) {
      return `错误：未知工具 '${toolName}'`;
    }

    try {
      const result = await client.callTool({
        name: toolName,
        arguments: toolInput,
      });
      const texts = (
        result.content as { type: string; text?: string }[]
      )
        .filter((c) => c.type === "text" && c.text)
        .map((c) => c.text!);
      return texts.length > 0 ? texts.join("\n") : "工具执行完成，无文本输出";
    } catch (e) {
      return `工具执行失败：${(e as Error).message}`;
    }
  }

  async chat(userMessage: string): Promise<string> {
    /** 与 Agent 对话（包含 MCP 工具调用循环） */
    const messages: Anthropic.MessageParam[] = [
      { role: "user", content: userMessage },
    ];

    while (true) {
      const response = await this.anthropic.messages.create({
        model: this.model,
        max_tokens: 4096,
        tools: this.tools.length > 0 ? (this.tools as Anthropic.Tool[]) : undefined,
        messages,
      });

      if (response.stop_reason === "tool_use") {
        messages.push({ role: "assistant", content: response.content });

        const toolResults: Anthropic.ToolResultBlockParam[] = [];
        for (const block of response.content) {
          if (block.type === "tool_use") {
            console.log(
              `[Agent] 调用: ${block.name}(${JSON.stringify(block.input)})`
            );
            const result = await this.executeTool(
              block.name,
              block.input as Record<string, unknown>
            );
            console.log(`[Agent] 结果: ${result.slice(0, 200)}...`);
            toolResults.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: result,
            });
          }
        }

        messages.push({ role: "user", content: toolResults });
      } else {
        // 提取最终文本回复
        return response.content
          .filter(
            (b): b is Anthropic.TextBlock => b.type === "text"
          )
          .map((b) => b.text)
          .join("");
      }
    }
  }

  async cleanup(): Promise<void> {
    /** 清理所有连接 */
    for (const client of this.sessions.values()) {
      try {
        await client.close();
      } catch {
        // ignore
      }
    }
    this.sessions.clear();
  }
}

// 使用示例
async function main() {
  const agent = new MCPAgent();
  try {
    await agent.connectServer(
      "database",
      "npx",
      ["tsx", "db_server.ts"],
      { DB_PATH: "app.db" }
    );
    const response = await agent.chat(
      "请查看数据库有哪些表，然后查询最近注册的 5 个用户"
    );
    console.log(`\n[回复]\n${response}`);
  } finally {
    await agent.cleanup();
  }
}

main().catch(console.error);
```

::: warning 连接生命周期管理
上面的示例为了简洁直接在 `connectServer` 中管理 transport 和 client。在生产环境中，建议统一管理所有 Client 实例的生命周期，确保在进程退出时正确调用 `client.close()` 释放资源。
:::

## 工具 Schema 转换

不同 LLM 的 Function Calling 格式略有差异：

```typescript
import type { Tool as McpTool } from "@modelcontextprotocol/sdk/types.js";

function mcpToolToOpenAI(tool: McpTool): Record<string, unknown> {
  /** 将 MCP 工具转换为 OpenAI 格式 */
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description || "",
      parameters: tool.inputSchema,
    },
  };
}

function mcpToolToAnthropic(tool: McpTool): Record<string, unknown> {
  /** 将 MCP 工具转换为 Anthropic 格式 */
  return {
    name: tool.name,
    description: tool.description || "",
    input_schema: tool.inputSchema,
  };
}
```

## 小结

- 用 `McpServer` 快速搭建 Server，通过 `server.tool()` / `server.resource()` / `server.prompt()` 方法定义能力
- 工具的描述字符串和 Zod schema 至关重要——LLM 依赖它们理解工具能力
- 安全第一：只读连接、输入验证、路径检查、操作白名单
- Client 端核心流程：连接 -> 发现工具 -> 转换为 LLM 格式 -> Agent 循环中调用
- 一个 Agent 可以同时连接多个 MCP Server，聚合不同领域的能力

## 练习

1. 开发一个"文件搜索" MCP Server：提供 `list_files`、`read_file`、`search_content` 三个工具。
2. 用 MCP Inspector 测试你的 Server，确认每个工具都能正确返回结果。
3. 将你的 Server 集成到 MCPAgent 中，让 Claude 能通过对话来搜索和读取文件。

## 参考资源

- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) -- 官方 TypeScript SDK
- [MCP Server 开发指南](https://modelcontextprotocol.io/docs/concepts/servers) -- Server 开发文档
- [MCP Client 开发文档](https://modelcontextprotocol.io/docs/concepts/clients) -- Client 概念文档
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector) -- 官方调试工具
- [Anthropic Tool Use 文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) -- Claude Function Calling
- [Building a MCP Client (Tutorial)](https://modelcontextprotocol.io/quickstart/client) -- Client 快速入门
