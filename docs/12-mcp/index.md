# 第 12 章 MCP 协议

2024 年底，Anthropic 发布了 **MCP（Model Context Protocol）**——一个为 AI Agent 与外部工具和数据源之间建立统一连接标准的开放协议。你可以把它理解为 AI 领域的 USB-C：不管你用什么 Agent 框架，不管工具的内部实现是什么，只要双方都遵循 MCP 协议，就能即插即用。

在 MCP 之前，每个 Agent 框架都有自己的工具集成方式：LangChain 用 `@tool` 装饰器，OpenAI 用 Function Calling JSON Schema，每个 IDE 插件都要单独适配。开发者写一个 GitHub 工具，想在 Claude Desktop、Cursor、自建 Agent 中都能用，就得重写多次。这就是经典的 **N x M 问题**——N 个框架 x M 个工具 = N x M 个适配器。MCP 把它简化为 **N + M**。

这一章从 MCP 的核心概念开始，带你理解协议架构，然后亲手开发 MCP Server 和 Client，最后深入到安全实践和生产部署。

## 本章内容

| 层级 | 内容 | 你将学到 |
|------|------|---------|
| [入门篇](./beginner.md) | MCP 解决什么问题、协议架构、三大核心能力 | Host/Client/Server 三层架构，Tools/Resources/Prompts |
| [进阶篇](./intermediate.md) | 开发 MCP Server、开发 MCP Client、集成到 Agent | 用 FastMCP 快速搭建 Server，将 MCP 工具桥接到 LLM |
| [高级篇](./advanced.md) | 生态与最佳实践、安全考虑、部署方案 | 权限最小化、输入验证、Docker 部署、多 Server 编排 |

## 参考资源

- [Model Context Protocol 官方文档](https://modelcontextprotocol.io/) -- 协议完整规范和教程
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) -- Python 官方 SDK
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) -- TypeScript 官方 SDK
- [Introducing the Model Context Protocol (Anthropic Blog)](https://www.anthropic.com/news/model-context-protocol) -- MCP 发布博客
- [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) -- 社区 MCP Server 合集
- [MCP Specification](https://spec.modelcontextprotocol.io/) -- 协议技术规范
