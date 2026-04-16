# 第 9 章 Agent 框架

当你从零手写 Agent 原型跑通之后，下一步就会面临一个经典问题：**继续手写，还是上框架？** 这一章帮你系统地回答这个问题，并带你走进主流框架的世界。

Agent 框架本质上是对「LLM + 工具调用 + 状态管理 + 可观测性」这套组合拳的封装。手写 50 行就能跑一个基本循环，但当你需要处理错误重试、多步工作流、状态持久化、Tracing 和成本追踪时，框架能帮你省下大量重复劳动。

当前主流的选择包括：**LangChain** 提供了最大的社区生态和丰富的集成；**LangGraph** 用状态图精确控制复杂工作流；**CrewAI** 让多角色协作变得直观；**Anthropic Agent SDK** 则走极简路线，保持最大灵活性。选择哪个取决于你的任务复杂度、团队偏好和性能要求。

当然，当现有框架无法满足你的生产需求时，**自建框架**也是一条可行的路。理解框架背后的核心抽象（Model Provider、Tool Registry、Memory Manager、Middleware Pipeline），你就能用 500 行代码搭出一个完全可控的轻量框架。

## 本章内容

| 层级 | 内容 | 你将学到 |
|------|------|---------|
| [入门篇](./beginner.md) | 为什么需要框架、选型指南、LangChain 核心概念 | 框架 vs 手写的决策方法，LangChain 快速上手 |
| [进阶篇](./intermediate.md) | LangGraph 工作流、CrewAI 多角色、Anthropic Agent SDK | 三大框架的实战使用和对比 |
| [高级篇](./advanced.md) | 框架源码分析、自建框架设计与完整实现 | 从原理到 500 行代码的 Mini Agent Framework |

## 参考资源

- [LangChain Documentation](https://python.langchain.com/docs/) -- LangChain 官方文档
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) -- LangGraph 官方文档
- [CrewAI Documentation](https://docs.crewai.com/) -- CrewAI 官方文档
- [Anthropic Agent SDK](https://github.com/anthropics/anthropic-sdk-python) -- Anthropic Python SDK
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) -- OpenAI 官方 Agent 框架
