# 第 10 章 多 Agent 系统

当单个 Agent 遇到跨领域、多视角、大规模的任务时，它的局限性就暴露了：一个 prompt 很难同时扮演研究员、分析师和写手。**多 Agent 系统**的核心思路很直接——让专业的 Agent 做专业的事，然后把它们的成果协调整合起来。

这和现实世界的团队协作是一回事。你不会让一个人同时写代码、做测试、写文档，而是分工协作。多 Agent 系统就是把这种分工模式搬到了 AI 世界：每个 Agent 有自己的角色定义、专属工具和上下文，通过某种协调机制（监督者、消息总线、共享黑板）来完成复杂任务。

当然，多 Agent 不是越多越好。Agent 之间的通信有成本，协调有复杂度，分歧需要解决。这一章会帮你搞清楚：什么时候该用多 Agent，怎么选架构模式，以及如何处理 Agent 之间的通信、调度和冲突。

## 本章内容

| 层级 | 内容 | 你将学到 |
|------|------|---------|
| [入门篇](./beginner.md) | 为什么需要多 Agent、五种基本架构模式 | 各模式的适用场景，Supervisor 和 Pipeline 的完整实现 |
| [进阶篇](./intermediate.md) | Agent 间通信、任务分解与调度、冲突解决与共识 | 消息总线、黑板模式、事件驱动、依赖管理、投票和仲裁机制 |
| [高级篇](./advanced.md) | 大规模编排、动态 Agent 生成、Agent 社会模拟 | 异步调度器、Agent 工厂、多 Agent 辩论与涌现行为 |

## 参考资源

- [LangGraph: Multi-Agent Architectures](https://langchain-ai.github.io/langgraph/concepts/multi_agent/) -- LangGraph 多 Agent 文档
- [AutoGen: Multi-Agent Conversation Framework](https://microsoft.github.io/autogen/) -- Microsoft 多 Agent 框架
- [CrewAI Documentation](https://docs.crewai.com/) -- CrewAI 多角色协作框架
- [MetaGPT (arXiv:2308.00352)](https://arxiv.org/abs/2308.00352) -- MetaGPT 多 Agent 论文
- [Andrew Ng: Multi-Agent Systems](https://www.youtube.com/watch?v=sal78ACtGTc) -- Andrew Ng 讲解多 Agent
