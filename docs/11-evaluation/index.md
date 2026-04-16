# 第 11 章 评估与测试

Agent 开发中最容易被忽视的一环就是评估。大多数人的开发流程是：写代码 -> 手动试几个 case -> 感觉差不多了 -> 上线。这在传统软件里勉强能混过去，但在 Agent 开发中会出大问题——因为 Agent 的输出是非确定性的，同一个输入可能给出不同的输出，你"感觉"它好使，不代表它真的稳定好使。

**评估不是上线前的一次性检查，而是贯穿整个开发周期的持续流程。** 你需要回答几个关键问题：Agent 的回答准确吗？稳定吗？快吗？贵吗？安全吗？

这一章从"为什么评估很难"开始，逐步教你：构建测试集、用 LLM 自动评分、搭建评估流水线、以及在生产环境中持续监控 Agent 的表现。

## 本章内容

| 层级 | 内容 | 你将学到 |
|------|------|---------|
| [入门篇](./beginner.md) | 为什么评估很难、评估维度、手动测试方法 | 评估的基本思维框架，离线 vs 在线评估 |
| [进阶篇](./intermediate.md) | 自动化评估流水线、LLM-as-Judge、基准测试设计 | 用 LLM 评估 LLM，可复现的回归测试体系 |
| [高级篇](./advanced.md) | 可观测性系统、Tracing、生产监控、回归测试 | LangSmith/Phoenix 接入，自建 Tracing，监控面板 |

## 参考资源

- [Anthropic: Evaluations Guide](https://docs.anthropic.com/en/docs/build-with-claude/develop-tests) -- Anthropic 官方评估指南
- [LangSmith Documentation](https://docs.smith.langchain.com/) -- LangSmith 评估平台
- [RAGAS: RAG Evaluation Framework](https://docs.ragas.io/) -- RAG 评估框架
- [Hamel Husain: Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) -- LLM 评估最佳实践
- [AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) -- Agent 基准测试论文
